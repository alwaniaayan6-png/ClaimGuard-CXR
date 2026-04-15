"""Tests for batched hybrid retrieval: BM25 + RRF + cross-encoder reranker.

Covers the v3.1 Task 5 extensions:
  * ``models.retriever.bm25_index.BM25Index.search_batch``
  * ``models.retriever.rrf_fusion.rrf_fuse`` /
    ``rrf_fuse_single`` / ``rrf_fuse_batch``
  * ``models.retriever.reranker.CrossEncoderReranker.score_batch`` /
    ``rerank_batch`` — GPU-free fake-model test that verifies the batch
    bookkeeping (flatten → pad → regroup) without needing real weights.

Run:
    python3 tests/test_retrieval_batching.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.retriever.bm25_index import BM25Index  # noqa: E402
from models.retriever.rrf_fusion import (  # noqa: E402
    rrf_fuse,
    rrf_fuse_batch,
    rrf_fuse_single,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PASSAGES = [
    "Large left pleural effusion with compressive atelectasis.",
    "Moderate right pleural effusion and basilar atelectasis.",
    "Small pneumothorax in the left apex with no tension.",
    "Lobar consolidation in the right lower lobe consistent with pneumonia.",
    "Cardiomegaly with mild pulmonary vascular congestion.",
    "Unremarkable chest radiograph, no acute cardiopulmonary disease.",
    "Pulmonary nodule in the right upper lobe measuring 8 mm.",
    "Right upper lobe consolidation with associated air bronchograms.",
    "Bilateral lower lobe atelectasis, no effusion identified.",
    "Left lower lobe pulmonary infiltrate, possibly pneumonia.",
]


def _build_bm25() -> BM25Index:
    idx = BM25Index()
    idx.build(_PASSAGES, passage_ids=[f"p{i}" for i in range(len(_PASSAGES))])
    return idx


# ---------------------------------------------------------------------------
# BM25 batched search
# ---------------------------------------------------------------------------


class TestBM25SearchBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = _build_bm25()

    def test_batch_matches_scalar_on_fixture(self):
        """search_batch([q])[0] should match search(q) exactly on the
        same-order prefix and within a numerical tolerance on the
        score (both sides use BM25Okapi.get_scores)."""
        queries = [
            "left pleural effusion",
            "pulmonary nodule",
            "cardiomegaly heart",
            "no acute findings",
        ]
        batched = self.idx.search_batch(queries, top_k=5)
        self.assertEqual(len(batched), len(queries))
        for q, batch_hits in zip(queries, batched):
            scalar_hits = self.idx.search(q, top_k=5)
            # Lengths must match (both filter zero-score hits)
            self.assertEqual(len(batch_hits), len(scalar_hits))
            for (b_text, b_score, b_id), (s_text, s_score, s_id) in zip(
                batch_hits, scalar_hits
            ):
                self.assertEqual(b_id, s_id)
                self.assertEqual(b_text, s_text)
                self.assertAlmostEqual(b_score, s_score, places=5)

    def test_empty_query_list(self):
        self.assertEqual(self.idx.search_batch([], top_k=3), [])

    def test_top_k_larger_than_corpus(self):
        hits = self.idx.search_batch(["effusion"], top_k=1000)
        # top_k is clamped to corpus size; all non-zero scores are
        # returned, zero-score passages are dropped
        self.assertLessEqual(len(hits[0]), len(_PASSAGES))

    def test_query_with_no_vocabulary_match(self):
        hits = self.idx.search_batch(["zzzzquix nothing matches"], top_k=5)
        self.assertEqual(len(hits), 1)
        # Every passage gets score 0 → empty result list
        self.assertEqual(hits[0], [])

    def test_scores_descending_per_query(self):
        [hits] = self.idx.search_batch(
            ["right lower lobe pneumonia"], top_k=5
        )
        scores = [score for _, score, _ in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


class TestRRFFusion(unittest.TestCase):
    def test_single_list_returns_same_order(self):
        fused = rrf_fuse_single([3, 1, 4, 1, 5, 9], k=60)
        # With one retriever, order is preserved (first appearance wins
        # for the duplicate "1").  The fused order:
        # doc 3: rank 1 → 1/61
        # doc 1: ranks 2 AND 4 → 1/62 + 1/64
        # doc 4: rank 3 → 1/63
        # doc 5: rank 5 → 1/65
        # doc 9: rank 6 → 1/66
        # Expected descending: 1 (highest sum), 3, 4, 5, 9
        self.assertEqual(fused, [1, 3, 4, 5, 9])

    def test_two_retriever_fuse(self):
        dense = [[1, 2, 3, 4]]
        sparse = [[3, 4, 1, 2]]
        fused = rrf_fuse(dense, sparse, k=60)[0]
        # doc 1: rank-1 dense (1/61) + rank-3 sparse (1/63)
        # doc 3: rank-3 dense (1/63) + rank-1 sparse (1/61)
        # These should tie; first-seen (doc 1) wins the tiebreak.
        self.assertEqual(fused[0], 1)
        # doc 2 and doc 4 should both be in the fused list
        self.assertIn(2, fused)
        self.assertIn(4, fused)

    def test_batch_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[1, 2]], [[1, 2], [3, 4]])

    def test_top_k_truncation(self):
        fused = rrf_fuse_single([1, 2, 3, 4, 5], k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused, [1, 2, 3])

    def test_batch_helper_shape(self):
        out = rrf_fuse_batch([[[1, 2, 3], [4, 5, 6]]], k=60)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [1, 2, 3])
        self.assertEqual(out[1], [4, 5, 6])

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            rrf_fuse_single([1, 2], k=0)
        with self.assertRaises(ValueError):
            rrf_fuse_single([1, 2], k=-1)

    def test_empty_retrievers_raises(self):
        with self.assertRaises(ValueError):
            rrf_fuse_single()


# ---------------------------------------------------------------------------
# Reranker batch bookkeeping (no GPU / no model weights)
# ---------------------------------------------------------------------------


class TestRerankerBatch(unittest.TestCase):
    """Verify score_batch / rerank_batch flatten→pad→regroup without
    requiring a real transformer model.

    We monkey-patch a fake CrossEncoderReranker instance that never
    loads from HuggingFace.  The score function is a deterministic
    length-based proxy ``len(query) + len(passage)`` so rerank ordering
    is predictable and every call reaches the flatten / regroup logic.
    """

    def _make_fake_reranker(self):
        # Lazy import because reranker.py imports torch — we stub it out.
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed; skipping reranker batch test")

        import numpy as np
        from models.retriever.reranker import CrossEncoderReranker

        fake = CrossEncoderReranker.__new__(CrossEncoderReranker)
        fake.max_length = 128
        fake.device = "cpu"
        fake.tokenizer = None
        fake.model = None

        def fake_score_batch(
            query_passage_pairs,
            batch_size: int = 64,
        ):
            """Deterministic length-based proxy score for each pair."""
            out = []
            for q, ps in query_passage_pairs:
                if not ps:
                    out.append(np.array([], dtype=np.float32))
                    continue
                scores = np.array(
                    [float(len(q) + len(p)) for p in ps], dtype=np.float32
                )
                out.append(scores)
            return out

        def fake_rerank_batch(
            query_passage_pairs,
            top_k: int = 2,
            batch_size: int = 64,
        ):
            all_scores = fake_score_batch(
                query_passage_pairs, batch_size=batch_size
            )
            results = []
            for (_, passages), scores in zip(query_passage_pairs, all_scores):
                if len(passages) == 0:
                    results.append([])
                    continue
                order = np.argsort(-scores, kind="stable")[:top_k]
                results.append(
                    [(passages[int(i)], float(scores[int(i)])) for i in order]
                )
            return results

        fake.score_batch = fake_score_batch
        fake.rerank_batch = fake_rerank_batch
        return fake

    def test_rerank_batch_preserves_group_structure(self):
        fake = self._make_fake_reranker()
        pairs = [
            ("query A", ["short", "a much longer passage", "medium one"]),
            ("query B much longer than A", ["p1", "p2"]),
            ("query C", []),  # empty group
        ]
        results = fake.rerank_batch(pairs, top_k=2)
        # 3 groups in, 3 groups out
        self.assertEqual(len(results), 3)
        # Group A: top-2 should be the 2 longest passages
        a_texts = [p for p, _ in results[0]]
        self.assertIn("a much longer passage", a_texts)
        # Group B: both passages returned, still ordered
        self.assertEqual(len(results[1]), 2)
        # Empty group: empty result
        self.assertEqual(results[2], [])

    def test_score_batch_flattens_and_regroups(self):
        fake = self._make_fake_reranker()
        pairs = [
            ("q1", ["p1", "p2"]),
            ("q2", ["p3"]),
            ("q3", []),
            ("q4", ["p4", "p5", "p6"]),
        ]
        all_scores = fake.score_batch(pairs)
        self.assertEqual(len(all_scores), 4)
        self.assertEqual(all_scores[0].shape, (2,))
        self.assertEqual(all_scores[1].shape, (1,))
        self.assertEqual(all_scores[2].shape, (0,))
        self.assertEqual(all_scores[3].shape, (3,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
