"""Tests for ``data.augmentation.causal_term_identifier``.

Focuses on the pure helpers that do not need captum / torch:

* ``CausalSpan`` — dataclass immutability / defaults / hashability
* ``score_to_spans`` — top-k extraction from per-token scores
* ``merge_contiguous_spans`` — adjacency merging with whitespace joins
* ``split_tokens_by_sep`` — RoBERTa SEP-based claim/evidence labeling

The heavy ``CausalTermIdentifier`` class is smoke-tested only for the
ImportError path (captum unavailable) — full integration needs a
trained v3 checkpoint and is exercised from the Modal DPO data-prep
pipeline, not the unit test suite.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.augmentation.causal_term_identifier import (  # noqa: E402
    CausalSpan,
    CausalTermIdentifier,
    merge_contiguous_spans,
    score_to_spans,
    split_tokens_by_sep,
)


# ---------------------------------------------------------------------------
# CausalSpan dataclass
# ---------------------------------------------------------------------------


class TestCausalSpan(unittest.TestCase):
    def _span(self, **overrides: object) -> CausalSpan:
        base = dict(
            text="heart",
            source="claim",
            score=0.5,
            start_char=0,
            end_char=5,
        )
        base.update(overrides)
        return CausalSpan(**base)  # type: ignore[arg-type]

    def test_default_token_indices_is_empty_tuple(self) -> None:
        s = self._span()
        self.assertEqual(s.token_indices, ())

    def test_frozen_rejects_mutation(self) -> None:
        s = self._span()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.score = 0.9  # type: ignore[misc]

    def test_equality_is_structural(self) -> None:
        a = self._span(score=0.5)
        b = self._span(score=0.5)
        self.assertEqual(a, b)

    def test_inequality_on_score(self) -> None:
        a = self._span(score=0.5)
        b = self._span(score=0.6)
        self.assertNotEqual(a, b)

    def test_hashable(self) -> None:
        # frozen=True dataclasses are hashable — this should not raise.
        a = self._span()
        b = self._span()
        self.assertEqual({a, b}, {a})


# ---------------------------------------------------------------------------
# score_to_spans — top-k extraction from per-token attribution scores
# ---------------------------------------------------------------------------


class TestScoreToSpans(unittest.TestCase):
    def test_top_k_sorted_by_score(self) -> None:
        text = "heart is enlarged"
        tokens = ["<s>", "heart", " is", " enlarged", "</s>"]
        offsets = [(0, 0), (0, 5), (5, 8), (8, 17), (0, 0)]
        scores = [0.05, 0.7, 0.1, 0.9, 0.05]
        source_ids = ["claim"] * 5
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": text, "evidence": ""},
            top_k=3,
        )
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0].text, " enlarged")
        self.assertAlmostEqual(spans[0].score, 0.9)
        self.assertEqual(spans[1].text, "heart")
        self.assertAlmostEqual(spans[1].score, 0.7)
        self.assertEqual(spans[2].text, " is")

    def test_special_tokens_filtered_by_zero_offsets(self) -> None:
        # CLS/SEP tokens carry (0, 0) offsets.  Even if the raw
        # attribution score is huge, they must not appear in the output.
        tokens = ["<s>", "heart", "</s>"]
        offsets = [(0, 0), (0, 5), (0, 0)]
        scores = [10.0, 0.5, 10.0]
        source_ids = ["claim"] * 3
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": "heart", "evidence": ""},
            top_k=5,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "heart")

    def test_negative_scores_scored_by_absolute_value(self) -> None:
        # IG attributions can be negative; we score by |attribution|.
        tokens = ["a", "b"]
        offsets = [(0, 1), (1, 2)]
        scores = [-0.9, 0.1]
        source_ids = ["claim", "claim"]
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": "ab", "evidence": ""},
            top_k=2,
        )
        self.assertEqual(spans[0].text, "a")
        self.assertAlmostEqual(spans[0].score, 0.9)

    def test_whitespace_only_text_filtered(self) -> None:
        tokens = ["<s>", " ", "x"]
        offsets = [(0, 0), (0, 1), (1, 2)]
        scores = [0.0, 0.9, 0.5]
        source_ids = ["claim"] * 3
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": " x", "evidence": ""},
            top_k=3,
        )
        # The whitespace-only span " " is filtered; only "x" remains.
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "x")

    def test_top_k_zero_returns_empty_list(self) -> None:
        spans = score_to_spans(
            tokens=["a"],
            offsets=[(0, 1)],
            scores=[0.5],
            source_ids=["claim"],  # type: ignore[arg-type]
            source_texts={"claim": "a", "evidence": ""},
            top_k=0,
        )
        self.assertEqual(spans, [])

    def test_negative_top_k_returns_empty(self) -> None:
        spans = score_to_spans(
            tokens=["a"],
            offsets=[(0, 1)],
            scores=[0.5],
            source_ids=["claim"],  # type: ignore[arg-type]
            source_texts={"claim": "a", "evidence": ""},
            top_k=-1,
        )
        self.assertEqual(spans, [])

    def test_top_k_larger_than_candidates(self) -> None:
        spans = score_to_spans(
            tokens=["a"],
            offsets=[(0, 1)],
            scores=[0.5],
            source_ids=["claim"],  # type: ignore[arg-type]
            source_texts={"claim": "a", "evidence": ""},
            top_k=10,
        )
        self.assertEqual(len(spans), 1)

    def test_length_mismatch_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            score_to_spans(
                tokens=["a", "b"],
                offsets=[(0, 1)],  # too short
                scores=[0.5, 0.5],
                source_ids=["claim", "claim"],  # type: ignore[arg-type]
                source_texts={"claim": "ab", "evidence": ""},
                top_k=2,
            )

    def test_unknown_source_is_skipped(self) -> None:
        tokens = ["a", "b"]
        offsets = [(0, 1), (1, 2)]
        scores = [0.9, 0.5]
        source_ids = ["claim", "other"]  # "other" not in source_texts
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": "ab", "evidence": ""},
            top_k=2,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "a")

    def test_tie_broken_by_token_index(self) -> None:
        tokens = ["a", "b"]
        offsets = [(0, 1), (1, 2)]
        scores = [0.5, 0.5]  # exact tie
        source_ids = ["claim", "claim"]
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": "ab", "evidence": ""},
            top_k=2,
        )
        # Stable sort: earlier token index wins ties.
        self.assertEqual(spans[0].token_indices, (0,))
        self.assertEqual(spans[1].token_indices, (1,))

    def test_claim_and_evidence_both_represented(self) -> None:
        tokens = ["a", "b", "c"]
        offsets = [(0, 1), (0, 1), (1, 2)]
        scores = [0.9, 0.8, 0.7]
        source_ids = ["claim", "evidence", "evidence"]
        spans = score_to_spans(
            tokens=tokens,
            offsets=offsets,
            scores=scores,
            source_ids=source_ids,  # type: ignore[arg-type]
            source_texts={"claim": "a", "evidence": "bc"},
            top_k=3,
        )
        self.assertEqual(len(spans), 3)
        sources = {s.source for s in spans}
        self.assertEqual(sources, {"claim", "evidence"})


# ---------------------------------------------------------------------------
# merge_contiguous_spans — phrase-level merging
# ---------------------------------------------------------------------------


class TestMergeContiguousSpans(unittest.TestCase):
    def test_adjacent_same_source_merges_gap_zero(self) -> None:
        source_texts = {"claim": "heartbeat", "evidence": ""}
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.7,
                start_char=0, end_char=5, token_indices=(1,),
            ),
            CausalSpan(
                text="beat", source="claim", score=0.5,
                start_char=5, end_char=9, token_indices=(2,),
            ),
        ]
        merged = merge_contiguous_spans(spans, source_texts)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "heartbeat")
        self.assertAlmostEqual(merged[0].score, 1.2)
        self.assertEqual(merged[0].token_indices, (1, 2))
        self.assertEqual(merged[0].start_char, 0)
        self.assertEqual(merged[0].end_char, 9)

    def test_adjacent_same_source_merges_with_whitespace(self) -> None:
        source_texts = {"claim": "heart beat", "evidence": ""}
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.7,
                start_char=0, end_char=5, token_indices=(1,),
            ),
            CausalSpan(
                text="beat", source="claim", score=0.5,
                start_char=6, end_char=10, token_indices=(2,),
            ),
        ]
        merged = merge_contiguous_spans(
            spans, source_texts, join_whitespace=True
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "heart beat")
        self.assertAlmostEqual(merged[0].score, 1.2)

    def test_whitespace_join_disabled_keeps_separate(self) -> None:
        source_texts = {"claim": "heart beat", "evidence": ""}
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.7,
                start_char=0, end_char=5,
            ),
            CausalSpan(
                text="beat", source="claim", score=0.5,
                start_char=6, end_char=10,
            ),
        ]
        merged = merge_contiguous_spans(
            spans, source_texts, join_whitespace=False
        )
        self.assertEqual(len(merged), 2)

    def test_different_sources_do_not_merge(self) -> None:
        source_texts = {"claim": "heart", "evidence": "beat"}
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.7,
                start_char=0, end_char=5,
            ),
            CausalSpan(
                text="beat", source="evidence", score=0.5,
                start_char=0, end_char=4,
            ),
        ]
        merged = merge_contiguous_spans(spans, source_texts)
        self.assertEqual(len(merged), 2)

    def test_non_adjacent_same_source_does_not_merge(self) -> None:
        source_texts = {"claim": "a b c d e", "evidence": ""}
        spans = [
            CausalSpan(
                text="a", source="claim", score=0.9,
                start_char=0, end_char=1,
            ),
            CausalSpan(
                text="e", source="claim", score=0.8,
                start_char=8, end_char=9,
            ),
        ]
        merged = merge_contiguous_spans(spans, source_texts)
        self.assertEqual(len(merged), 2)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(
            merge_contiguous_spans([], {"claim": "", "evidence": ""}),
            [],
        )

    def test_merged_sorted_by_score_descending(self) -> None:
        source_texts = {"claim": "heart beat and loudly", "evidence": ""}
        spans = [
            CausalSpan(
                text="heart", source="claim", score=0.4,
                start_char=0, end_char=5,
            ),
            CausalSpan(
                text="beat", source="claim", score=0.3,
                start_char=6, end_char=10,
            ),
            CausalSpan(
                text="loudly", source="claim", score=0.9,
                start_char=15, end_char=21,
            ),
        ]
        merged = merge_contiguous_spans(spans, source_texts)
        # heart + beat merge (gap=1), loudly stays solo (gap=5).
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "loudly")
        self.assertAlmostEqual(merged[0].score, 0.9)
        self.assertEqual(merged[1].text, "heart beat")
        self.assertAlmostEqual(merged[1].score, 0.7)

    def test_three_way_chain_collapses(self) -> None:
        source_texts = {"claim": "abc", "evidence": ""}
        spans = [
            CausalSpan(
                text="a", source="claim", score=0.1,
                start_char=0, end_char=1, token_indices=(0,),
            ),
            CausalSpan(
                text="b", source="claim", score=0.2,
                start_char=1, end_char=2, token_indices=(1,),
            ),
            CausalSpan(
                text="c", source="claim", score=0.3,
                start_char=2, end_char=3, token_indices=(2,),
            ),
        ]
        merged = merge_contiguous_spans(spans, source_texts)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "abc")
        self.assertAlmostEqual(merged[0].score, 0.6)
        self.assertEqual(merged[0].token_indices, (0, 1, 2))

    def test_input_order_invariant(self) -> None:
        source_texts = {"claim": "heartbeat", "evidence": ""}
        forward = [
            CausalSpan(
                text="heart", source="claim", score=0.7,
                start_char=0, end_char=5,
            ),
            CausalSpan(
                text="beat", source="claim", score=0.5,
                start_char=5, end_char=9,
            ),
        ]
        reversed_order = list(reversed(forward))
        self.assertEqual(
            merge_contiguous_spans(forward, source_texts),
            merge_contiguous_spans(reversed_order, source_texts),
        )


# ---------------------------------------------------------------------------
# split_tokens_by_sep — SEP-index-based claim/evidence labeling
# ---------------------------------------------------------------------------


class TestSplitTokensBySep(unittest.TestCase):
    def test_roberta_style_split(self) -> None:
        # <s> claim1 claim2 </s> </s> ev1 ev2 </s>
        # specials at 0, 3, 4, 7 — first evidence-side token is 4.
        special_mask = [1, 0, 0, 1, 1, 0, 0, 1]
        labels = split_tokens_by_sep(special_mask, sep_index=4)
        self.assertEqual(
            labels,
            [
                "claim", "claim", "claim", "claim",
                "evidence", "evidence", "evidence", "evidence",
            ],
        )

    def test_none_sep_labels_all_claim(self) -> None:
        labels = split_tokens_by_sep([1, 0, 0, 1], sep_index=None)
        self.assertEqual(labels, ["claim", "claim", "claim", "claim"])

    def test_empty_input(self) -> None:
        self.assertEqual(split_tokens_by_sep([], sep_index=None), [])

    def test_sep_at_zero_labels_all_evidence(self) -> None:
        # Pathological: sep_index=0 means the first token is already evidence.
        labels = split_tokens_by_sep([1, 0, 0], sep_index=0)
        self.assertEqual(labels, ["evidence", "evidence", "evidence"])

    def test_sep_index_larger_than_sequence(self) -> None:
        # Out-of-range sep index gracefully degrades to "all claim".
        labels = split_tokens_by_sep([1, 0, 0], sep_index=99)
        self.assertEqual(labels, ["claim", "claim", "claim"])


# ---------------------------------------------------------------------------
# Heavy class — ImportError surface only
# ---------------------------------------------------------------------------


class TestCausalTermIdentifierImportPath(unittest.TestCase):
    def test_class_is_importable_without_heavy_deps(self) -> None:
        # Importing ``CausalTermIdentifier`` must succeed regardless of
        # whether captum / torch / transformers are installed.  Only
        # instantiation triggers those imports.
        self.assertTrue(callable(CausalTermIdentifier))

    def test_missing_captum_raises_importerror_with_hint(self) -> None:
        # Force captum to look missing — ``mock.patch.dict`` sets
        # ``sys.modules[key] = None`` which Python interprets as "import
        # failed".  We set the top-level and the ``.attr`` submodule so
        # both ``import captum`` and ``from captum.attr import ...`` hit
        # the ImportError branch.
        with mock.patch.dict(
            sys.modules,
            {"captum": None, "captum.attr": None},
        ):
            with self.assertRaises(ImportError) as cm:
                CausalTermIdentifier(model_path="/nonexistent.pt")
        self.assertIn("captum", str(cm.exception))
        self.assertIn("pip install", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
