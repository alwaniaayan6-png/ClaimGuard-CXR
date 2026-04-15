"""Cross-encoder reranker for ClaimGuard-CXR evidence retrieval.

Reranks the top-k candidates from hybrid retrieval (MedCPT + BM25)
down to the final top-2 evidence passages per claim.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """DeBERTa-based cross-encoder for passage reranking.

    Takes (query, passage) pairs and scores relevance.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
        max_length: Maximum input token length.
        device: Device to run on.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

        logger.info(f"Loaded reranker: {model_name} on {self.device}")

    @torch.no_grad()
    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        """Score relevance of passages to a query.

        Args:
            query: Query/claim text.
            passages: List of candidate passages.

        Returns:
            Array of relevance scores (higher = more relevant).
        """
        if not passages:
            return np.array([])

        pairs = [(query, passage) for passage in passages]

        features = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**features)
        scores = outputs.logits.squeeze(-1).cpu().numpy()

        return scores

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int = 2,
    ) -> list[tuple[str, float]]:
        """Rerank passages and return top-k.

        Args:
            query: Query/claim text.
            passages: Candidate passages to rerank.
            top_k: Number of top passages to return.

        Returns:
            List of (passage_text, score) tuples, sorted by relevance.
        """
        scores = self.score(query, passages)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(passages[i], float(scores[i])) for i in top_indices]

    @torch.no_grad()
    def score_batch(
        self,
        query_passage_pairs: list[tuple[str, list[str]]],
        batch_size: int = 64,
    ) -> list[np.ndarray]:
        """Score many (query, passages) groups in a single GPU pass.

        Flattens all (query_i, passage_ij) pairs, runs the cross-encoder
        in fixed-size mini-batches, and regroups scores by input query.

        This is typically 5-10× faster than repeatedly calling
        ``score`` in a Python loop because (a) it avoids per-call
        tokenizer and CUDA-launch overhead and (b) a single padded
        batch usually fits on-GPU at much higher effective throughput.

        Args:
            query_passage_pairs: One tuple per query.  Each tuple is
                ``(query_text, list_of_candidate_passages)``.  A query
                with zero passages is allowed; it contributes an empty
                ``np.ndarray`` to the output.
            batch_size: Cross-encoder mini-batch size.  Defaults to 64,
                which is comfortable on a single H100 at max_length=512.

        Returns:
            A list of score arrays, parallel to ``query_passage_pairs``.
            Entry ``i`` has shape ``(len(passages_i),)``.
        """
        if not query_passage_pairs:
            return []

        # Flatten pairs while remembering the group boundaries so we
        # can regroup the flat score tensor afterwards.
        flat_queries: list[str] = []
        flat_passages: list[str] = []
        group_sizes: list[int] = []
        for query, passages in query_passage_pairs:
            group_sizes.append(len(passages))
            for passage in passages:
                flat_queries.append(query)
                flat_passages.append(passage)

        if not flat_queries:
            return [np.array([]) for _ in query_passage_pairs]

        flat_scores = np.empty(len(flat_queries), dtype=np.float32)
        for start in range(0, len(flat_queries), batch_size):
            end = start + batch_size
            batch_q = flat_queries[start:end]
            batch_p = flat_passages[start:end]
            features = self.tokenizer(
                batch_q,
                batch_p,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**features)
            logits = outputs.logits.squeeze(-1)
            flat_scores[start:end] = logits.detach().cpu().numpy().astype(np.float32)

        # Regroup by original query.
        out: list[np.ndarray] = []
        cursor = 0
        for size in group_sizes:
            out.append(flat_scores[cursor : cursor + size].copy())
            cursor += size
        return out

    def rerank_batch(
        self,
        query_passage_pairs: list[tuple[str, list[str]]],
        top_k: int = 2,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """Batched version of :meth:`rerank`.

        Produces per-query top-``top_k`` lists of
        ``(passage_text, score)`` tuples using a single flattened GPU
        pass via :meth:`score_batch`.

        Args:
            query_passage_pairs: ``[(query, [passage, ...]), ...]``.
            top_k: Number of top passages to keep per query.
            batch_size: Mini-batch size for the underlying GPU call.

        Returns:
            List parallel to ``query_passage_pairs``; entry ``i`` is the
            per-query top-``top_k`` list of reranked passages.
        """
        all_scores = self.score_batch(query_passage_pairs, batch_size=batch_size)
        results: list[list[tuple[str, float]]] = []
        for (_, passages), scores in zip(query_passage_pairs, all_scores):
            if len(passages) == 0:
                results.append([])
                continue
            order = np.argsort(-scores, kind="stable")[:top_k]
            results.append(
                [(passages[int(i)], float(scores[int(i)])) for i in order]
            )
        return results
