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
