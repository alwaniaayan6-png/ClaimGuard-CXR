"""BM25 sparse retrieval index for ClaimGuard-CXR.

Builds and queries a BM25 index over training split report passages.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Medical stop words to remove
_MEDICAL_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "this", "that", "these", "those", "it", "its",
    "patient", "history", "exam", "examination", "study",
    "xxxx",  # CheXpert Plus de-identification placeholder
}


def tokenize_medical(text: str) -> list[str]:
    """Tokenize text for BM25 indexing with medical-aware preprocessing.

    Args:
        text: Input text.

    Returns:
        List of lowercase tokens with stop words removed.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', ' ', text)  # keep hyphens (e.g., "right-sided")
    tokens = text.split()
    return [t for t in tokens if t not in _MEDICAL_STOPWORDS and len(t) > 1]


class BM25Index:
    """BM25 sparse retrieval index.

    Args:
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index = None
        self.passages: list[str] = []
        self.passage_ids: list[str] = []

    def build(
        self,
        passages: list[str],
        passage_ids: Optional[list[str]] = None,
    ) -> None:
        """Build the BM25 index from passages.

        Args:
            passages: List of text passages to index.
            passage_ids: Optional IDs for each passage.
        """
        from rank_bm25 import BM25Okapi

        self.passages = passages
        self.passage_ids = passage_ids or [str(i) for i in range(len(passages))]

        tokenized = [tokenize_medical(p) for p in passages]
        self.index = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        logger.info(f"Built BM25 index over {len(passages)} passages")

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, float, str]]:
        """Search the index for passages matching a query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of (passage_text, score, passage_id) tuples, sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = tokenize_medical(query)
        scores = self.index.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.passages[i], float(scores[i]), self.passage_ids[i])
            for i in top_indices
            if scores[i] > 0
        ]

        return results

    def save(self, path: str | Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'passages': self.passages,
                'passage_ids': self.passage_ids,
                'k1': self.k1,
                'b': self.b,
            }, f)
        logger.info(f"Saved BM25 index to {path}")

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.index = data['index']
        self.passages = data['passages']
        self.passage_ids = data['passage_ids']
        self.k1 = data['k1']
        self.b = data['b']
        logger.info(f"Loaded BM25 index with {len(self.passages)} passages from {path}")
