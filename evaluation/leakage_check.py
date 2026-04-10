"""Train/test leakage detection for ClaimGuard-CXR.

Provides BM25-based lexical similarity and MinHash near-duplicate detection
to catch radiology reports that are near-identical between train and test
splits (which would inflate evaluation metrics).
"""

from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# BM25 helpers (lightweight, no external dependency)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _idf(df: int, n_docs: int) -> float:
    """Robertson-Sparck Jones IDF with smoothing."""
    return log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def bm25_similarity_distribution(
    test_reports: list[str],
    train_reports: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    """Compute BM25 similarity from each test report to its best matching train report.

    For every test report, scores all train reports and records the maximum
    BM25 similarity.  Returns summary statistics and a histogram.

    Args:
        test_reports: List of test-set report strings.
        train_reports: List of train-set report strings.
        k1: BM25 term saturation parameter.  Default 1.5.
        b: BM25 length normalisation parameter.  Default 0.75.

    Returns:
        dict with keys:

        - ``"mean"``      — mean of per-test-report max BM25 score
        - ``"std"``       — standard deviation
        - ``"max"``       — highest BM25 score observed
        - ``"histogram"`` — dict with ``"counts"`` and ``"bin_edges"``
          (20 bins)
        - ``"max_scores"`` — np.ndarray of shape (n_test,) with per-report
          maximum BM25 scores
    """
    if not train_reports or not test_reports:
        empty: np.ndarray = np.array([], dtype=float)
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"),
                "histogram": {"counts": [], "bin_edges": []},
                "max_scores": empty}

    # Tokenise
    train_tokens: list[list[str]] = [_tokenise(r) for r in train_reports]
    test_tokens: list[list[str]] = [_tokenise(r) for r in test_reports]

    # Build IDF from training corpus
    n_train = len(train_tokens)
    df_counter: Counter = Counter()
    for tokens in train_tokens:
        for t in set(tokens):
            df_counter[t] += 1

    # Avg document length in train
    avg_dl = float(np.mean([len(t) for t in train_tokens])) if train_tokens else 1.0

    # Pre-compute term frequencies for training docs
    train_tf: list[Counter] = [Counter(t) for t in train_tokens]
    train_dl: list[int] = [len(t) for t in train_tokens]

    max_scores = np.zeros(len(test_reports), dtype=float)

    for i, query_tokens in enumerate(test_tokens):
        query_terms = set(query_tokens)
        best = -1.0

        for j, tf in enumerate(train_tf):
            dl_j = train_dl[j]
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                df = df_counter.get(term, 0)
                idf_val = _idf(df, n_train)
                tf_val = tf[term]
                numerator = tf_val * (k1 + 1.0)
                denominator = tf_val + k1 * (1.0 - b + b * dl_j / avg_dl)
                score += idf_val * (numerator / denominator)
            if score > best:
                best = score

        max_scores[i] = best

    counts, bin_edges = np.histogram(max_scores, bins=20)

    return {
        "mean": float(np.mean(max_scores)),
        "std": float(np.std(max_scores, ddof=1)) if len(max_scores) > 1 else 0.0,
        "max": float(np.max(max_scores)),
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "max_scores": max_scores,
    }


# ---------------------------------------------------------------------------
# MinHash near-duplicate detection
# ---------------------------------------------------------------------------

def minhash_near_duplicate_check(
    test_reports: list[str],
    train_reports: list[str],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> dict:
    """Detect near-duplicate reports between test and train splits via MinHash.

    Uses the ``datasketch`` library's MinHash for efficient Jaccard-similarity
    estimation.  Falls back to a brute-force Jaccard implementation if
    ``datasketch`` is not installed.

    Args:
        test_reports: List of test-set report strings.
        train_reports: List of train-set report strings.
        threshold: Jaccard similarity threshold above which a pair is
            considered a near-duplicate.  Default 0.8.
        num_perm: Number of permutation functions for MinHash.  Higher values
            give better Jaccard estimates.  Default 128.

    Returns:
        dict with keys:

        - ``"n_duplicates"``   — number of test reports with at least one
          near-duplicate in the training set
        - ``"duplicate_pairs"`` — list of (test_idx, train_idx, jaccard_est)
          tuples for each detected pair
        - ``"backend"``        — ``"datasketch"`` or ``"brute_force"``
    """
    if not test_reports or not train_reports:
        return {"n_duplicates": 0, "duplicate_pairs": [], "backend": "none"}

    test_shingles = [set(_tokenise(r)) for r in test_reports]
    train_shingles = [set(_tokenise(r)) for r in train_reports]

    duplicate_pairs: list[tuple[int, int, float]] = []
    backend = "brute_force"

    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import]

        backend = "datasketch"
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        # Insert training docs
        train_minhashes: list[MinHash] = []
        for j, shingles in enumerate(train_shingles):
            m = MinHash(num_perm=num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))
            lsh.insert(f"train_{j}", m)
            train_minhashes.append(m)

        # Query with test docs
        for i, shingles in enumerate(test_shingles):
            m_test = MinHash(num_perm=num_perm)
            for s in shingles:
                m_test.update(s.encode("utf-8"))
            results = lsh.query(m_test)
            for key in results:
                j = int(key.split("_")[1])
                jaccard_est = float(m_test.jaccard(train_minhashes[j]))
                duplicate_pairs.append((i, j, jaccard_est))

    except ImportError:
        # Brute-force fallback: O(|test| * |train|), fine for small datasets
        for i, ts in enumerate(test_shingles):
            for j, tr in enumerate(train_shingles):
                if not ts and not tr:
                    continue
                intersection = len(ts & tr)
                union = len(ts | tr)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= threshold:
                    duplicate_pairs.append((i, j, jaccard))

    # Deduplicate: count unique test indices
    test_indices_with_dup = {pair[0] for pair in duplicate_pairs}

    return {
        "n_duplicates": len(test_indices_with_dup),
        "duplicate_pairs": duplicate_pairs,
        "backend": backend,
    }
