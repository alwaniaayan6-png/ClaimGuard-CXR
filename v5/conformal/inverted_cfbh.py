"""Inverted cfBH (v4 carry-over, generalized for v5).

Treats H0: claim is CONTRADICTED (hallucinated). Calibration pool = contradicted
calibration claims. Small p-value ⇒ anomalously high supported-score vs
contradicted calibration ⇒ evidence against H0 ⇒ can enter the green set.

See ARCHITECTURE_V5_IMAGE_GROUNDED.md §8.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InvertedCfBHResult:
    p_values: np.ndarray
    green_mask: np.ndarray  # True iff rejected at level alpha (= certified-safe)
    bh_threshold: float
    n_green: int


def inverted_cfbh(
    test_scores: np.ndarray,
    cal_contra_scores: np.ndarray,
    alpha: float = 0.05,
) -> InvertedCfBHResult:
    """Apply Jin & Candes (2023) cfBH with inverted calibration.

    Args:
        test_scores: shape (n_test,). The verifier's supported-probability
            on test claims.
        cal_contra_scores: shape (n_cal,). Supported-probability on
            calibration claims whose TRUE label is contradicted.
        alpha: target FDR level.

    Returns:
        InvertedCfBHResult with conformal p-values and the green mask.
    """
    test = np.asarray(test_scores, dtype=float)
    cal = np.asarray(cal_contra_scores, dtype=float)
    n_cal = len(cal)
    if n_cal < 20:
        raise ValueError(
            f"n_cal={n_cal} is too small for stable cfBH; need at least 20"
        )
    # p_j = (|{i in C_contra : s_i >= s_j}| + 1) / (n_cal + 1)
    # For efficiency: sort calibration scores descending and use searchsorted.
    cal_sorted = np.sort(cal)  # ascending
    # Count of cal >= each test score = n_cal - searchsorted(cal_sorted, test, 'left')
    counts = n_cal - np.searchsorted(cal_sorted, test, side="left")
    p = (counts + 1.0) / (n_cal + 1.0)

    # Global BH at level alpha:
    order = np.argsort(p)
    p_sorted = p[order]
    n = len(p)
    thresholds = alpha * np.arange(1, n + 1) / n
    below = p_sorted <= thresholds
    if not below.any():
        return InvertedCfBHResult(
            p_values=p, green_mask=np.zeros_like(p, dtype=bool), bh_threshold=0.0, n_green=0
        )
    k = np.max(np.where(below)[0]) + 1
    bh_thr = thresholds[k - 1]
    green = p <= bh_thr
    return InvertedCfBHResult(
        p_values=p,
        green_mask=green,
        bh_threshold=float(bh_thr),
        n_green=int(green.sum()),
    )
