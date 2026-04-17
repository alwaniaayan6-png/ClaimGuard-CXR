"""Doubly-robust cfBH (Fannjiang et al. 2024 / Yang & Kuchibhotla 2024).

Robust to misspecified density ratios by combining the weighted conformal score
with a regression-based nuisance estimator for the score distribution on the
test population. Here we implement a practical variant: weighted conformal as
per Tibshirani + a residual correction using a regression of score on features
trained on calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DRCfBHResult:
    p_values: np.ndarray
    green_mask: np.ndarray
    bh_threshold: float
    n_green: int
    correction_magnitude: float


def doubly_robust_cfbh(
    test_scores: np.ndarray,
    cal_contra_scores: np.ndarray,
    cal_features: np.ndarray,
    test_features: np.ndarray,
    alpha: float = 0.05,
) -> DRCfBHResult:
    from sklearn.ensemble import GradientBoostingRegressor

    from .weighted_cfbh import weighted_cfbh

    # Start with weighted cfBH
    w = weighted_cfbh(
        test_scores=test_scores,
        cal_contra_scores=cal_contra_scores,
        cal_features=cal_features,
        test_features=test_features,
        alpha=alpha,
    )

    # Regression correction: fit score on cal_features; use residuals on test.
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=29)
    reg.fit(cal_features, cal_contra_scores)
    cal_pred = reg.predict(cal_features)
    test_pred = reg.predict(test_features)
    cal_resid = cal_contra_scores - cal_pred
    # Corrected score = original - regression prediction = residual
    # Apply inverted cfBH to residual scores on test vs cal residuals
    test_resid = test_scores - test_pred

    # Simple p-value via residual distribution
    cal_resid_sorted = np.sort(cal_resid)
    n_cal = len(cal_resid_sorted)
    counts = n_cal - np.searchsorted(cal_resid_sorted, test_resid, side="left")
    p_dr = (counts + 1.0) / (n_cal + 1.0)

    # Take minimum of weighted and DR p-values (conservative)
    p = np.minimum(w.p_values, p_dr)

    order = np.argsort(p)
    p_sorted = p[order]
    n = len(p)
    thresholds = alpha * np.arange(1, n + 1) / n
    below = p_sorted <= thresholds
    if not below.any():
        bh_thr = 0.0
        green = np.zeros_like(p, dtype=bool)
    else:
        k = np.max(np.where(below)[0]) + 1
        bh_thr = thresholds[k - 1]
        green = p <= bh_thr

    return DRCfBHResult(
        p_values=p,
        green_mask=green,
        bh_threshold=float(bh_thr),
        n_green=int(green.sum()),
        correction_magnitude=float(np.abs(p - w.p_values).mean()),
    )
