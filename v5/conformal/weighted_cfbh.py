"""Weighted cfBH (Tibshirani et al. 2019) for v5 under covariate shift.

Calibration-to-test covariate shift is handled by density-ratio reweighting.
The classifier-based density-ratio estimator (gradient-boosted, 200 trees) is
trained to distinguish calibration features (label=0) from test features
(label=1). Weights w_i = P(test | x_i) / P(cal | x_i) are normalized and used
in the BH step.

See architecture doc §8.1 (weighted cfBH).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeightedCfBHResult:
    p_values: np.ndarray
    green_mask: np.ndarray
    weights: np.ndarray
    ess: float
    bh_threshold: float
    n_green: int


def _classifier_density_ratio(
    cal_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights_cal, weights_test) via logistic regression classifier."""
    from sklearn.ensemble import GradientBoostingClassifier

    X = np.concatenate([cal_features, test_features], axis=0)
    y = np.concatenate([np.zeros(len(cal_features)), np.ones(len(test_features))])
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=17)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    ratio = probs / (1 - probs)  # P(test) / P(cal)
    w_cal = ratio[: len(cal_features)]
    w_test = ratio[len(cal_features) :]
    return w_cal, w_test


def weighted_cfbh(
    test_scores: np.ndarray,
    cal_contra_scores: np.ndarray,
    cal_features: np.ndarray,
    test_features: np.ndarray,
    alpha: float = 0.05,
) -> WeightedCfBHResult:
    """Weighted version of inverted cfBH.

    Features should be site-invariant surface statistics: report length, age,
    claim length, claim finding-family, etc.
    """
    test_scores = np.asarray(test_scores, dtype=float)
    cal = np.asarray(cal_contra_scores, dtype=float)
    n_cal = len(cal)
    w_cal, w_test = _classifier_density_ratio(cal_features, test_features)

    # Normalize per-test weight-of-cal: w_i_j = w_cal / (sum w_cal + w_test_j)
    # For simplicity we use marginal weights (Tibshirani split-conformal variant).
    w_cal_norm = w_cal / w_cal.sum()

    # Weighted p-value:
    # p_j = [sum_{i: s_i >= s_j} w_cal_norm_i * (n_cal+1)/n_cal + 1/(n_cal+1)]
    # Simpler: p_j = sum_i w_cal_norm_i * 1[s_i >= s_j] + 1/(n_cal+1), clipped.
    order = np.argsort(cal)
    cal_sorted = cal[order]
    w_sorted = w_cal_norm[order]
    cumsum = np.cumsum(w_sorted[::-1])[::-1]  # cumsum from right: mass of cal >= threshold
    # For a test score s, the mass of cal >= s is cumsum[idx] where idx = searchsorted
    idx = np.searchsorted(cal_sorted, test_scores, side="left")
    mass = np.where(idx < n_cal, cumsum[np.clip(idx, 0, n_cal - 1)], 0.0)
    mass = np.where(idx >= n_cal, 0.0, mass)
    p = mass + 1.0 / (n_cal + 1.0)
    p = np.clip(p, 1.0 / (n_cal + 1.0), 1.0)

    # BH at level alpha
    order2 = np.argsort(p)
    p_sorted = p[order2]
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

    ess = (w_cal.sum() ** 2) / (w_cal ** 2).sum()
    return WeightedCfBHResult(
        p_values=p,
        green_mask=green,
        weights=w_cal,
        ess=float(ess),
        bh_threshold=float(bh_thr),
        n_green=int(green.sum()),
    )
