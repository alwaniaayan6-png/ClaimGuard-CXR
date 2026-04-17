"""Calibration metrics: ECE, MCE, reliability diagrams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    n_bins: int
    bin_boundaries: list[float]
    bin_accuracy: list[float]
    bin_confidence: list[float]
    bin_counts: list[int]


def compute_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> CalibrationReport:
    """ECE with equal-width bins over confidence (= max softmax)."""
    # For binary classification with P(not_contra) = scores, model confidence is
    # max(scores, 1 - scores), predicted class is 0 iff scores >= 0.5.
    conf = np.maximum(scores, 1 - scores)
    preds = (scores < 0.5).astype(int)  # 1 = contradicted
    correct = (preds == labels).astype(float)

    bins = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is always >= 0.5
    accs: list[float] = []
    confs: list[float] = []
    counts: list[int] = []
    ece = 0.0
    mce = 0.0
    N = len(scores)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            accs.append(0.0)
            confs.append((lo + hi) / 2)
            counts.append(0)
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        accs.append(acc_bin)
        confs.append(conf_bin)
        counts.append(n_bin)
        gap = abs(acc_bin - conf_bin)
        ece += (n_bin / N) * gap
        mce = max(mce, gap)
    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        n_bins=n_bins,
        bin_boundaries=bins.tolist(),
        bin_accuracy=accs,
        bin_confidence=confs,
        bin_counts=counts,
    )
