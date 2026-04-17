"""Inverted conformal Benjamini-Hochberg (cfBH) procedure.

The verifier outputs a supported-probability s in [0, 1]. Standard cfBH
calibrates on faithful (not-contradicted) claims — which concentrates at
the softmax ceiling and makes p-values uninformative (see ARCHITECTURE_PATH_B.md
Section 6 and the original D-series decisions in the v1 codebase).

Under the null H0: "claim j is contradicted (hallucinated)", we calibrate
using contradicted calibration claims. Their scores sit near zero,
providing a well-separated reference distribution. The conformal p-value
for test claim j is

    p_j = (|{i in C_contra : s_i >= s_j}| + 1) / (|C_contra| + 1)

A small p_j indicates s_j is anomalously high vs contradicted calibration
-> evidence against H0 -> claim is NOT a hallucination. Apply BH at level
alpha to the p_j for all test claims; the set of rejected nulls is the
"green" (safe) set, with FDR controlled at alpha under exchangeability of
test-contradicted and calibration-contradicted scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class BHResult:
    """Output of a single Benjamini-Hochberg application."""
    alpha: float
    n_green: int
    n_total: int
    green_mask: np.ndarray           # (n_total,) boolean — which test claims are "safe"
    p_values: np.ndarray             # (n_total,) conformal p-values
    bh_threshold: float
    power: Optional[float] = None    # populated by audit() when truth is known
    fdr: Optional[float] = None

    @property
    def coverage(self) -> float:
        return float(self.n_green) / max(self.n_total, 1)


class InvertedCfBH:
    """Inverted conformal BH with one-sided p-values calibrated on contradicted claims.

    Usage
    -----
    >>> procedure = InvertedCfBH()
    >>> procedure.fit(calibration_scores_contradicted)
    >>> result = procedure.predict(test_scores, alpha=0.05)
    >>> # result.green_mask is the set of claims the system certifies as safe
    """

    def __init__(self):
        self._cal_scores: Optional[np.ndarray] = None

    def fit(self, calibration_contradicted_scores: Sequence[float]) -> "InvertedCfBH":
        """Fit using scores of KNOWN-contradicted calibration claims only.

        The calibration set must be the supported-probabilities of claims that
        are ground-truth-labeled as contradicted. Score near 0 expected.
        """
        cal = np.asarray(list(calibration_contradicted_scores), dtype=float)
        if cal.size == 0:
            raise ValueError("Calibration set is empty; need >=1 contradicted claim")
        self._cal_scores = np.sort(cal)
        return self

    def _p_value(self, score: float) -> float:
        assert self._cal_scores is not None, "Call fit() first"
        # p = (# cal >= test + 1) / (n_cal + 1)
        # Use searchsorted: index where score would insert in ascending array.
        # Count of cal >= score = n_cal - insertion_right.
        n = self._cal_scores.size
        idx = np.searchsorted(self._cal_scores, score, side="left")
        n_geq = n - idx
        return (n_geq + 1) / (n + 1)

    def p_values(self, test_scores: Sequence[float]) -> np.ndarray:
        return np.asarray([self._p_value(s) for s in test_scores], dtype=float)

    def predict(self, test_scores: Sequence[float], alpha: float) -> BHResult:
        """Run BH at level alpha; return green-set mask and diagnostics."""
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        ps = self.p_values(test_scores)
        n = ps.size

        order = np.argsort(ps)
        sorted_ps = ps[order]
        ranks = np.arange(1, n + 1)
        bh_crit = ranks * alpha / n
        passing = sorted_ps <= bh_crit
        if passing.any():
            k = int(np.max(np.where(passing)[0])) + 1
            threshold = sorted_ps[k - 1]
        else:
            k = 0
            threshold = 0.0

        green_mask = np.zeros(n, dtype=bool)
        if k > 0:
            green_mask[order[:k]] = True

        return BHResult(
            alpha=alpha,
            n_green=int(green_mask.sum()),
            n_total=n,
            green_mask=green_mask,
            p_values=ps,
            bh_threshold=float(threshold),
        )

    @staticmethod
    def audit(result: BHResult, true_labels: Sequence[int]) -> BHResult:
        """Populate empirical FDR and power given ground-truth labels.

        true_labels: 0 = not contradicted (faithful), 1 = contradicted.
        FDR = fraction of green claims that are contradicted.
        Power = fraction of truly-faithful claims that are green.
        """
        y = np.asarray(list(true_labels), dtype=int)
        if y.size != result.n_total:
            raise ValueError("Length mismatch between true labels and test claims")
        if result.n_green == 0:
            result.fdr = 0.0
        else:
            result.fdr = float(y[result.green_mask].sum()) / result.n_green
        n_faithful = int((y == 0).sum())
        if n_faithful == 0:
            result.power = float("nan")
        else:
            result.power = float(((y == 0) & result.green_mask).sum()) / n_faithful
        return result
