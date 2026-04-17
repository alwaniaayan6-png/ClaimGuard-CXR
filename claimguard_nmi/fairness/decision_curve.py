"""Decision-curve analysis (DCA) for claim-level certification.

Vickers & Elkin 2006 net-benefit framework, adapted:
    Net benefit at threshold p_t =
        (true positives / n)
        − (false positives / n) · (p_t / (1 − p_t))

Where "positive" = the system flagged the claim as contradicted.

For clinical utility interpretation: a flagged claim that IS
hallucinated is a true positive (caught); a flagged claim that is NOT
hallucinated is a false positive (wasted reader time). The threshold
p_t encodes the clinician's willingness to tolerate false positives.

Reports:
  - Net benefit curve for the model
  - Reference curves: treat-all (flag everything) + treat-none (flag nothing)
  - Operating-point summary at clinically plausible thresholds
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class NetBenefitCurve:
    thresholds: np.ndarray
    model: np.ndarray
    treat_all: np.ndarray
    treat_none: np.ndarray

    def as_dict(self) -> dict:
        return {
            "thresholds": self.thresholds.tolist(),
            "model": self.model.tolist(),
            "treat_all": self.treat_all.tolist(),
            "treat_none": self.treat_none.tolist(),
        }


def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
) -> NetBenefitCurve:
    """Compute decision-curve net benefit."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have same shape")
    n = float(y_true.size)
    prevalence = float(y_true.sum()) / n

    thresholds_arr = np.asarray(list(thresholds), dtype=float)
    model_nb = np.zeros_like(thresholds_arr)
    treat_all_nb = np.zeros_like(thresholds_arr)
    treat_none_nb = np.zeros_like(thresholds_arr)

    for i, t in enumerate(thresholds_arr):
        if not (0.0 < t < 1.0):
            continue
        flagged = y_prob >= t
        tp = float(((y_true == 1) & flagged).sum())
        fp = float(((y_true == 0) & flagged).sum())
        w = t / (1.0 - t)
        model_nb[i] = (tp / n) - (fp / n) * w
        # Treat all: everyone flagged; tp = all positives, fp = all negatives.
        treat_all_nb[i] = prevalence - (1 - prevalence) * w
        # Treat none: trivially zero.
        treat_none_nb[i] = 0.0
    return NetBenefitCurve(
        thresholds=thresholds_arr,
        model=model_nb,
        treat_all=treat_all_nb,
        treat_none=treat_none_nb,
    )


def operating_point_summary(
    curve: NetBenefitCurve,
    clinical_thresholds: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict[str, Dict[str, float]]:
    """Pick the net-benefit values at the clinician's declared thresholds."""
    out: Dict[str, Dict[str, float]] = {}
    for t in clinical_thresholds:
        idx = int(np.argmin(np.abs(curve.thresholds - t)))
        out[f"t={t:.2f}"] = {
            "model_nb": float(curve.model[idx]),
            "treat_all_nb": float(curve.treat_all[idx]),
            "treat_none_nb": float(curve.treat_none[idx]),
            "advantage_over_all": float(curve.model[idx] - curve.treat_all[idx]),
            "advantage_over_none": float(curve.model[idx] - curve.treat_none[idx]),
        }
    return out
