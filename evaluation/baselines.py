"""Baseline comparison dispatcher for ClaimGuard-CXR.

Implements three baselines against which the full ClaimGuard-CXR system is
compared in the paper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from verifact.evaluation.metrics import (
    claim_hallucination_metrics,
    fdr_among_green,
    green_claim_fraction,
)
from verifact.inference.conformal_triage import ConformalClaimTriage, compute_fdr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline: no verification
# ---------------------------------------------------------------------------

def _baseline_no_verification(config: dict) -> dict:
    """Baseline: accept all claims as green without any verification.

    This measures the natural hallucination rate in raw generated reports —
    the ceiling on how bad a system that does nothing would perform.

    Expects config keys:

    - ``labels`` (np.ndarray) — ground-truth binary labels (0=Faithful,
      1=Unfaithful), one per claim.
    """
    labels = np.asarray(config["labels"], dtype=int)
    n = len(labels)

    # Every claim is accepted ("green")
    all_green = ["green"] * n
    fdr_info = fdr_among_green(all_green, labels)

    hallucination_rate = float(labels.mean())

    return {
        "fdr": fdr_info["fdr"],
        "green_fraction": 1.0,
        "n_green": n,
        "hallucination_rate": hallucination_rate,
        "n_claims": n,
    }


# ---------------------------------------------------------------------------
# Baseline: ConfLVLM-style threshold
# ---------------------------------------------------------------------------

def _baseline_conflvlm(config: dict) -> dict:
    """Baseline: single global confidence threshold (ConfLVLM-style).

    Accepts claims whose verifier score exceeds a fixed threshold chosen to
    achieve ``target_fdr`` on the calibration split.  Unlike the conformal
    approach, this is a non-adaptive threshold — it doesn't adjust per
    pathology group or re-calibrate p-values.

    Expects config keys:

    - ``scores`` (np.ndarray) — verifier scores.
    - ``labels`` (np.ndarray) — ground-truth binary labels.
    - ``cal_fraction`` (float, optional) — fraction used for calibration.
      Default 0.5.
    - ``target_fdr`` (float, optional) — desired FDR on calibration set.
      Default 0.05.
    """
    scores = np.asarray(config["scores"], dtype=float)
    labels = np.asarray(config["labels"], dtype=int)
    cal_frac = float(config.get("cal_fraction", 0.5))
    target_fdr = float(config.get("target_fdr", 0.05))

    n_cal = int(len(scores) * cal_frac)
    cal_scores = scores[:n_cal]
    cal_labels = labels[:n_cal]
    test_scores = scores[n_cal:]
    test_labels = labels[n_cal:]

    # Find the threshold on calibration set that gives FDR <= target_fdr
    # Sweep thresholds from high to low (more accepting as we lower the bar)
    best_threshold = 1.0
    for tau in np.linspace(0.99, 0.0, 200):
        accepted_mask = cal_scores >= tau
        n_acc = int(accepted_mask.sum())
        if n_acc == 0:
            continue
        fdr_cal = float((cal_labels[accepted_mask] == 1).sum()) / n_acc
        if fdr_cal <= target_fdr:
            best_threshold = tau
            break

    # Evaluate on test split
    test_accepted = test_scores >= best_threshold
    test_labels_triage = ["green" if a else "red" for a in test_accepted]

    fdr_info = fdr_among_green(test_labels_triage, test_labels)
    gf = green_claim_fraction(test_labels_triage)

    return {
        "threshold": float(best_threshold),
        "fdr": fdr_info["fdr"],
        "green_fraction": gf,
        "n_green": fdr_info["n_green"],
        "n_test": len(test_labels),
        "target_fdr": target_fdr,
    }


# ---------------------------------------------------------------------------
# Baseline: PRM-style best-of-N reranking without conformal triage
# ---------------------------------------------------------------------------

def _baseline_prm_style(config: dict) -> dict:
    """Baseline: process-reward-model-style reranking without conformal triage.

    Selects the highest-scoring candidate for each report using the verifier
    score as a direct reward, then marks every claim in the selected report
    green.  No FDR-controlled triage is applied.

    This tests whether best-of-N selection alone (without conformal coverage)
    is sufficient for safe deployment.

    Expects config keys:

    - ``candidates_per_report`` (list[list[dict]]) — outer list indexed by
      report; inner list is N candidate dicts, each with ``"score"`` (float)
      and ``"labels"`` (list[int], ground-truth faithfulness per claim).
    """
    candidates_per_report: list[list[dict]] = config["candidates_per_report"]

    all_gt: list[int] = []
    all_triage: list[str] = []

    for candidates in candidates_per_report:
        if not candidates:
            continue
        # Pick candidate with highest verifier score
        best = max(candidates, key=lambda c: float(c.get("score", 0.0)))
        claim_labels: list[int] = best.get("labels", [])
        for lbl in claim_labels:
            all_gt.append(int(lbl))
            all_triage.append("green")  # PRM: all selected claims are trusted

    gt_arr = np.array(all_gt, dtype=int)
    fdr_info = fdr_among_green(all_triage, gt_arr)
    hallucination_rate = float(gt_arr.mean()) if len(gt_arr) > 0 else float("nan")

    return {
        "fdr": fdr_info["fdr"],
        "green_fraction": 1.0,
        "n_claims": len(all_gt),
        "n_reports": len(candidates_per_report),
        "hallucination_rate": hallucination_rate,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BASELINE_REGISTRY: dict[str, Any] = {
    "no_verification": _baseline_no_verification,
    "conflvlm": _baseline_conflvlm,
    "prm_style": _baseline_prm_style,
}


def run_baseline(baseline_name: str, config: dict) -> dict:
    """Run a named baseline experiment.

    Dispatches to the corresponding baseline function.  Available baselines:

    - ``"no_verification"`` — accept all claims; measures natural hallucination
      rate with zero verification overhead.
    - ``"conflvlm"``        — single global confidence threshold (no conformal
      calibration or pathology stratification).
    - ``"prm_style"``       — PRM-style best-of-N reranking with no triage;
      every claim in the selected report is trusted unconditionally.

    Args:
        baseline_name: One of the strings listed above.
        config: Configuration dict forwarded verbatim to the baseline function.
            See individual function docstrings for required keys.

    Returns:
        dict of metrics from the named baseline.  Always contains a
        ``"baseline_name"`` key for traceability.

    Raises:
        ValueError: If ``baseline_name`` is not registered.
    """
    if baseline_name not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {sorted(_BASELINE_REGISTRY)}"
        )

    logger.info(f"Running baseline: {baseline_name}")
    result = _BASELINE_REGISTRY[baseline_name](config)
    result["baseline_name"] = baseline_name
    return result
