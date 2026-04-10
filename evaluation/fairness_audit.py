"""Fairness audit for ClaimGuard-CXR.

Computes hallucination rate, green fraction, and FDR stratified by
demographic subgroups (sex, age bucket, race) and reports the maximum
disparity across groups.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from verifact.inference.conformal_triage import TriageResult
from verifact.evaluation.metrics import fdr_among_green, green_claim_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hallucination_rate(gt_labels: np.ndarray) -> float:
    """Fraction of claims that are unfaithful (gt_label == 1)."""
    if len(gt_labels) == 0:
        return float("nan")
    return float((gt_labels == 1).mean())


def _metrics_for_group(
    triage_results: list[TriageResult],
    gt_labels: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute per-group metrics using the provided boolean mask."""
    if mask.sum() == 0:
        return {
            "n": 0,
            "green_fraction": float("nan"),
            "fdr": float("nan"),
            "hallucination_rate": float("nan"),
        }

    group_results = [triage_results[i] for i in np.where(mask)[0]]
    group_gt = gt_labels[mask]
    group_labels = [r.label for r in group_results]

    fdr_info = fdr_among_green(group_labels, group_gt)
    gf = green_claim_fraction(group_labels)
    hr = _hallucination_rate(group_gt)

    return {
        "n": int(mask.sum()),
        "green_fraction": gf,
        "fdr": fdr_info["fdr"],
        "hallucination_rate": hr,
    }


def _max_disparity(group_metrics: dict[str, dict], metric: str) -> float:
    """Return the max absolute difference across groups for a single metric."""
    values = [
        v[metric]
        for v in group_metrics.values()
        if not np.isnan(v[metric])
    ]
    if len(values) < 2:
        return float("nan")
    return float(max(values) - min(values))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stratified_metrics(
    triage_results: list[TriageResult],
    gt_labels: np.ndarray,
    demographics: dict[str, np.ndarray],
) -> dict:
    """Compute FDR, green fraction, and hallucination rate stratified by demographics.

    Evaluates fairness across up to three demographic axes: ``sex``,
    ``age_bucket``, and ``race``.  For each axis present in ``demographics``,
    the function groups claims by unique category values, computes per-group
    metrics, and reports the maximum absolute disparity.

    Args:
        triage_results: List of :class:`~verifact.inference.conformal_triage.TriageResult`
            objects, one per claim.
        gt_labels: Ground-truth binary array (0=Faithful, 1=Unfaithful),
            same length as ``triage_results``.
        demographics: Mapping from demographic axis name (``"sex"``,
            ``"age_bucket"``, ``"race"``, or any other string) to a 1-D
            array of category labels for each claim.  Arrays must have the
            same length as ``triage_results``.

    Returns:
        dict with one key per demographic axis found in ``demographics``,
        each containing:

        - ``"groups"`` — dict of category_value -> per-group metric dict
          (keys: ``"n"``, ``"green_fraction"``, ``"fdr"``,
          ``"hallucination_rate"``)
        - ``"max_disparity"`` — dict of metric_name -> max absolute
          difference across groups

        Plus a top-level ``"overall"`` key with the same metric dict
        computed over all claims.
    """
    gt_arr = np.asarray(gt_labels, dtype=int)
    n = len(triage_results)

    if len(gt_arr) != n:
        raise ValueError(
            f"gt_labels length {len(gt_arr)} does not match "
            f"triage_results length {n}"
        )

    # Overall metrics
    all_labels = [r.label for r in triage_results]
    fdr_all = fdr_among_green(all_labels, gt_arr)
    result: dict = {
        "overall": {
            "n": n,
            "green_fraction": green_claim_fraction(all_labels),
            "fdr": fdr_all["fdr"],
            "hallucination_rate": _hallucination_rate(gt_arr),
        }
    }

    _metric_keys = ["green_fraction", "fdr", "hallucination_rate"]

    for axis_name, axis_values in demographics.items():
        axis_arr = np.asarray(axis_values)
        if len(axis_arr) != n:
            raise ValueError(
                f"demographics['{axis_name}'] has length {len(axis_arr)}, "
                f"expected {n}"
            )

        unique_cats = np.unique(axis_arr)
        group_metrics: dict[str, dict] = {}
        for cat in unique_cats:
            mask = axis_arr == cat
            group_metrics[str(cat)] = _metrics_for_group(
                triage_results, gt_arr, mask
            )

        max_disparity = {
            metric: _max_disparity(group_metrics, metric)
            for metric in _metric_keys
        }

        result[axis_name] = {
            "groups": group_metrics,
            "max_disparity": max_disparity,
        }

    return result
