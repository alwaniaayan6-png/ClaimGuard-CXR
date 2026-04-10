"""
Evaluation metrics for ClaimGuard-CXR.

Covers hallucination detection at the claim level, laterality error analysis,
triage false discovery rates, calibration diagnostics, and cluster-robust
statistical inference (bootstrap + permutation) that respects the
patient -> report -> claim hierarchy.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
VerdictList = List[str]
ClaimDictList = List[Dict[str, Any]]
TriageLabelArray = Union[List[str], np.ndarray]
GroundTruthArray = Union[List[int], np.ndarray]
PatientIdArray = Union[List[str], np.ndarray]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HALLUCINATION_VERDICTS = {"Contradicted", "Insufficient Evidence"}
_FAITHFUL_VERDICT = "Supported"
_LATERALITY_KEYWORDS = {"left", "right"}


# ---------------------------------------------------------------------------
# 1. Claim-level hallucination metrics
# ---------------------------------------------------------------------------

def claim_hallucination_metrics(
    predictions: VerdictList,
    ground_truth: VerdictList,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for hallucination detection at the claim level.

    A hallucination is any verdict of "Contradicted" or "Insufficient Evidence".
    "Supported" is treated as the negative (faithful) class.

    Parameters
    ----------
    predictions:
        Model-predicted verdict strings for each claim.
    ground_truth:
        Human-annotated verdict strings for each claim.

    Returns
    -------
    dict with keys:
        - ``precision``  (float)
        - ``recall``     (float)
        - ``f1``         (float)
        - ``n_claims``   (int)
        - ``n_hallucinated_gt`` (int) — ground-truth hallucinations
        - ``n_hallucinated_pred`` (int) — predicted hallucinations
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same length, "
            f"got {len(predictions)} vs {len(ground_truth)}"
        )

    pred_binary = [1 if v in _HALLUCINATION_VERDICTS else 0 for v in predictions]
    gt_binary = [1 if v in _HALLUCINATION_VERDICTS else 0 for v in ground_truth]

    pred_arr = np.array(pred_binary)
    gt_arr = np.array(gt_binary)

    # zero_division=0 matches standard behaviour when there are no positives
    precision = float(precision_score(gt_arr, pred_arr, zero_division=0))
    recall = float(recall_score(gt_arr, pred_arr, zero_division=0))
    f1 = float(f1_score(gt_arr, pred_arr, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_claims": len(predictions),
        "n_hallucinated_gt": int(gt_arr.sum()),
        "n_hallucinated_pred": int(pred_arr.sum()),
    }


# ---------------------------------------------------------------------------
# 2. Laterality error rate
# ---------------------------------------------------------------------------

def laterality_error_rate(claims: ClaimDictList) -> float:
    """Compute the rate of laterality errors among claims that mention left/right.

    A laterality error is a claim that (a) mentions a laterality keyword
    ("left" or "right", case-insensitive) AND (b) has a verdict of
    "Contradicted" — indicating the model or report got the side wrong.

    Parameters
    ----------
    claims:
        List of dicts, each with at least:
        - ``"text"``    (str) — the claim text
        - ``"verdict"`` (str) — one of "Supported", "Contradicted",
          "Insufficient Evidence"

    Returns
    -------
    float
        ``n_laterality_errors / n_laterality_claims``.
        Returns 0.0 if no claims mention a laterality keyword.
    """
    laterality_claims: List[Dict[str, Any]] = []
    for c in claims:
        text_lower = c["text"].lower()
        if any(kw in text_lower for kw in _LATERALITY_KEYWORDS):
            laterality_claims.append(c)

    if not laterality_claims:
        return 0.0

    n_errors = sum(
        1 for c in laterality_claims if c["verdict"] == "Contradicted"
    )
    return n_errors / len(laterality_claims)


# ---------------------------------------------------------------------------
# 3. False discovery rate among green-triaged claims
# ---------------------------------------------------------------------------

def fdr_among_green(
    triage_labels: TriageLabelArray,
    ground_truth: GroundTruthArray,
) -> Dict[str, Union[float, int]]:
    """Compute the false discovery rate (FDR) among claims triaged as "green".

    Green is the low-risk label; clinicians are most likely to act on green
    claims without further review, so FDR there is the safety-critical metric.

    Parameters
    ----------
    triage_labels:
        Array-like of strings: "green", "yellow", or "red" for each claim.
    ground_truth:
        Array-like of ints: 0 = Faithful, 1 = Unfaithful (hallucinated).

    Returns
    -------
    dict with keys:
        - ``fdr``              (float) — false discoveries / green claims
        - ``n_green``          (int)   — total green-labeled claims
        - ``n_false_discoveries`` (int) — green claims that are actually unfaithful
    """
    triage_arr = np.asarray(triage_labels, dtype=str)
    gt_arr = np.asarray(ground_truth, dtype=int)

    if len(triage_arr) != len(gt_arr):
        raise ValueError(
            f"triage_labels and ground_truth must have the same length, "
            f"got {len(triage_arr)} vs {len(gt_arr)}"
        )

    green_mask = triage_arr == "green"
    n_green = int(green_mask.sum())

    if n_green == 0:
        return {"fdr": 0.0, "n_green": 0, "n_false_discoveries": 0}

    n_false_discoveries = int((gt_arr[green_mask] == 1).sum())
    fdr = n_false_discoveries / n_green

    return {
        "fdr": fdr,
        "n_green": n_green,
        "n_false_discoveries": n_false_discoveries,
    }


# ---------------------------------------------------------------------------
# 4. Green claim fraction
# ---------------------------------------------------------------------------

def green_claim_fraction(triage_labels: TriageLabelArray) -> float:
    """Return the fraction of claims labeled "green".

    Parameters
    ----------
    triage_labels:
        Array-like of strings: "green", "yellow", or "red".

    Returns
    -------
    float
        Fraction in [0, 1]. Returns 0.0 on empty input.
    """
    arr = np.asarray(triage_labels, dtype=str)
    if arr.size == 0:
        return 0.0
    return float((arr == "green").sum() / arr.size)


# ---------------------------------------------------------------------------
# 5. Patient-cluster bootstrap
# ---------------------------------------------------------------------------

def patient_cluster_bootstrap(
    metric_fn: Callable[[Any], float],
    data: Any,
    patient_ids: PatientIdArray,
    n_iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence intervals by resampling patients (cluster bootstrap).

    Preserves the nested patient -> report -> claim structure: when a patient
    is resampled, ALL of their claims are included together, avoiding the
    independence violation that naive claim-level resampling would introduce.

    Parameters
    ----------
    metric_fn:
        Callable that accepts a *subset* of ``data`` (same type/shape as
        ``data``, filtered to the resampled patients) and returns a float.
        For numpy arrays this subset is an index-filtered array.
        For lists, it is a list of selected items.
    data:
        The full dataset. Must be index-subscriptable via a boolean or integer
        numpy array (e.g., ``np.ndarray``, ``list``).
    patient_ids:
        Array-like of patient ID strings/ints, one per claim/row in ``data``.
        Must be the same length as ``data``.
    n_iterations:
        Number of bootstrap resamples. Default 2000.
    confidence:
        Coverage for the percentile interval, e.g. 0.95 for 95% CI.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - ``mean``      (float)
        - ``ci_lower``  (float)
        - ``ci_upper``  (float)
        - ``std``       (float)
    """
    rng = np.random.default_rng(seed)

    patient_ids_arr = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids_arr)
    n_patients = len(unique_patients)

    # Pre-build patient -> index mapping for efficiency
    patient_to_indices: Dict[Any, np.ndarray] = {
        pid: np.where(patient_ids_arr == pid)[0]
        for pid in unique_patients
    }

    bootstrap_stats: List[float] = []
    for _ in range(n_iterations):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        # Gather claim indices for the resampled patients (with repetition)
        selected_indices = np.concatenate(
            [patient_to_indices[pid] for pid in sampled_patients]
        )
        if isinstance(data, np.ndarray):
            resampled_data = data[selected_indices]
        else:
            resampled_data = [data[i] for i in selected_indices]

        try:
            stat = metric_fn(resampled_data)
            bootstrap_stats.append(float(stat))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"metric_fn raised an exception on a bootstrap resample: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if not bootstrap_stats:
        raise RuntimeError("metric_fn failed on all bootstrap resamples.")

    stats_arr = np.array(bootstrap_stats)
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(stats_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(stats_arr, 100 * (1 - alpha / 2)))

    return {
        "mean": float(stats_arr.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(stats_arr.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# 6. Reliability diagram (calibration)
# ---------------------------------------------------------------------------

def reliability_diagram(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute data for a reliability (calibration) diagram.

    Bins predicted probabilities into ``n_bins`` equal-width bins and computes
    the mean confidence and mean accuracy within each bin.  Also returns
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Parameters
    ----------
    scores:
        Predicted probabilities (floats in [0, 1]) that a claim is
        unfaithful / hallucinated.
    labels:
        Ground-truth binary labels: 0 = Faithful, 1 = Unfaithful.
    n_bins:
        Number of equal-width bins. Default 10.

    Returns
    -------
    dict with keys:
        - ``bin_edges``  (np.ndarray, shape (n_bins+1,))
        - ``bin_accs``   (np.ndarray, shape (n_bins,)) — mean accuracy per bin
        - ``bin_confs``  (np.ndarray, shape (n_bins,)) — mean confidence per bin
        - ``bin_counts`` (np.ndarray, shape (n_bins,), int) — samples per bin
        - ``ece``        (float) — Expected Calibration Error
        - ``mce``        (float) — Maximum Calibration Error
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    if scores_arr.ndim != 1 or labels_arr.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays.")
    if len(scores_arr) != len(labels_arr):
        raise ValueError(
            f"scores and labels must have the same length, "
            f"got {len(scores_arr)} vs {len(labels_arr)}"
        )
    if not (0 <= scores_arr.min() and scores_arr.max() <= 1):
        raise ValueError("scores must be probabilities in [0, 1].")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge in the last bin
        if i < n_bins - 1:
            mask = (scores_arr >= lo) & (scores_arr < hi)
        else:
            mask = (scores_arr >= lo) & (scores_arr <= hi)

        n = int(mask.sum())
        bin_counts[i] = n
        if n > 0:
            bin_accs[i] = float(labels_arr[mask].mean())
            bin_confs[i] = float(scores_arr[mask].mean())

    n_total = len(scores_arr)
    gap = np.abs(bin_accs - bin_confs)

    ece = float(np.sum(bin_counts * gap) / n_total) if n_total > 0 else 0.0
    # MCE only over non-empty bins
    non_empty = bin_counts > 0
    mce = float(gap[non_empty].max()) if non_empty.any() else 0.0

    return {
        "bin_edges": bin_edges,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "ece": ece,
        "mce": mce,
    }


# ---------------------------------------------------------------------------
# 7. Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    metric_fn: Callable[[Any, Any], float],
    data_a: Any,
    data_b: Any,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-sample permutation test for a difference in a scalar metric.

    Under H0 the two groups are exchangeable; labels are permuted and the
    metric difference is re-computed on each permutation.

    Parameters
    ----------
    metric_fn:
        Callable ``(group_a_data, group_b_data) -> float``.
        The observed difference is ``metric_fn(data_a, data_b)``.
    data_a:
        Data for group A. Must support numpy fancy indexing.
    data_b:
        Data for group B. Must support numpy fancy indexing.
    n_permutations:
        Number of permutation draws. Default 10 000.
    seed:
        Random seed.

    Returns
    -------
    dict with keys:
        - ``observed_diff``    (float) — metric_fn(data_a, data_b)
        - ``p_value``          (float) — two-tailed p-value
        - ``null_distribution`` (np.ndarray) — permuted differences
    """
    rng = np.random.default_rng(seed)

    observed_diff = float(metric_fn(data_a, data_b))

    # Combine data; we'll resplit by random permutation
    if isinstance(data_a, np.ndarray) and isinstance(data_b, np.ndarray):
        combined = np.concatenate([data_a, data_b], axis=0)
        n_a = len(data_a)

        def _split(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return combined[idx[:n_a]], combined[idx[n_a:]]

    else:
        combined_list = list(data_a) + list(data_b)
        n_a = len(list(data_a))
        combined = np.array(combined_list, dtype=object)

        def _split(idx: np.ndarray) -> tuple[list, list]:  # type: ignore[misc]
            return list(combined[idx[:n_a]]), list(combined[idx[n_a:]])

    n_total = len(combined)
    null_distribution = np.empty(n_permutations)

    for k in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        group_a, group_b = _split(perm_idx)
        null_distribution[k] = float(metric_fn(group_a, group_b))

    # Two-tailed p-value: fraction of null diffs at least as extreme as observed
    p_value = float(
        (np.abs(null_distribution) >= np.abs(observed_diff)).mean()
    )

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "null_distribution": null_distribution,
    }


# ---------------------------------------------------------------------------
# 8. Master entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    triage_results: List[Dict[str, Any]],
    ground_truth_labels: GroundTruthArray,
    patient_ids: PatientIdArray,
    pathology_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """Compute the full ClaimGuard-CXR evaluation suite.

    Parameters
    ----------
    triage_results:
        List of per-claim result dicts.  Each dict must contain:
        - ``"triage_label"``  (str)   — "green", "yellow", or "red"
        - ``"verdict"``       (str)   — "Supported", "Contradicted",
          "Insufficient Evidence"
        - ``"text"``          (str)   — claim text
        - ``"score"``         (float) — verifier probability in [0, 1]
        Optional:
        - ``"gt_verdict"``    (str)   — human verdict (needed for claim-level
          hallucination metrics and calibration)
    ground_truth_labels:
        Binary array (0=Faithful, 1=Unfaithful), one per entry in
        ``triage_results``.
    patient_ids:
        Patient ID per claim, same length as ``triage_results``.
    pathology_groups:
        Optional mapping from pathology name to list of claim indices for
        per-pathology FDR breakdown. E.g.::

            {"pneumonia": [0, 4, 7], "effusion": [1, 2, 5]}

    Returns
    -------
    dict with keys:
        - ``"hallucination"``          — from :func:`claim_hallucination_metrics`
          (only when ``gt_verdict`` is present in results)
        - ``"laterality_error_rate"``  — float
        - ``"fdr_green"``              — from :func:`fdr_among_green`
        - ``"green_fraction"``         — float
        - ``"calibration"``            — from :func:`reliability_diagram`
          (only when scores are present)
        - ``"fdr_green_bootstrap"``    — cluster-bootstrapped CI for green FDR
        - ``"per_pathology_fdr"``      — dict keyed by pathology name
          (only when ``pathology_groups`` is provided)
        - ``"n_claims"``               — int
    """
    gt_arr = np.asarray(ground_truth_labels, dtype=int)
    triage_labels = [r["triage_label"] for r in triage_results]

    results: Dict[str, Any] = {
        "n_claims": len(triage_results),
    }

    # --- Laterality error rate ---
    results["laterality_error_rate"] = laterality_error_rate(triage_results)

    # --- Triage FDR ---
    results["fdr_green"] = fdr_among_green(triage_labels, gt_arr)
    results["green_fraction"] = green_claim_fraction(triage_labels)

    # --- Claim-level hallucination metrics (requires gt_verdict field) ---
    if triage_results and "gt_verdict" in triage_results[0]:
        pred_verdicts = [r["verdict"] for r in triage_results]
        gt_verdicts = [r["gt_verdict"] for r in triage_results]
        results["hallucination"] = claim_hallucination_metrics(
            pred_verdicts, gt_verdicts
        )

    # --- Calibration (requires score field) ---
    if triage_results and "score" in triage_results[0]:
        scores = np.array([r["score"] for r in triage_results], dtype=float)
        results["calibration"] = reliability_diagram(scores, gt_arr)

    # --- Cluster-bootstrapped CI for green FDR ---
    def _fdr_fn(resampled: List[Dict[str, Any]]) -> float:
        labels_r = np.array([r["triage_label"] for r in resampled], dtype=str)
        gt_r = np.array([r["_gt"] for r in resampled], dtype=int)
        return fdr_among_green(labels_r, gt_r)["fdr"]

    # Attach ground-truth labels into each claim dict temporarily
    enriched = [
        {**r, "_gt": int(g)} for r, g in zip(triage_results, gt_arr)
    ]
    try:
        results["fdr_green_bootstrap"] = patient_cluster_bootstrap(
            metric_fn=_fdr_fn,
            data=enriched,
            patient_ids=patient_ids,
            n_iterations=2000,
            confidence=0.95,
            seed=42,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Cluster bootstrap failed: {exc}",
            RuntimeWarning,
            stacklevel=1,
        )
        results["fdr_green_bootstrap"] = None

    # --- Per-pathology FDR breakdown ---
    if pathology_groups:
        per_path: Dict[str, Any] = {}
        for pathology, indices in pathology_groups.items():
            idx_arr = np.array(indices, dtype=int)
            path_labels = [triage_labels[i] for i in idx_arr]
            path_gt = gt_arr[idx_arr]
            per_path[pathology] = fdr_among_green(path_labels, path_gt)
        results["per_pathology_fdr"] = per_path

    return results
