"""Ablation study dispatcher for ClaimGuard-CXR.

Each ablation isolates one component of the pipeline to measure its
marginal contribution to overall faithfulness and triage performance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from verifact.evaluation.metrics import (
    compute_all_metrics,
    fdr_among_green,
    green_claim_fraction,
)
from verifact.inference.conformal_triage import ConformalClaimTriage, compute_fdr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual ablation functions
# ---------------------------------------------------------------------------

def _ablation_no_retrieval(config: dict) -> dict:
    """Ablation: verifier receives the claim but no retrieved evidence.

    Expects config keys:

    - ``scores_no_retrieval`` (np.ndarray) — verifier scores when evidence
      retrieval is disabled.
    - ``scores_full`` (np.ndarray) — verifier scores with full retrieval.
    - ``labels`` (np.ndarray) — ground-truth binary labels.
    - ``pathology_groups`` (np.ndarray) — per-claim pathology group.
    - ``report_ids`` (np.ndarray) — per-claim report ID.
    - ``alpha`` (float, optional) — target FDR, default 0.05.
    - ``cal_fraction`` (float, optional) — fraction for calibration, default 0.5.
    """
    alpha = float(config.get("alpha", 0.05))
    scores_nr = np.asarray(config["scores_no_retrieval"])
    scores_full = np.asarray(config["scores_full"])
    labels = np.asarray(config["labels"], dtype=int)
    pg = np.asarray(config["pathology_groups"])
    rids = np.asarray(config["report_ids"])

    def _eval(scores: np.ndarray, label: str) -> dict[str, Any]:
        n_cal = int(len(scores) * float(config.get("cal_fraction", 0.5)))
        triage = ConformalClaimTriage(alpha=alpha, min_group_size=50)
        triage.calibrate(
            scores=scores[:n_cal],
            labels=labels[:n_cal],
            pathology_groups=pg[:n_cal],
            report_ids=rids[:n_cal],
        )
        results = triage.triage(scores=scores[n_cal:], pathology_groups=pg[n_cal:])
        fdr_info = compute_fdr(results, labels[n_cal:])
        triage_labels = [r.label for r in results]
        return {
            "condition": label,
            "fdr": fdr_info["fdr"],
            "green_fraction": green_claim_fraction(triage_labels),
            "n_green": fdr_info["n_green"],
        }

    full_metrics = _eval(scores_full, "full_retrieval")
    nr_metrics = _eval(scores_nr, "no_retrieval")

    return {
        "full_retrieval": full_metrics,
        "no_retrieval": nr_metrics,
        "delta_fdr": nr_metrics["fdr"] - full_metrics["fdr"],
        "delta_green_fraction": nr_metrics["green_fraction"] - full_metrics["green_fraction"],
    }


def _ablation_random_negatives(config: dict) -> dict:
    """Ablation: replace hard negatives with random negatives during training.

    Expects config keys:

    - ``scores_hard_neg`` (np.ndarray) — verifier scores from model trained on
      hard negatives.
    - ``scores_random_neg`` (np.ndarray) — verifier scores from model trained
      on random negatives.
    - ``labels``, ``pathology_groups``, ``report_ids`` — same as above.
    - ``alpha`` (float, optional).
    - ``cal_fraction`` (float, optional).
    """
    alpha = float(config.get("alpha", 0.05))
    labels = np.asarray(config["labels"], dtype=int)
    pg = np.asarray(config["pathology_groups"])
    rids = np.asarray(config["report_ids"])
    n_cal = int(len(labels) * float(config.get("cal_fraction", 0.5)))

    def _eval(scores: np.ndarray, label: str) -> dict[str, Any]:
        triage = ConformalClaimTriage(alpha=alpha, min_group_size=50)
        triage.calibrate(
            scores=scores[:n_cal],
            labels=labels[:n_cal],
            pathology_groups=pg[:n_cal],
            report_ids=rids[:n_cal],
        )
        results = triage.triage(scores=scores[n_cal:], pathology_groups=pg[n_cal:])
        fdr_info = compute_fdr(results, labels[n_cal:])
        triage_labels = [r.label for r in results]
        return {
            "condition": label,
            "fdr": fdr_info["fdr"],
            "green_fraction": green_claim_fraction(triage_labels),
        }

    hard_metrics = _eval(np.asarray(config["scores_hard_neg"]), "hard_negatives")
    rand_metrics = _eval(np.asarray(config["scores_random_neg"]), "random_negatives")

    return {
        "hard_negatives": hard_metrics,
        "random_negatives": rand_metrics,
        "delta_fdr": rand_metrics["fdr"] - hard_metrics["fdr"],
        "delta_green_fraction": rand_metrics["green_fraction"] - hard_metrics["green_fraction"],
    }


def _ablation_n4_vs_n8(config: dict) -> dict:
    """Ablation: N=4 best-of-N vs N=8.

    Expects config keys:

    - ``results_n4`` (list[dict]) — per-report result dicts for N=4.
    - ``results_n8`` (list[dict]) — per-report result dicts for N=8.

    Each result dict should contain ``verifier_score``, ``radgraph_f1``,
    and ``chexbert_f1``.
    """
    from verifact.evaluation.reward_hacking_check import check_reward_hacking

    results_n4 = config["results_n4"]
    results_n8 = config["results_n8"]

    hacking_report = check_reward_hacking({4: results_n4, 8: results_n8})

    def _agg(results: list[dict]) -> dict[str, float]:
        def _mean(key: str) -> float:
            vals = [r[key] for r in results if key in r]
            return float(np.mean(vals)) if vals else float("nan")
        return {
            "verifier_score": _mean("verifier_score"),
            "radgraph_f1": _mean("radgraph_f1"),
            "chexbert_f1": _mean("chexbert_f1"),
        }

    return {
        "n4": _agg(results_n4),
        "n8": _agg(results_n8),
        "hacking_report": hacking_report,
    }


def _ablation_global_vs_group(config: dict) -> dict:
    """Ablation: global (pooled) conformal threshold vs per-pathology-group thresholds.

    Expects config keys:

    - ``scores``, ``labels``, ``pathology_groups``, ``report_ids`` — arrays.
    - ``alpha`` (float, optional).
    - ``cal_fraction`` (float, optional).
    """
    alpha = float(config.get("alpha", 0.05))
    scores = np.asarray(config["scores"])
    labels = np.asarray(config["labels"], dtype=int)
    pg = np.asarray(config["pathology_groups"])
    rids = np.asarray(config["report_ids"])
    n_cal = int(len(scores) * float(config.get("cal_fraction", 0.5)))

    # Group-stratified (normal operation)
    triage_group = ConformalClaimTriage(alpha=alpha, min_group_size=50)
    triage_group.calibrate(
        scores=scores[:n_cal],
        labels=labels[:n_cal],
        pathology_groups=pg[:n_cal],
        report_ids=rids[:n_cal],
    )
    group_results = triage_group.triage(scores=scores[n_cal:], pathology_groups=pg[n_cal:])
    group_fdr = compute_fdr(group_results, labels[n_cal:])
    group_gf = green_claim_fraction([r.label for r in group_results])

    # Global (pool all groups into one by using a dummy constant group label)
    dummy_pg = np.full(len(scores), "all", dtype=object)
    triage_global = ConformalClaimTriage(alpha=alpha, min_group_size=1)
    triage_global.calibrate(
        scores=scores[:n_cal],
        labels=labels[:n_cal],
        pathology_groups=dummy_pg[:n_cal],
        report_ids=rids[:n_cal],
    )
    global_results = triage_global.triage(scores=scores[n_cal:], pathology_groups=dummy_pg[n_cal:])
    global_fdr = compute_fdr(global_results, labels[n_cal:])
    global_gf = green_claim_fraction([r.label for r in global_results])

    return {
        "global": {
            "fdr": global_fdr["fdr"],
            "green_fraction": global_gf,
            "n_green": global_fdr["n_green"],
        },
        "group_stratified": {
            "fdr": group_fdr["fdr"],
            "green_fraction": group_gf,
            "n_green": group_fdr["n_green"],
        },
        "delta_fdr": global_fdr["fdr"] - group_fdr["fdr"],
        "delta_green_fraction": global_gf - group_gf,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ABLATION_REGISTRY: dict[str, Any] = {
    "no_retrieval": _ablation_no_retrieval,
    "random_negatives": _ablation_random_negatives,
    "n4_vs_n8": _ablation_n4_vs_n8,
    "global_vs_group": _ablation_global_vs_group,
}


def run_ablation(ablation_name: str, config: dict) -> dict:
    """Run a named ablation experiment.

    Dispatches to the corresponding ablation function.  Available ablations:

    - ``"no_retrieval"``     — disables evidence retrieval.
    - ``"random_negatives"`` — uses random rather than hard negatives in training.
    - ``"n4_vs_n8"``         — compares best-of-4 vs best-of-8 selection.
    - ``"global_vs_group"``  — global pooled threshold vs per-pathology thresholds.

    Args:
        ablation_name: One of the strings listed above.
        config: Configuration dict forwarded verbatim to the ablation function.
            See individual function docstrings for required keys.

    Returns:
        dict of metrics from the named ablation.  Always contains a
        ``"ablation_name"`` key for traceability.

    Raises:
        ValueError: If ``ablation_name`` is not registered.
    """
    if ablation_name not in _ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown ablation '{ablation_name}'. "
            f"Available: {sorted(_ABLATION_REGISTRY)}"
        )

    logger.info(f"Running ablation: {ablation_name}")
    result = _ABLATION_REGISTRY[ablation_name](config)
    result["ablation_name"] = ablation_name
    return result
