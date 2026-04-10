"""Label sensitivity analysis for ClaimGuard-CXR.

Quantifies how much evaluation metrics change when ground-truth labels come
from CheXbert alone versus an ensemble of CheXbert + NegBio + GPT-4.

High sensitivity means the choice of labeller materially affects conclusions,
which should be flagged in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from verifact.evaluation.metrics import (
    fdr_among_green,
    green_claim_fraction,
    claim_hallucination_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_metrics(
    predictions: list[str],
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute hallucination metrics given a binary label array.

    Args:
        predictions: Model verdict strings ("Supported", "Contradicted", …).
        labels: Binary ground-truth (0=Faithful, 1=Unfaithful).

    Returns:
        dict with ``precision``, ``recall``, ``f1``, ``hallucination_rate``.
    """
    gt_verdicts = [
        "Contradicted" if lbl == 1 else "Supported" for lbl in labels
    ]
    cm = claim_hallucination_metrics(predictions, gt_verdicts)
    return {
        "precision": cm["precision"],
        "recall": cm["recall"],
        "f1": cm["f1"],
        "hallucination_rate": float(labels.mean()),
    }


def _label_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compute agreement statistics between two binary label arrays."""
    n = len(a)
    agree = int((a == b).sum())
    cohen_k = _cohen_kappa(a, b)
    try:
        pcc, pcc_p = pearsonr(a.astype(float), b.astype(float))
    except Exception:
        pcc, pcc_p = float("nan"), float("nan")
    try:
        scc, scc_p = spearmanr(a.astype(float), b.astype(float))
    except Exception:
        scc, scc_p = float("nan"), float("nan")

    return {
        "agreement_rate": agree / n,
        "cohen_kappa": cohen_k,
        "pearson_r": float(pcc),
        "pearson_p": float(pcc_p),
        "spearman_r": float(scc),
        "spearman_p": float(scc_p),
    }


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = (a == b).mean()
    # Expected agreement by chance
    p_a1 = a.mean()
    p_b1 = b.mean()
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_label_sources(
    predictions: list[str],
    chexbert_labels: np.ndarray,
    ensemble_labels: np.ndarray,
) -> dict:
    """Report how evaluation metrics differ between CheXbert-only and ensemble labels.

    Computes hallucination metrics (precision, recall, F1) under each label
    source, the absolute metric deltas, and cross-label-source correlation.

    Args:
        predictions: Model verdict strings for each claim
            (e.g. ``"Supported"``, ``"Contradicted"``).
        chexbert_labels: Binary ground-truth from CheXbert alone
            (0=Faithful, 1=Unfaithful), shape (N,).
        ensemble_labels: Binary ground-truth from the CheXbert + NegBio +
            GPT-4 ensemble, shape (N,).

    Returns:
        dict with keys:

        - ``"chexbert"``   — metrics under CheXbert-only labels
        - ``"ensemble"``   — metrics under ensemble labels
        - ``"delta"``      — absolute difference (ensemble - chexbert) per
          metric
        - ``"agreement"``  — agreement statistics between the two label
          sources (``agreement_rate``, ``cohen_kappa``, ``pearson_r``,
          ``spearman_r``, …)
        - ``"sensitivity_flag"`` — True if any |delta| > 0.05
    """
    cb = np.asarray(chexbert_labels, dtype=int)
    ens = np.asarray(ensemble_labels, dtype=int)

    if len(cb) != len(ens):
        raise ValueError(
            f"chexbert_labels ({len(cb)}) and ensemble_labels ({len(ens)}) "
            "must have the same length"
        )
    if len(cb) != len(predictions):
        raise ValueError(
            f"predictions ({len(predictions)}) and label arrays ({len(cb)}) "
            "must have the same length"
        )

    chexbert_metrics = _binary_metrics(predictions, cb)
    ensemble_metrics = _binary_metrics(predictions, ens)

    delta: dict[str, float] = {
        key: ensemble_metrics[key] - chexbert_metrics[key]
        for key in chexbert_metrics
    }

    sensitivity_flag = any(abs(v) > 0.05 for v in delta.values())

    agreement = _label_agreement(cb, ens)

    return {
        "chexbert": chexbert_metrics,
        "ensemble": ensemble_metrics,
        "delta": delta,
        "agreement": agreement,
        "sensitivity_flag": sensitivity_flag,
    }
