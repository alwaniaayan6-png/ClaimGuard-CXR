"""Subgroup / fairness analysis.

Given per-claim verdicts + metadata, compute:
  - per-subgroup accuracy, contradiction recall, AUROC with CIs
  - parity gaps (max minus min) between subgroups
  - calibration curves per subgroup
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from claimguard_nmi.eval.metrics import MetricWithCI, compute_verdict_metrics


@dataclass
class SubgroupResult:
    subgroup_name: str
    subgroup_value: str
    n: int
    metrics: Dict[str, MetricWithCI]


def stratify_by(
    metadata: List[Dict[str, object]],
    key: str,
    bin_fn: Optional[Callable[[object], str]] = None,
) -> Dict[str, np.ndarray]:
    """Return {bucket_name: index_array} for stratification.

    If bin_fn is None, uses the raw value (coerced to str).
    """
    buckets: Dict[str, List[int]] = {}
    for i, meta in enumerate(metadata):
        val = meta.get(key) if isinstance(meta, dict) else None
        bucket = bin_fn(val) if bin_fn is not None else (str(val) if val is not None else "unknown")
        buckets.setdefault(bucket, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in buckets.items()}


def age_quartile(val) -> str:
    try:
        age = float(val)
    except (TypeError, ValueError):
        return "unknown"
    if age < 35:
        return "q1_lt35"
    if age < 55:
        return "q2_35_54"
    if age < 70:
        return "q3_55_69"
    return "q4_ge70"


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_contradicted_score: np.ndarray,
    metadata: List[Dict[str, object]],
    key: str,
    bin_fn: Optional[Callable[[object], str]] = None,
    n_boot: int = 500,
) -> List[SubgroupResult]:
    """Stratify and compute metrics per subgroup."""
    if not (len(metadata) == y_true.size == y_pred.size == y_contradicted_score.size):
        raise ValueError("y_true, y_pred, y_contradicted_score and metadata must all align")

    buckets = stratify_by(metadata, key, bin_fn)
    results: List[SubgroupResult] = []
    for bucket_name, idx in buckets.items():
        if idx.size < 20:
            # Too small to trust bootstrap CIs; skip but log.
            continue
        metrics = compute_verdict_metrics(
            y_true[idx], y_pred[idx], y_contradicted_score[idx], n_boot=n_boot,
        )
        results.append(SubgroupResult(
            subgroup_name=key,
            subgroup_value=bucket_name,
            n=int(idx.size),
            metrics=metrics,
        ))
    return results


def parity_gap(results: List[SubgroupResult], metric: str = "accuracy") -> float:
    """max(metric) - min(metric) across subgroups."""
    vals = [r.metrics[metric].value for r in results if metric in r.metrics]
    if not vals:
        return float("nan")
    return float(max(vals) - min(vals))
