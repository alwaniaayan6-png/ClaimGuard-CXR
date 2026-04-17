"""Statistical helpers for ClaimGuard-Bench-Grounded evaluation.

Everything returns 95% bootstrap CIs by default — no naked point estimates
in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass
class MetricWithCI:
    value: float
    ci_low: float
    ci_high: float

    def fmt(self, precision: int = 3) -> str:
        return f"{self.value:.{precision}f} [{self.ci_low:.{precision}f}, {self.ci_high:.{precision}f}]"


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MetricWithCI:
    """Compute 95% CI via paired bootstrap over one or more aligned arrays.

    All arrays are resampled with the SAME index vector on each bootstrap
    iteration, so paired relationships (y_true[i] <-> y_pred[i] <-> y_score[i])
    are preserved.
    """
    if rng is None:
        rng = np.random.default_rng(17)
    arrays = tuple(np.asarray(a) for a in arrays)
    if not arrays:
        raise ValueError("bootstrap_ci requires at least one array")
    n = arrays[0].size
    for a in arrays:
        if a.size != n:
            raise ValueError("all arrays passed to bootstrap_ci must have the same length")

    point = metric_fn(*arrays)
    estimates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            estimates[b] = metric_fn(*[a[idx] for a in arrays])
        except Exception:
            estimates[b] = np.nan
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return MetricWithCI(value=float(point), ci_low=float("nan"), ci_high=float("nan"))
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return MetricWithCI(value=float(point), ci_low=lo, ci_high=hi)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _recall_for_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    mask = y_true == cls
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == cls).mean())


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Dependency-light implementation: fall back to sklearn if available.
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        pass
    # Mann-Whitney U formulation. All positives appear in slots [0, n_pos) of
    # `all_scores` by construction of the concatenation order.
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, all_scores.size + 1)
    rank_pos_sum = float(ranks[: pos.size].sum())
    auc = (rank_pos_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def compute_verdict_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_contradicted_score: np.ndarray,
    n_boot: int = 1000,
) -> Dict[str, MetricWithCI]:
    """Core verdict metrics with properly-paired bootstrap CIs."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_contradicted_score = np.asarray(y_contradicted_score, dtype=float)

    out: Dict[str, MetricWithCI] = {}
    out["accuracy"] = bootstrap_ci(
        _accuracy, y_true, y_pred, n_boot=n_boot,
    )
    out["contradiction_recall"] = bootstrap_ci(
        lambda yt, yp: _recall_for_class(yt, yp, cls=1),
        y_true, y_pred, n_boot=n_boot,
    )
    out["supported_recall"] = bootstrap_ci(
        lambda yt, yp: _recall_for_class(yt, yp, cls=0),
        y_true, y_pred, n_boot=n_boot,
    )
    out["auroc"] = bootstrap_ci(
        _auroc, y_true, y_contradicted_score, n_boot=n_boot,
    )
    return out


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float]:
    """McNemar's test for paired accuracy comparison.

    Returns
    -------
    statistic, p_value
    """
    y_true = np.asarray(y_true)
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    b01 = int((a_correct & ~b_correct).sum())
    b10 = int((~a_correct & b_correct).sum())
    if b01 + b10 == 0:
        return 0.0, 1.0
    # Continuity-corrected chi-square (McNemar's exact variant available via scipy).
    try:
        from scipy.stats import chi2

        stat = ((abs(b01 - b10) - 1.0) ** 2) / (b01 + b10)
        p = 1.0 - float(chi2.cdf(stat, df=1))
        return float(stat), p
    except ImportError:
        # Binomial approximation without scipy.
        p = min(1.0, 2 * min(b01, b10) / (b01 + b10))
        return float(abs(b01 - b10)), float(p)


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> Tuple[float, float]:
    """DeLong test for paired AUROC comparison. Requires scipy.

    Returns (Z-statistic, two-sided p-value). NaN/1.0 if scipy unavailable.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        return float("nan"), 1.0

    def _compute_ground_truth_statistics(ground_truth: np.ndarray):
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    def _compute_midrank(x: np.ndarray) -> np.ndarray:
        J = np.argsort(x, kind="mergesort")
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    def _auc_cov(predictions_sorted_transposed: np.ndarray, label_1_count: int):
        m, n = label_1_count, predictions_sorted_transposed.shape[1] - label_1_count
        positive = predictions_sorted_transposed[:, :m]
        negative = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r] = _compute_midrank(positive[r])
            ty[r] = _compute_midrank(negative[r])
            tz[r] = _compute_midrank(predictions_sorted_transposed[r])

        aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        if np.ndim(sx) == 0:
            sx = np.array([[sx]])
            sy = np.array([[sy]])
        delongcov = sx / m + sy / n
        return aucs, delongcov

    order, label_1_count = _compute_ground_truth_statistics(y_true)
    predictions_sorted = np.vstack([score_a[order], score_b[order]])
    aucs, delongcov = _auc_cov(predictions_sorted, label_1_count)
    l = np.array([[1, -1]])
    z = float((l @ aucs) / np.sqrt(l @ delongcov @ l.T))
    p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
    return z, p


def per_site_summary(
    results_per_site: Dict[str, dict],
) -> Dict[str, Dict[str, MetricWithCI]]:
    """Dict of {site: {metric: MetricWithCI}} given per-site (y_true, y_pred, y_score)."""
    out: Dict[str, Dict[str, MetricWithCI]] = {}
    for site, r in results_per_site.items():
        out[site] = compute_verdict_metrics(
            r["y_true"], r["y_pred"], r["y_contradicted_score"],
        )
    return out
