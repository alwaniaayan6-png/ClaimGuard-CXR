"""Regression tests for eval/metrics.py. Specifically exercises the
bootstrap-CI pairing and AUROC-fallback correctness bugs that v2 review
caught."""
from __future__ import annotations

import numpy as np

from claimguard_nmi.eval.metrics import (
    _auroc,
    bootstrap_ci,
    compute_verdict_metrics,
)


def test_bootstrap_ci_pairs_two_arrays_correctly():
    # y_true and y_pred are perfectly paired — accuracy should be 1.0 both
    # as a point estimate and every bootstrap draw.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    r = bootstrap_ci(lambda t, p: float((t == p).mean()), y_true, y_pred, n_boot=200)
    assert r.value == 1.0
    assert r.ci_low == 1.0 and r.ci_high == 1.0


def test_bootstrap_ci_detects_random_pairing():
    # If we swap y_pred with an uncorrelated array, accuracy point estimate
    # should fall near 0.5, not stay at 1.0.
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    r = bootstrap_ci(lambda t, p: float((t == p).mean()), y_true, y_pred, n_boot=200)
    assert 0.3 < r.value < 0.7


def test_bootstrap_ci_rejects_mismatched_lengths():
    try:
        bootstrap_ci(lambda a, b: 0.0, np.array([1, 2, 3]), np.array([1, 2]))
        assert False, "should have raised"
    except ValueError:
        pass


def test_auroc_fallback_matches_sklearn_if_available():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=1000)
    # Gap the scores so AUROC is well-defined and far from 0.5
    scores = rng.normal(loc=y.astype(float), scale=0.5)
    auc_primary = _auroc(y, scores)
    # Directly compute via the dependency-light formula to cross-check.
    pos = scores[y == 1]
    neg = scores[y == 0]
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, all_scores.size + 1)
    rank_pos_sum = float(ranks[: pos.size].sum())
    auc_manual = (rank_pos_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    assert abs(auc_primary - auc_manual) < 1e-6


def test_auroc_polarity_is_correct():
    # If higher score => more likely positive, AUROC should be > 0.5, not < 0.5.
    rng = np.random.default_rng(11)
    y = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([
        rng.normal(loc=0.0, scale=1.0, size=500),
        rng.normal(loc=2.0, scale=1.0, size=500),
    ])
    auc = _auroc(y, scores)
    assert auc > 0.8, f"AUROC polarity flipped? got {auc}"


def test_compute_verdict_metrics_returns_all_keys():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    # Flip 20% randomly
    flip = rng.random(400) < 0.2
    y_pred[flip] = 1 - y_pred[flip]
    y_score = rng.uniform(size=400)
    out = compute_verdict_metrics(y_true, y_pred, y_score, n_boot=200)
    assert set(out.keys()) == {"accuracy", "contradiction_recall", "supported_recall", "auroc"}
    for v in out.values():
        assert v.ci_low <= v.value <= v.ci_high + 1e-9  # CI contains point
