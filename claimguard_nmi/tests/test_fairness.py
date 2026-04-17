"""Unit tests for the fairness / subgroup / decision-curve modules."""
from __future__ import annotations

import numpy as np

from claimguard_nmi.fairness import (
    age_quartile,
    compute_subgroup_metrics,
    net_benefit,
    operating_point_summary,
    parity_gap,
    stratify_by,
)


def test_age_quartile_buckets():
    assert age_quartile(20) == "q1_lt35"
    assert age_quartile(50) == "q2_35_54"
    assert age_quartile(65) == "q3_55_69"
    assert age_quartile(80) == "q4_ge70"
    assert age_quartile(None) == "unknown"
    assert age_quartile("garbage") == "unknown"


def test_stratify_by_uses_bin_fn():
    metadata = [{"age": v} for v in (30, 45, 60, 80, None, 28)]
    buckets = stratify_by(metadata, "age", bin_fn=age_quartile)
    assert buckets["q1_lt35"].tolist() == [0, 5]
    assert buckets["q4_ge70"].tolist() == [3]
    assert buckets["unknown"].tolist() == [4]


def test_subgroup_metrics_skips_tiny_buckets():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_score = rng.uniform(size=n)
    # All males -> single large bucket. Two females -> bucket of 2, should be skipped.
    metadata = [{"sex": ("F" if i < 2 else "M")} for i in range(n)]
    results = compute_subgroup_metrics(
        y_true, y_pred, y_score, metadata, key="sex", n_boot=50,
    )
    assert len(results) == 1
    assert results[0].subgroup_value == "M"


def test_parity_gap_returns_zero_for_identical_subgroups():
    # When all subgroups have identical y_true/y_pred, accuracy is 1.0 everywhere.
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_score = rng.uniform(size=n)
    metadata = [{"sex": ("F" if i % 2 == 0 else "M")} for i in range(n)]
    results = compute_subgroup_metrics(
        y_true, y_pred, y_score, metadata, key="sex", n_boot=100,
    )
    gap = parity_gap(results, metric="accuracy")
    assert gap == 0.0


def test_net_benefit_treat_all_beats_none_at_low_threshold():
    rng = np.random.default_rng(3)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_prob = y_true.astype(float) * 0.5 + rng.uniform(0, 0.5, size=n)
    curve = net_benefit(y_true, y_prob, thresholds=[0.05])
    # At alpha = 0.05, treat-all has positive net benefit when prevalence is ~0.5.
    assert curve.treat_all[0] > 0


def test_operating_point_summary_returns_declared_thresholds():
    rng = np.random.default_rng(5)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.uniform(size=n)
    curve = net_benefit(y_true, y_prob, thresholds=[0.05, 0.10, 0.20])
    out = operating_point_summary(curve, clinical_thresholds=[0.05, 0.20])
    assert set(out.keys()) == {"t=0.05", "t=0.20"}
    for entry in out.values():
        assert "model_nb" in entry
        assert "advantage_over_all" in entry
