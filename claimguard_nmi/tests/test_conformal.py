"""Unit tests for the conformal FDR modules."""
from __future__ import annotations

import numpy as np

from claimguard_nmi.conformal import InvertedCfBH, WeightedCfBH


def _simulate(n_cal_contra=500, n_test_faith=800, n_test_contra=200, seed=17):
    rng = np.random.default_rng(seed)
    cal_scores = rng.beta(2.0, 20.0, size=n_cal_contra)           # contradicted near 0
    test_faithful = rng.beta(20.0, 2.0, size=n_test_faith)        # faithful near 1
    test_contradicted = rng.beta(2.0, 20.0, size=n_test_contra)   # contradicted near 0
    test_scores = np.concatenate([test_faithful, test_contradicted])
    labels = np.concatenate([np.zeros(n_test_faith, dtype=int),
                             np.ones(n_test_contra, dtype=int)])
    order = rng.permutation(labels.size)
    return cal_scores, test_scores[order], labels[order]


def test_inverted_cfbh_controls_fdr_when_exchangeable():
    cal, test_scores, labels = _simulate()
    proc = InvertedCfBH().fit(cal)
    result = proc.predict(test_scores, alpha=0.05)
    result = InvertedCfBH.audit(result, labels)
    # With exchangeable contradicted scores, FDR should be <= alpha by a wide margin.
    assert result.fdr is not None
    assert result.fdr <= 0.10, f"FDR {result.fdr} too high"
    # Power should be substantially > 0.
    assert result.power is not None and result.power > 0.5


def test_inverted_cfbh_rejects_zero_at_tight_alpha_with_tiny_cal():
    rng = np.random.default_rng(3)
    cal = rng.beta(2.0, 20.0, size=10)   # n_cal=10; min p = 1/11 ~ 0.091
    test = rng.uniform(0.95, 1.0, size=100)
    proc = InvertedCfBH().fit(cal)
    result = proc.predict(test, alpha=0.01)
    # With n_cal=10, min p-value is 1/11 > 0.01 - BH must accept zero.
    assert result.n_green == 0


def test_weighted_cfbh_returns_valid_result():
    cal, test_scores, labels = _simulate(n_cal_contra=200, n_test_faith=300, n_test_contra=100)
    # 2-dim feature vector — use (score, 0) as dummy features
    cal_feat = np.column_stack([cal, np.zeros_like(cal)])
    test_feat = np.column_stack([test_scores, np.zeros_like(test_scores)])
    proc = WeightedCfBH(ess_floor=0.1).fit(cal, cal_feat, test_feat)
    result = proc.predict(test_scores, test_feat, alpha=0.05)
    assert result.p_values.shape == test_scores.shape
    assert 0 <= result.n_green <= test_scores.size


def test_bh_green_count_nondecreasing_in_alpha():
    cal, test_scores, _ = _simulate()
    proc = InvertedCfBH().fit(cal)
    g_05 = proc.predict(test_scores, alpha=0.05).n_green
    g_10 = proc.predict(test_scores, alpha=0.10).n_green
    g_20 = proc.predict(test_scores, alpha=0.20).n_green
    assert g_05 <= g_10 <= g_20
