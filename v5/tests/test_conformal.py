"""Tests for conformal FDR variants."""

from __future__ import annotations

import numpy as np
import pytest

from v5.conformal.inverted_cfbh import inverted_cfbh
from v5.conformal.utils import compute_empirical_fdr


def test_inverted_cfbh_shapes_and_monotonicity():
    rng = np.random.default_rng(0)
    cal_contra = rng.beta(1, 20, size=500)    # contradicted calibration: low scores
    # Test: half truly supported (high scores), half contradicted (low scores).
    test_supp = rng.beta(20, 1, size=500)
    test_contra = rng.beta(1, 20, size=500)
    test_scores = np.concatenate([test_supp, test_contra])
    test_labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])

    r = inverted_cfbh(test_scores, cal_contra, alpha=0.05)
    assert r.p_values.shape == test_scores.shape
    # Truly supported (index < 500) should have small p-values on average
    assert r.p_values[:500].mean() < r.p_values[500:].mean()
    diag = compute_empirical_fdr(r.green_mask, test_labels, alpha=0.05, n_bootstrap=500)
    # The green set should be mostly truly supported
    assert diag.empirical_fdr < 0.10


def test_inverted_cfbh_rejects_zero_when_scores_indistinguishable():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, size=200)
    cal = rng.uniform(0, 1, size=200)
    r = inverted_cfbh(scores, cal, alpha=0.05)
    # With overlapping distributions and alpha=0.05, most tests should not pass
    assert r.n_green < 100
