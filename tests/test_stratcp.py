"""Tests for ``inference.stratcp`` — the StratCP baseline.

Task 6 of the v3 sprint.  StratCP (Zitnik-Lab medRxiv Feb 2026) is our
comparison baseline for cfBH; these tests cover:

1. Static math (``_quantile_level``).
2. Constructor + calibration input validation.
3. Per-stratum behaviour: small strata fall back to pooled quantile.
4. Prediction API: boolean mask, shape, unseen-stratum fallback,
   pre-calibration RuntimeError, shape errors.
5. Finite-sample coverage on synthetic Gaussian strata.  Under
   exchangeability, the marginal rejection rate over many trials must
   be ≤ α (within ±2 pp at the chosen sample size).
6. Diagnostic helpers: per_stratum_thresholds, stratum_sizes,
   pooled_threshold, empirical_coverage.

Run:
    python3 tests/test_stratcp.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.stratcp import StratCPConfig, StratCPPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Static helper: quantile-level formula
# ---------------------------------------------------------------------------


class TestQuantileLevel(unittest.TestCase):
    def test_basic(self):
        # n=100, α=0.05 → (101 × 0.95) / 100 = 0.9595
        self.assertAlmostEqual(
            StratCPPredictor._quantile_level(100, 0.05),
            0.9595,
            places=6,
        )

    def test_tiny_n_clips_to_one(self):
        # n=1, α=0.05 → (2 × 0.95) / 1 = 1.9 → clipped to 1.0.
        # At the boundary numpy with method="higher" returns the max.
        self.assertEqual(StratCPPredictor._quantile_level(1, 0.05), 1.0)

    def test_large_alpha_not_clipped(self):
        # n=50, α=0.5 → (51 × 0.5) / 50 = 0.51 (no clipping needed).
        self.assertAlmostEqual(
            StratCPPredictor._quantile_level(50, 0.5),
            0.51,
            places=6,
        )

    def test_zero_n_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor._quantile_level(0, 0.05)

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor._quantile_level(-3, 0.05)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestInitValidation(unittest.TestCase):
    def test_alpha_zero_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor(alpha=0.0)

    def test_alpha_one_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor(alpha=1.0)

    def test_alpha_negative_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor(alpha=-0.1)

    def test_alpha_above_one_raises(self):
        with self.assertRaises(ValueError):
            StratCPPredictor(alpha=1.5)

    def test_min_stratum_size_cast_to_int(self):
        p = StratCPPredictor(alpha=0.05, min_stratum_size=10)
        self.assertEqual(p.min_stratum_size, 10)
        self.assertIsInstance(p.min_stratum_size, int)


# ---------------------------------------------------------------------------
# Calibration validation
# ---------------------------------------------------------------------------


class TestCalibrateValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = StratCPPredictor(alpha=0.05, min_stratum_size=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.calibrate(
                np.array([1.0, 2.0, 3.0]),
                np.array(["A", "B"]),
            )

    def test_ndim_scores_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.calibrate(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array(["A", "B"]),
            )

    def test_ndim_strata_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.calibrate(
                np.array([1.0, 2.0]),
                np.array([["A"], ["B"]]),
            )

    def test_empty_scores_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.calibrate(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Calibration behaviour
# ---------------------------------------------------------------------------


class TestCalibrateBasics(unittest.TestCase):
    def test_small_stratum_falls_back_to_pooled(self):
        # 3 strata, one of which is smaller than min_stratum_size=10.
        rng = np.random.default_rng(0)
        cal = np.concatenate([
            rng.normal(0.0, 1.0, size=30),   # stratum A: n=30
            rng.normal(0.5, 1.0, size=30),   # stratum B: n=30
            rng.normal(1.0, 1.0, size=5),    # stratum C: n=5 (too small)
        ])
        strata = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 5)
        predictor = StratCPPredictor(alpha=0.05, min_stratum_size=10)
        predictor.calibrate(cal, strata)

        q_table = predictor.per_stratum_thresholds()
        self.assertIn("A", q_table)
        self.assertIn("B", q_table)
        self.assertIn("C", q_table)
        # C (too small) should equal the pooled quantile exactly.
        self.assertEqual(q_table["C"], predictor.pooled_threshold())

    def test_sizes_reported(self):
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        strata = np.array(["X", "X", "Y", "Y", "Y"])
        p = StratCPPredictor(alpha=0.05, min_stratum_size=2)
        p.calibrate(cal, strata)
        sizes = p.stratum_sizes()
        self.assertEqual(sizes, {"X": 2, "Y": 3})

    def test_fluent_return(self):
        p = StratCPPredictor(alpha=0.05, min_stratum_size=2)
        ret = p.calibrate(np.array([1.0, 2.0]), np.array(["A", "A"]))
        self.assertIs(ret, p)

    def test_non_string_strata_coerced(self):
        # Int strata should work — they get cast to str internally.
        p = StratCPPredictor(alpha=0.05, min_stratum_size=2)
        p.calibrate(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0, 0, 1, 1]),
        )
        self.assertEqual(set(p.stratum_sizes().keys()), {"0", "1"})


# ---------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------


class TestPredict(unittest.TestCase):
    def _make_fitted(self) -> StratCPPredictor:
        rng = np.random.default_rng(42)
        cal = np.concatenate([
            rng.normal(0.0, 1.0, size=100),
            rng.normal(0.0, 1.0, size=100),
        ])
        strata = np.array(["A"] * 100 + ["B"] * 100)
        p = StratCPPredictor(alpha=0.1, min_stratum_size=20)
        p.calibrate(cal, strata)
        return p

    def test_predict_before_calibrate_raises(self):
        p = StratCPPredictor(alpha=0.05)
        with self.assertRaises(RuntimeError):
            p.predict(np.array([0.0]), np.array(["A"]))

    def test_returns_bool_array_of_right_shape(self):
        p = self._make_fitted()
        out = p.predict(np.array([5.0, -5.0]), np.array(["A", "A"]))
        self.assertEqual(out.dtype, bool)
        self.assertEqual(out.shape, (2,))
        # 5.0 >> any calibration score ⇒ reject.
        self.assertTrue(bool(out[0]))
        # -5.0 << any calibration score ⇒ no reject.
        self.assertFalse(bool(out[1]))

    def test_shape_mismatch_raises(self):
        p = self._make_fitted()
        with self.assertRaises(ValueError):
            p.predict(np.array([0.0, 1.0]), np.array(["A"]))

    def test_non_1d_raises(self):
        p = self._make_fitted()
        with self.assertRaises(ValueError):
            p.predict(
                np.array([[0.0, 1.0]]),
                np.array([["A", "B"]]),
            )

    def test_unseen_stratum_falls_back_to_pooled(self):
        p = self._make_fitted()
        pooled = p.pooled_threshold()
        assert pooled is not None
        # A test point at exactly the pooled threshold should reject
        # (since predict uses >=), one just below should not.
        out = p.predict(
            np.array([pooled, pooled - 1e-6]),
            np.array(["UNSEEN_LABEL", "UNSEEN_LABEL"]),
        )
        self.assertTrue(bool(out[0]))
        self.assertFalse(bool(out[1]))


# ---------------------------------------------------------------------------
# Finite-sample coverage — core correctness test
# ---------------------------------------------------------------------------


class TestCoverageSynthetic(unittest.TestCase):
    """Finite-sample coverage check.

    Under exchangeability, the split-conformal marginal rejection rate
    on test data drawn from the same distribution as calibration should
    be bounded by ``α`` (Vovk-Shafer-Gammerman / Lei-Wasserman Thm 1).
    """

    def test_exchangeable_rejection_rate_near_alpha(self):
        alpha = 0.05
        n_cal = 300
        n_test = 300
        n_trials = 100
        rng = np.random.default_rng(12345)
        rejection_rates: list[float] = []
        for _ in range(n_trials):
            # 3 Gaussian strata with different means.  StratCP must
            # calibrate each one independently.
            cal_a = rng.normal(0.0, 1.0, size=n_cal // 3)
            cal_b = rng.normal(1.0, 1.0, size=n_cal // 3)
            cal_c = rng.normal(-1.0, 1.0, size=n_cal - 2 * (n_cal // 3))
            cal_scores = np.concatenate([cal_a, cal_b, cal_c])
            cal_strata = np.array(
                ["A"] * len(cal_a)
                + ["B"] * len(cal_b)
                + ["C"] * len(cal_c)
            )

            test_a = rng.normal(0.0, 1.0, size=n_test // 3)
            test_b = rng.normal(1.0, 1.0, size=n_test // 3)
            test_c = rng.normal(-1.0, 1.0, size=n_test - 2 * (n_test // 3))
            test_scores = np.concatenate([test_a, test_b, test_c])
            test_strata = np.array(
                ["A"] * len(test_a)
                + ["B"] * len(test_b)
                + ["C"] * len(test_c)
            )

            p = StratCPPredictor(alpha=alpha, min_stratum_size=20)
            p.calibrate(cal_scores, cal_strata)
            mask = p.predict(test_scores, test_strata)
            rejection_rates.append(float(mask.mean()))

        mean_rate = float(np.mean(rejection_rates))
        # Split-CP guarantees P(reject) ≤ α under exchangeability.
        # The "higher"-interpolation variant is typically
        # α - 1/(n_s+1) in expectation per stratum.  Within ±2 pp is a
        # safe finite-sample band for n_cal=300, n_test=300, 100 trials.
        self.assertLessEqual(
            mean_rate,
            alpha + 0.02,
            f"Mean rejection rate {mean_rate:.4f} exceeds "
            f"α+0.02={alpha+0.02}; calibration may be broken.",
        )
        # Not absurdly conservative either.
        self.assertGreaterEqual(
            mean_rate,
            alpha - 0.03,
            f"Mean rejection rate {mean_rate:.4f} is far below "
            f"α-0.03={alpha-0.03}; likely a bug.",
        )

    def test_shifted_positive_class_rejects_more(self):
        """If the test distribution is shifted upward (representing a
        real contradicted population with higher scores), StratCP
        should reject well above α — otherwise it's not a detector."""
        alpha = 0.05
        rng = np.random.default_rng(7)
        cal_scores = rng.normal(0.0, 1.0, size=600)
        cal_strata = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * 200)
        p = StratCPPredictor(alpha=alpha, min_stratum_size=20)
        p.calibrate(cal_scores, cal_strata)

        # Test positives shifted by 3σ — most should reject.
        test_scores = rng.normal(3.0, 1.0, size=300)
        test_strata = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        mask = p.predict(test_scores, test_strata)
        rate = float(mask.mean())
        self.assertGreater(
            rate, 0.5,
            f"Shifted-positive rejection rate only {rate:.3f}; "
            "StratCP should detect a 3σ distribution shift.",
        )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


class TestDiagnosticHelpers(unittest.TestCase):
    def test_empirical_coverage_keys(self):
        rng = np.random.default_rng(3)
        cal = rng.normal(0.0, 1.0, size=200)
        cal_strata = np.array(["A"] * 100 + ["B"] * 100)
        p = StratCPPredictor(alpha=0.1, min_stratum_size=20)
        p.calibrate(cal, cal_strata)

        test = rng.normal(1.0, 1.0, size=50)
        test_strata = np.array(["A"] * 25 + ["B"] * 25)
        test_labels = np.array([1] * 25 + [0] * 25)
        out = p.empirical_coverage(test, test_labels, test_strata)
        self.assertEqual(
            set(out.keys()),
            {"power", "fdr", "rejection_rate", "n_test"},
        )
        self.assertEqual(out["n_test"], 50)
        self.assertGreaterEqual(out["rejection_rate"], 0.0)
        self.assertLessEqual(out["rejection_rate"], 1.0)

    def test_empirical_coverage_no_positives_returns_nan_power(self):
        rng = np.random.default_rng(4)
        cal = rng.normal(0.0, 1.0, size=100)
        cal_strata = np.array(["A"] * 100)
        p = StratCPPredictor(alpha=0.1, min_stratum_size=10)
        p.calibrate(cal, cal_strata)

        test = rng.normal(0.0, 1.0, size=20)
        test_strata = np.array(["A"] * 20)
        test_labels = np.zeros(20, dtype=int)
        out = p.empirical_coverage(test, test_labels, test_strata)
        self.assertTrue(math.isnan(out["power"]))

    def test_pooled_threshold_returned(self):
        p = StratCPPredictor(alpha=0.05)
        # Before calibrate: None.
        self.assertIsNone(p.pooled_threshold())
        p.calibrate(np.array([0.0, 1.0, 2.0]), np.array(["A", "A", "A"]))
        self.assertIsInstance(p.pooled_threshold(), float)

    def test_diagnostic_returns_are_copies(self):
        p = StratCPPredictor(alpha=0.05, min_stratum_size=2)
        p.calibrate(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array(["A", "A", "B", "B"]),
        )
        thresholds = p.per_stratum_thresholds()
        sizes = p.stratum_sizes()
        # Mutating the returned dict must not change the internal state.
        thresholds["A"] = 99999.0
        sizes["A"] = 99999
        self.assertNotEqual(p.per_stratum_thresholds()["A"], 99999.0)
        self.assertNotEqual(p.stratum_sizes()["A"], 99999)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestConfigDataclass(unittest.TestCase):
    def test_defaults(self):
        cfg = StratCPConfig(alpha=0.05)
        self.assertEqual(cfg.min_stratum_size, 20)
        self.assertEqual(cfg.alpha, 0.05)

    def test_custom(self):
        cfg = StratCPConfig(alpha=0.1, min_stratum_size=50)
        self.assertEqual(cfg.alpha, 0.1)
        self.assertEqual(cfg.min_stratum_size, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
