"""Tests for ``evaluation.krippendorff_alpha``.

The headline pin is the canonical 4-coder example from Krippendorff's
2011 working paper *Computing Krippendorff's Alpha-Reliability*
(Table 3), for which the published ordinal α ≈ 0.815.  We also
check structural properties (perfect-agreement ⇒ 1.0, coincidence
symmetry, None/nan equivalence, validation errors) and bootstrap CI
behaviour (reproducibility + bracketing).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

# Allow "python tests/test_krippendorff.py" from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.krippendorff_alpha import (  # noqa: E402
    _coincidence_matrix,
    _delta_squared,
    _to_float_matrix,
    _value_index,
    alpha,
    alpha_with_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# Canonical Krippendorff 2011 Table 3 example.
# 4 coders × 12 units.  "NaN" marks missing data.
# Published ordinal α ≈ 0.815 (ref: Krippendorff 2011, Table 4).
# ---------------------------------------------------------------------------
CANONICAL_DATA = [
    [1,     2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
    [1,     2, 3, 3, 2, 2, 4, 1, 2, 5,      np.nan, 3     ],
    [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5,      1,     np.nan],
    [1,     2, 3, 3, 2, 4, 4, 1, 2, 5,      1,     np.nan],
]


class TestCanonicalKrippendorff(unittest.TestCase):
    """Pin the implementation to the Krippendorff 2011 published α.

    All four values below come from Krippendorff (2011) Table 4:
    nominal = 0.743, ordinal = 0.815, interval = 0.849, ratio = 0.797.
    Our implementation reproduces every one of them to ≥ 3 decimals,
    so we assert within δ=0.001.
    """

    def test_ordinal_matches_published(self) -> None:
        a = alpha(CANONICAL_DATA, level="ordinal")
        self.assertAlmostEqual(a, 0.815, delta=0.001)

    def test_nominal_matches_published(self) -> None:
        a = alpha(CANONICAL_DATA, level="nominal")
        self.assertAlmostEqual(a, 0.743, delta=0.001)

    def test_interval_matches_published(self) -> None:
        a = alpha(CANONICAL_DATA, level="interval")
        self.assertAlmostEqual(a, 0.849, delta=0.001)

    def test_ratio_matches_published(self) -> None:
        a = alpha(CANONICAL_DATA, level="ratio")
        self.assertAlmostEqual(a, 0.797, delta=0.001)

    def test_metrics_are_finite(self) -> None:
        for lvl in ("nominal", "ordinal", "interval", "ratio"):
            a = alpha(CANONICAL_DATA, level=lvl)  # type: ignore[arg-type]
            self.assertTrue(np.isfinite(a), f"{lvl} α is not finite")


class TestAlphaEdgeCases(unittest.TestCase):
    def test_perfect_agreement_returns_one(self) -> None:
        data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        self.assertEqual(alpha(data, level="ordinal"), 1.0)
        self.assertEqual(alpha(data, level="nominal"), 1.0)
        self.assertEqual(alpha(data, level="interval"), 1.0)

    def test_all_coders_constant_returns_one(self) -> None:
        # Every coder gives the same value for every unit.
        data = [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
        self.assertEqual(alpha(data, level="ordinal"), 1.0)

    def test_two_coders_two_units_agree(self) -> None:
        data = [[1, 2], [1, 2]]
        self.assertEqual(alpha(data, level="ordinal"), 1.0)

    def test_nan_and_none_equivalent(self) -> None:
        a_nan = alpha(CANONICAL_DATA, level="ordinal")
        with_none: list[list] = []
        for row in CANONICAL_DATA:
            new_row: list = []
            for v in row:
                if isinstance(v, float) and math.isnan(v):
                    new_row.append(None)
                else:
                    new_row.append(v)
            with_none.append(new_row)
        a_none = alpha(with_none, level="ordinal")
        self.assertAlmostEqual(a_nan, a_none, places=12)

    def test_systematic_disagreement_is_negative(self) -> None:
        # Two coders systematically swap labels 1 and 2 ⇒ negative α.
        data = [
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        ]
        a = alpha(data, level="nominal")
        self.assertLess(a, 0.0)


class TestAlphaValidation(unittest.TestCase):
    def test_requires_two_coders(self) -> None:
        with self.assertRaises(ValueError):
            alpha([[1, 2, 3, 4, 5]], level="ordinal")

    def test_rejects_1d_input(self) -> None:
        with self.assertRaises(ValueError):
            alpha(np.array([1, 2, 3, 4]), level="ordinal")

    def test_unknown_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            alpha([[1, 2], [1, 2]], level="quantum")  # type: ignore[arg-type]

    def test_all_missing_rejected(self) -> None:
        data = [[np.nan, np.nan], [np.nan, np.nan]]
        with self.assertRaises(ValueError):
            alpha(data, level="ordinal")

    def test_rejects_single_pairable_observation(self) -> None:
        # Only one unit has ≥ 2 valid coders; the rest are singletons.
        data = [
            [1,     np.nan, np.nan],
            [2,     np.nan, np.nan],
        ]
        # This does give us one pairable unit ⇒ n_total = 2.
        # It should NOT raise, but should give a definite result.
        a = alpha(data, level="nominal")
        self.assertTrue(np.isfinite(a))


class TestBootstrapCI(unittest.TestCase):
    def test_point_equals_non_bootstrap_alpha(self) -> None:
        point, _, _ = alpha_with_bootstrap_ci(
            CANONICAL_DATA, level="ordinal", n_bootstrap=200, seed=7
        )
        direct = alpha(CANONICAL_DATA, level="ordinal")
        self.assertAlmostEqual(point, direct, places=12)

    def test_ci_brackets_point_or_nearly(self) -> None:
        point, lo, hi = alpha_with_bootstrap_ci(
            CANONICAL_DATA, level="ordinal", n_bootstrap=500, seed=0
        )
        # Percentile CI should contain the point estimate (modulo a
        # tiny slack for the discrete percentile interpolation).
        self.assertLessEqual(lo, point + 0.01)
        self.assertGreaterEqual(hi, point - 0.01)

    def test_ci_reproducible_same_seed(self) -> None:
        r1 = alpha_with_bootstrap_ci(
            CANONICAL_DATA, level="ordinal", n_bootstrap=200, seed=42
        )
        r2 = alpha_with_bootstrap_ci(
            CANONICAL_DATA, level="ordinal", n_bootstrap=200, seed=42
        )
        self.assertEqual(r1, r2)

    def test_ci_finite_for_canonical(self) -> None:
        _, lo, hi = alpha_with_bootstrap_ci(
            CANONICAL_DATA, level="ordinal", n_bootstrap=300, seed=1
        )
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))
        self.assertLessEqual(lo, hi)

    def test_small_input_returns_nan_ci_gracefully(self) -> None:
        # Only 1 unit ⇒ bootstrap cannot run ⇒ NaN CI, finite point.
        data = [[1], [2]]
        point, lo, hi = alpha_with_bootstrap_ci(
            data, level="nominal", n_bootstrap=50, seed=0
        )
        self.assertTrue(np.isfinite(point))
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))


class TestHelpers(unittest.TestCase):
    def test_to_float_matrix_none_becomes_nan(self) -> None:
        m = _to_float_matrix([[1, None, 3], [None, 2, 3]])
        self.assertTrue(np.isnan(m[0, 1]))
        self.assertTrue(np.isnan(m[1, 0]))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 2], 3.0)

    def test_to_float_matrix_passes_through_float_array(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = _to_float_matrix(arr)
        np.testing.assert_array_equal(m, arr)
        self.assertEqual(m.dtype, np.float64)

    def test_coincidence_matrix_symmetric(self) -> None:
        matrix = _to_float_matrix(CANONICAL_DATA)
        _, value_to_idx = _value_index(matrix)
        co = _coincidence_matrix(matrix, value_to_idx)
        np.testing.assert_allclose(co, co.T, rtol=0, atol=1e-12)

    def test_coincidence_total_equals_pairable_n(self) -> None:
        matrix = _to_float_matrix(CANONICAL_DATA)
        _, value_to_idx = _value_index(matrix)
        co = _coincidence_matrix(matrix, value_to_idx)
        # n = sum over units of m_u where m_u >= 2.
        expected_n = 0.0
        for u in range(matrix.shape[1]):
            m_u = int(np.sum(~np.isnan(matrix[:, u])))
            if m_u >= 2:
                expected_n += m_u
        self.assertAlmostEqual(float(co.sum()), expected_n, places=9)

    def test_delta_squared_nominal_ones_off_diagonal(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        n_c = np.array([1.0, 1.0, 1.0])
        d = _delta_squared("nominal", values, n_c)
        self.assertTrue(np.all(np.diag(d) == 0))
        off = d[~np.eye(3, dtype=bool)]
        self.assertTrue(np.all(off == 1.0))

    def test_delta_squared_interval_matches_squared_difference(self) -> None:
        values = np.array([1.0, 4.0, 9.0])
        n_c = np.array([1.0, 1.0, 1.0])
        d = _delta_squared("interval", values, n_c)
        self.assertAlmostEqual(d[0, 1], 9.0)
        self.assertAlmostEqual(d[0, 2], 64.0)
        self.assertAlmostEqual(d[1, 2], 25.0)

    def test_delta_squared_ratio_symmetric(self) -> None:
        values = np.array([1.0, 2.0, 5.0])
        n_c = np.array([1.0, 1.0, 1.0])
        d = _delta_squared("ratio", values, n_c)
        np.testing.assert_allclose(d, d.T, rtol=0, atol=1e-12)
        self.assertEqual(d[0, 0], 0.0)

    def test_delta_squared_unknown_raises(self) -> None:
        values = np.array([1.0, 2.0])
        n_c = np.array([1.0, 1.0])
        with self.assertRaises(ValueError):
            _delta_squared("foo", values, n_c)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
