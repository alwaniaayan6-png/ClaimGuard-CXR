"""Tests for pure helpers in ``scripts.compute_user_vs_ensemble_alpha``.

Covers the fallback-ladder logic that decides whether the user +
silver graders hit the plan's α ≥ 0.80 target.  The Krippendorff
math itself is tested in ``tests/test_krippendorff.py`` — here we
just lock down the glue: row alignment, matrix construction, the
drop-UNCERTAIN and binary-coarsen rungs, and the ``passing_rung``
selection.

Fixture design
--------------
Tests build 4-coder synthetic data where the agreement pattern is
completely controlled:

    * "full agreement" fixtures → every rung passes with α = 1.0
    * "3-way match, user disagrees" → ordinal α < 1 but still > 0.8
    * "random labels" → α near 0 → all rungs fail

The bootstrap CIs are computed with small ``n_bootstrap`` (e.g. 50)
to keep test runtime under half a second.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from scripts.compile_silver_standard_results import (  # noqa: E402
    GRADER_LABEL_KEYS,
)
from scripts.compute_user_vs_ensemble_alpha import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CI,
    DEFAULT_MIN_ALPHA,
    DEFAULT_N_BOOTSTRAP,
    USER_COLUMN_KEY,
    align_rows,
    align_rows_with_drops,
    build_4coder_matrix,
    build_binary_matrix,
    compute_fallback_ladder,
    drop_uncertain_units,
    drop_uncertain_values,
    format_summary,
    load_self_annotation,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestModuleConstants(unittest.TestCase):
    def test_default_min_alpha_matches_plan(self) -> None:
        self.assertEqual(DEFAULT_MIN_ALPHA, 0.80)

    def test_default_bootstrap_count(self) -> None:
        self.assertEqual(DEFAULT_N_BOOTSTRAP, 1000)

    def test_default_seed_is_42(self) -> None:
        self.assertEqual(DEFAULT_BOOTSTRAP_SEED, 42)

    def test_default_ci(self) -> None:
        self.assertEqual(DEFAULT_CI, 0.95)

    def test_user_column_key(self) -> None:
        self.assertEqual(USER_COLUMN_KEY, "user_label")


# ---------------------------------------------------------------------------
# load_self_annotation
# ---------------------------------------------------------------------------


class TestLoadSelfAnnotation(unittest.TestCase):
    def test_loads_nonempty_list(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                [{"claim_id": "c1", "user_label": "SUPPORTED"}],
                f,
            )
            path = f.name
        try:
            rows = load_self_annotation(path)
            self.assertEqual(len(rows), 1)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([], f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_self_annotation(path)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_self_annotation("/nonexistent/path.json")


# ---------------------------------------------------------------------------
# align_rows
# ---------------------------------------------------------------------------


def _silver(claim_id: str, grader_label: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "grader_chexbert_label": grader_label,
        "grader_claude_label": grader_label,
        "grader_medgemma_label": grader_label,
    }


def _user(claim_id: str, label: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "user_label": label,
    }


class TestAlignRows(unittest.TestCase):
    def test_basic_join(self) -> None:
        silver = [_silver("c1", "SUPPORTED"), _silver("c2", "CONTRADICTED")]
        user = [_user("c1", "SUPPORTED"), _user("c2", "CONTRADICTED")]
        merged = align_rows(silver, user)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["user_label"], "SUPPORTED")
        self.assertEqual(merged[0]["grader_chexbert_label"], "SUPPORTED")

    def test_user_row_missing_from_silver_dropped(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [_user("c1", "SUPPORTED"), _user("orphan", "SUPPORTED")]
        merged = align_rows(silver, user)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["claim_id"], "c1")

    def test_invalid_user_label_dropped(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [_user("c1", "BOGUS")]
        self.assertEqual(align_rows(silver, user), [])

    def test_user_row_missing_claim_id_dropped(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [{"user_label": "SUPPORTED"}]  # no claim_id
        self.assertEqual(align_rows(silver, user), [])

    def test_silver_row_no_valid_graders_dropped(self) -> None:
        silver = [{
            "claim_id": "c1",
            "grader_chexbert_label": "",
            "grader_claude_label": "bogus",
            "grader_medgemma_label": None,
        }]
        user = [_user("c1", "SUPPORTED")]
        self.assertEqual(align_rows(silver, user), [])

    def test_preserves_all_grader_columns(self) -> None:
        silver = [{
            "claim_id": "c1",
            "grader_chexbert_label": "SUPPORTED",
            "grader_claude_label": "CONTRADICTED",
            "grader_medgemma_label": "UNCERTAIN",
        }]
        user = [_user("c1", "SUPPORTED")]
        merged = align_rows(silver, user)
        self.assertEqual(merged[0]["grader_chexbert_label"], "SUPPORTED")
        self.assertEqual(merged[0]["grader_claude_label"], "CONTRADICTED")
        self.assertEqual(merged[0]["grader_medgemma_label"], "UNCERTAIN")


# ---------------------------------------------------------------------------
# align_rows_with_drops — reviewer-requested diagnostic on silent drops
# ---------------------------------------------------------------------------


class TestAlignRowsWithDrops(unittest.TestCase):
    def test_all_valid_zero_drops(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [_user("c1", "SUPPORTED")]
        merged, drops = align_rows_with_drops(silver, user)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            drops,
            {
                "missing_claim_id": 0,
                "no_silver_match": 0,
                "invalid_user_label": 0,
                "silver_no_valid_graders": 0,
            },
        )

    def test_missing_claim_id_counted(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [
            _user("c1", "SUPPORTED"),
            {"user_label": "SUPPORTED"},  # no claim_id
        ]
        _, drops = align_rows_with_drops(silver, user)
        self.assertEqual(drops["missing_claim_id"], 1)

    def test_no_silver_match_counted(self) -> None:
        silver = [_silver("c1", "SUPPORTED")]
        user = [
            _user("c1", "SUPPORTED"),
            _user("orphan", "SUPPORTED"),
        ]
        _, drops = align_rows_with_drops(silver, user)
        self.assertEqual(drops["no_silver_match"], 1)

    def test_invalid_user_label_counted(self) -> None:
        silver = [_silver("c1", "SUPPORTED"), _silver("c2", "SUPPORTED")]
        user = [
            _user("c1", "BOGUS"),
            _user("c2", "SUPPORTED"),
        ]
        _, drops = align_rows_with_drops(silver, user)
        self.assertEqual(drops["invalid_user_label"], 1)

    def test_silver_no_valid_graders_counted(self) -> None:
        silver = [
            _silver("c1", "SUPPORTED"),
            {
                "claim_id": "c2",
                "grader_chexbert_label": "",
                "grader_claude_label": "",
                "grader_medgemma_label": "",
            },
        ]
        user = [
            _user("c1", "SUPPORTED"),
            _user("c2", "SUPPORTED"),
        ]
        _, drops = align_rows_with_drops(silver, user)
        self.assertEqual(drops["silver_no_valid_graders"], 1)

    def test_align_rows_wrapper_matches_merged_result(self) -> None:
        """The backward-compat wrapper returns only the merged list."""
        silver = [_silver("c1", "SUPPORTED")]
        user = [_user("c1", "SUPPORTED")]
        merged_wrapped = align_rows(silver, user)
        merged_full, _ = align_rows_with_drops(silver, user)
        self.assertEqual(merged_wrapped, merged_full)

    def test_drops_keys_are_stable(self) -> None:
        """Lock the drop_counts schema so downstream report readers
        can rely on the four keys being present and spelled
        consistently."""
        _, drops = align_rows_with_drops([], [])
        self.assertEqual(
            set(drops.keys()),
            {
                "missing_claim_id",
                "no_silver_match",
                "invalid_user_label",
                "silver_no_valid_graders",
            },
        )


# ---------------------------------------------------------------------------
# drop_uncertain_values — canonical Krippendorff value-level drop
# ---------------------------------------------------------------------------


class TestDropUncertainValues(unittest.TestCase):
    def _row(self, *labels: str) -> dict[str, Any]:
        return {
            "grader_chexbert_label": labels[0],
            "grader_claude_label": labels[1],
            "grader_medgemma_label": labels[2],
            "user_label": labels[3],
        }

    def test_no_uncertain_is_noop(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        result = drop_uncertain_values(merged)
        self.assertEqual(result, merged)

    def test_single_uncertain_nanifies_that_cell_only(self) -> None:
        merged = [
            self._row("UNCERTAIN", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        result = drop_uncertain_values(merged)
        # Unit is preserved.
        self.assertEqual(len(result), 1)
        # The UNCERTAIN cell is replaced with empty string.
        self.assertEqual(result[0]["grader_chexbert_label"], "")
        # Other cells are untouched.
        self.assertEqual(result[0]["grader_claude_label"], "SUPPORTED")
        self.assertEqual(result[0]["grader_medgemma_label"], "SUPPORTED")
        self.assertEqual(result[0]["user_label"], "SUPPORTED")

    def test_does_not_mutate_input(self) -> None:
        merged = [
            self._row("UNCERTAIN", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        _ = drop_uncertain_values(merged)
        # Original is unchanged.
        self.assertEqual(merged[0]["grader_chexbert_label"], "UNCERTAIN")

    def test_unit_with_all_uncertain_becomes_all_empty(self) -> None:
        merged = [
            self._row("UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"),
        ]
        result = drop_uncertain_values(merged)
        self.assertEqual(len(result), 1)
        for k in (
            "grader_chexbert_label",
            "grader_claude_label",
            "grader_medgemma_label",
            "user_label",
        ):
            self.assertEqual(result[0][k], "")

    def test_preserves_non_coder_fields(self) -> None:
        row = self._row(
            "UNCERTAIN", "SUPPORTED", "SUPPORTED", "SUPPORTED"
        )
        row["claim_id"] = "c1"
        row["extra_field"] = "preserve me"
        result = drop_uncertain_values([row])
        self.assertEqual(result[0]["claim_id"], "c1")
        self.assertEqual(result[0]["extra_field"], "preserve me")

    def test_softer_than_unit_level_drop(self) -> None:
        """Unit-level drops units with ANY uncertain; value-level
        keeps them (with that cell nan-ified)."""
        merged = [
            self._row(
                "UNCERTAIN", "SUPPORTED", "SUPPORTED", "SUPPORTED"
            ),
            self._row(
                "SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"
            ),
        ]
        unit_dropped = drop_uncertain_units(merged)
        value_dropped = drop_uncertain_values(merged)
        self.assertEqual(len(unit_dropped), 1)
        self.assertEqual(len(value_dropped), 2)


# ---------------------------------------------------------------------------
# build_4coder_matrix
# ---------------------------------------------------------------------------


class TestBuild4CoderMatrix(unittest.TestCase):
    def _row(self, *labels: str) -> dict[str, Any]:
        # Order: chexbert, claude, medgemma, user.
        return {
            "grader_chexbert_label": labels[0],
            "grader_claude_label": labels[1],
            "grader_medgemma_label": labels[2],
            "user_label": labels[3],
        }

    def test_shape_is_4_by_n(self) -> None:
        merged = [
            self._row(
                "SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"
            ),
            self._row(
                "CONTRADICTED", "CONTRADICTED",
                "CONTRADICTED", "CONTRADICTED",
            ),
        ]
        m = build_4coder_matrix(merged)
        self.assertEqual(m.shape, (4, 2))

    def test_full_agreement_gives_uniform_column(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        m = build_4coder_matrix(merged)
        np.testing.assert_array_equal(m[:, 0], np.zeros(4))

    def test_invalid_labels_become_nan(self) -> None:
        merged = [
            self._row("BOGUS", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        m = build_4coder_matrix(merged)
        self.assertTrue(math.isnan(m[0, 0]))
        self.assertEqual(m[1, 0], 0.0)

    def test_empty_merged_gives_empty_matrix(self) -> None:
        m = build_4coder_matrix([])
        self.assertEqual(m.shape, (4, 0))

    def test_ordinal_encoding_matches_label_index(self) -> None:
        merged = [
            self._row(
                "SUPPORTED", "CONTRADICTED",
                "NOVEL_PLAUSIBLE", "NOVEL_HALLUCINATED",
            ),
        ]
        m = build_4coder_matrix(merged)
        # SUPPORTED=0, CONTRADICTED=1, NOVEL_PLAUSIBLE=2, NOVEL_HALLUCINATED=3.
        np.testing.assert_array_equal(m[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# drop_uncertain_units
# ---------------------------------------------------------------------------


class TestDropUncertainUnits(unittest.TestCase):
    def _row(self, *labels: str) -> dict[str, Any]:
        return {
            "grader_chexbert_label": labels[0],
            "grader_claude_label": labels[1],
            "grader_medgemma_label": labels[2],
            "user_label": labels[3],
        }

    def test_no_uncertain_keeps_all(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
            self._row(
                "CONTRADICTED", "CONTRADICTED",
                "CONTRADICTED", "CONTRADICTED",
            ),
        ]
        self.assertEqual(len(drop_uncertain_units(merged)), 2)

    def test_one_uncertain_drops_that_unit(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
            self._row("UNCERTAIN", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        dropped = drop_uncertain_units(merged)
        self.assertEqual(len(dropped), 1)
        self.assertNotIn("UNCERTAIN", dropped[0].values())

    def test_user_uncertain_also_drops_unit(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "UNCERTAIN"),
        ]
        self.assertEqual(drop_uncertain_units(merged), [])

    def test_all_uncertain_gives_empty(self) -> None:
        merged = [
            self._row("UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"),
            self._row("UNCERTAIN", "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"),
        ]
        self.assertEqual(drop_uncertain_units(merged), [])


# ---------------------------------------------------------------------------
# build_binary_matrix
# ---------------------------------------------------------------------------


class TestBuildBinaryMatrix(unittest.TestCase):
    def _row(self, *labels: str) -> dict[str, Any]:
        return {
            "grader_chexbert_label": labels[0],
            "grader_claude_label": labels[1],
            "grader_medgemma_label": labels[2],
            "user_label": labels[3],
        }

    def test_supported_maps_to_zero(self) -> None:
        merged = [
            self._row("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ]
        m = build_binary_matrix(merged)
        np.testing.assert_array_equal(m[:, 0], np.zeros(4))

    def test_all_others_map_to_one(self) -> None:
        merged = [
            self._row(
                "CONTRADICTED", "NOVEL_PLAUSIBLE",
                "NOVEL_HALLUCINATED", "UNCERTAIN",
            ),
        ]
        m = build_binary_matrix(merged)
        np.testing.assert_array_equal(m[:, 0], np.ones(4))

    def test_mixed_row(self) -> None:
        merged = [
            self._row(
                "SUPPORTED", "CONTRADICTED",
                "SUPPORTED", "UNCERTAIN",
            ),
        ]
        m = build_binary_matrix(merged)
        np.testing.assert_array_equal(m[:, 0], np.array([0, 1, 0, 1]))

    def test_invalid_becomes_nan(self) -> None:
        merged = [
            self._row(
                "BOGUS", "SUPPORTED", "SUPPORTED", "SUPPORTED"
            ),
        ]
        m = build_binary_matrix(merged)
        self.assertTrue(math.isnan(m[0, 0]))
        self.assertEqual(m[1, 0], 0.0)


# ---------------------------------------------------------------------------
# compute_fallback_ladder
# ---------------------------------------------------------------------------


def _full_agreement_merged(
    n_units: int = 20,
    label: str = "SUPPORTED",
) -> list[dict[str, Any]]:
    """All 4 coders give the same label on every unit → α = 1.0 on every rung."""
    return [
        {
            "claim_id": f"c{i:03d}",
            "grader_chexbert_label": label,
            "grader_claude_label": label,
            "grader_medgemma_label": label,
            "user_label": label,
        }
        for i in range(n_units)
    ]


def _random_labels_merged(
    n_units: int = 20,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Independent uniform labels → α near 0 → all rungs fail."""
    import random as _r
    from scripts.compile_silver_standard_results import VALID_LABELS
    rng = _r.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n_units):
        rows.append({
            "claim_id": f"c{i:03d}",
            "grader_chexbert_label": rng.choice(VALID_LABELS),
            "grader_claude_label": rng.choice(VALID_LABELS),
            "grader_medgemma_label": rng.choice(VALID_LABELS),
            "user_label": rng.choice(VALID_LABELS),
        })
    return rows


class TestComputeFallbackLadder(unittest.TestCase):
    def test_full_agreement_first_rung_passes(self) -> None:
        merged = _full_agreement_merged(n_units=20, label="SUPPORTED")
        report = compute_fallback_ladder(
            merged, n_bootstrap=50, min_alpha=0.8,
        )
        self.assertEqual(report["passing_rung"], "full_ordinal")
        self.assertTrue(report["rungs"]["full_ordinal"]["passes"])
        self.assertAlmostEqual(
            report["rungs"]["full_ordinal"]["alpha"], 1.0, places=6,
        )

    def test_full_agreement_mixed_labels_still_passes(self) -> None:
        # Agreement on a mix of labels (not all SUPPORTED) → α = 1.0.
        merged = (
            _full_agreement_merged(10, "SUPPORTED")
            + _full_agreement_merged(10, "CONTRADICTED")
        )
        report = compute_fallback_ladder(
            merged, n_bootstrap=50, min_alpha=0.8,
        )
        self.assertAlmostEqual(
            report["rungs"]["full_ordinal"]["alpha"], 1.0, places=6,
        )
        self.assertEqual(report["passing_rung"], "full_ordinal")

    def test_random_labels_all_rungs_fail(self) -> None:
        merged = _random_labels_merged(n_units=50, seed=7)
        report = compute_fallback_ladder(
            merged, n_bootstrap=50, min_alpha=0.8,
        )
        self.assertIsNone(report["passing_rung"])
        for rung_name in ("full_ordinal", "drop_uncertain", "binary_coarsen"):
            self.assertFalse(
                report["rungs"][rung_name]["passes"],
                f"rung {rung_name} unexpectedly passed on random data",
            )

    def test_ladder_has_all_three_rungs(self) -> None:
        merged = _full_agreement_merged(n_units=10)
        report = compute_fallback_ladder(merged, n_bootstrap=50)
        self.assertEqual(
            set(report["rungs"].keys()),
            {"full_ordinal", "drop_uncertain", "binary_coarsen"},
        )

    def test_min_alpha_threshold_respected(self) -> None:
        # High agreement but nowhere near 1.0 — 3 graders say SUPPORTED,
        # user says NOVEL_PLAUSIBLE on every unit.  ordinal α drops.
        merged = [
            {
                "claim_id": f"c{i:03d}",
                "grader_chexbert_label": "SUPPORTED",
                "grader_claude_label": "SUPPORTED",
                "grader_medgemma_label": "SUPPORTED",
                "user_label": "NOVEL_PLAUSIBLE",
            }
            for i in range(20)
        ]
        # With a very high threshold (0.99), this likely fails full.
        report_high = compute_fallback_ladder(
            merged, min_alpha=0.99, n_bootstrap=50,
        )
        # At least one rung should fail because user disagrees.
        self.assertFalse(
            all(
                report_high["rungs"][k]["passes"]
                for k in ("full_ordinal", "drop_uncertain", "binary_coarsen")
            )
        )

    def test_min_alpha_target_in_report(self) -> None:
        merged = _full_agreement_merged(n_units=10)
        report = compute_fallback_ladder(
            merged, min_alpha=0.75, n_bootstrap=50,
        )
        self.assertEqual(report["min_alpha_target"], 0.75)

    def test_degenerate_single_unit_produces_nan_reports(self) -> None:
        merged = _full_agreement_merged(n_units=1)
        report = compute_fallback_ladder(merged, n_bootstrap=50)
        # Single unit → Krippendorff α undefined for bootstrap path.
        # Helper returns NaN + passes=False.
        full = report["rungs"]["full_ordinal"]
        self.assertFalse(full["passes"])
        self.assertTrue(math.isnan(full["alpha"]))


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary(unittest.TestCase):
    def test_contains_all_rung_names(self) -> None:
        merged = _full_agreement_merged(n_units=10)
        report = compute_fallback_ladder(merged, n_bootstrap=50)
        report["n_user_annotated"] = 10
        report["n_matched_with_silver"] = 10
        summary = format_summary(report)
        self.assertIn("full_ordinal", summary)
        self.assertIn("drop_uncertain", summary)
        self.assertIn("binary_coarsen", summary)

    def test_passing_rung_shown_when_present(self) -> None:
        merged = _full_agreement_merged(n_units=10)
        report = compute_fallback_ladder(merged, n_bootstrap=50)
        report["n_user_annotated"] = 10
        report["n_matched_with_silver"] = 10
        summary = format_summary(report)
        self.assertIn("passing rung", summary)
        self.assertIn("full_ordinal", summary)

    def test_all_rungs_failed_message(self) -> None:
        merged = _random_labels_merged(n_units=50, seed=7)
        report = compute_fallback_ladder(merged, n_bootstrap=50)
        report["n_user_annotated"] = 50
        report["n_matched_with_silver"] = 50
        summary = format_summary(report)
        self.assertIn("all rungs failed", summary)

    def test_header_mentions_task_8(self) -> None:
        report = {
            "rungs": {
                "full_ordinal": {
                    "alpha": 0.9, "ci_low": 0.85, "ci_high": 0.95,
                    "n_units": 100, "passes": True,
                },
                "drop_uncertain": {
                    "alpha": 0.9, "ci_low": 0.85, "ci_high": 0.95,
                    "n_units": 80, "passes": True,
                },
                "binary_coarsen": {
                    "alpha": 0.95, "ci_low": 0.90, "ci_high": 1.0,
                    "n_units": 100, "passes": True,
                },
            },
            "passing_rung": "full_ordinal",
            "min_alpha_target": 0.80,
            "n_user_annotated": 100,
            "n_matched_with_silver": 100,
        }
        summary = format_summary(report)
        self.assertIn("Task 8", summary)
        self.assertIn("0.80", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
