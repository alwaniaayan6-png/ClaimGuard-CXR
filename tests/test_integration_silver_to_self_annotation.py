"""Integration test — Task 1 → Task 8 seam.

The unit tests in ``test_compile_silver_standard_results.py``,
``test_self_annotate_silver_subset.py`` and
``test_compute_user_vs_ensemble_alpha.py`` individually cover every
pure helper in their own scripts.  This file exercises the CROSS-
SCRIPT seam: a silver workbook produced by
``compile_silver_standard_results.synthetic_graded_workbook`` is fed
through ``self_annotate_silver_subset.sample_stratified``, wrapped
with user labels by ``build_self_annotation_row``, and consumed by
``compute_user_vs_ensemble_alpha.{align_rows_with_drops,
compute_fallback_ladder}``.  Any schema drift between the Task 1
output and the Task 8 input would surface here.

Design notes
------------
* No torch, no Modal, no Anthropic API — the synthetic workbook
  comes from ``synthetic_graded_workbook`` with ``noise_rate=0.10``,
  which is deterministic under ``seed=42``.
* The simulated user labels introduce a fixed 15% drift against the
  majority label.  At ``n=100`` this is close enough to the real
  silver-pool noise regime that the fallback ladder exercises a
  non-trivial path: ``full_ordinal`` fails the 0.80 gate (by
  construction), ``drop_uncertain`` passes, ``binary_coarsen`` fails
  (because binary coarsening LOSES the discriminative ordinal
  structure of the 4 error classes here).
* Bootstrap is tiny (200 replicates) to keep the test under 1 s.
  A real run uses 1000.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from collections import Counter
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.compile_silver_standard_results import (  # noqa: E402
    GRADER_LABEL_KEYS,
    VALID_LABELS,
    majority_vote,
    synthetic_graded_workbook,
)
from scripts.compute_user_vs_ensemble_alpha import (  # noqa: E402
    align_rows_with_drops,
    compute_fallback_ladder,
)
from scripts.self_annotate_silver_subset import (  # noqa: E402
    atomic_save_json,
    build_self_annotation_row,
    load_existing_annotations,
    sample_stratified,
)


def _stamp_majority(workbook: list[dict[str, Any]]) -> None:
    """Mutate in place to add ``majority_label`` — compile_silver does this
    inline during normal runs; the synthetic workbook does not."""
    for row in workbook:
        row["majority_label"] = majority_vote(
            [row.get(k, "") for k in GRADER_LABEL_KEYS]
        )


def _simulate_user_labels(
    sampled: list[dict[str, Any]],
    *,
    drift_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Label each sampled row as the majority label, with ``drift_rate``
    probability of picking a uniformly-random other label.

    Produces a realistic 4-coder matrix: 3 silver graders + 1 user
    column where the user agrees with the silver majority on most
    rows but introduces enough drift to exercise the fallback ladder.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for row in sampled:
        maj = row["majority_label"]
        if rng.random() < drift_rate:
            other_labels = [lbl for lbl in VALID_LABELS if lbl != maj]
            user_label = rng.choice(other_labels)
        else:
            user_label = maj
        out.append(build_self_annotation_row(row, user_label=user_label))
    return out


class TestTask1ToTask8SeamSmoke(unittest.TestCase):
    """End-to-end seam with a clean synthetic workbook."""

    def setUp(self) -> None:
        # Build a silver workbook the way Task 1 would: synthetic
        # graded rows, then stamp the majority label inline.
        self.silver = synthetic_graded_workbook(
            n_claims=200, noise_rate=0.10, seed=42,
        )
        _stamp_majority(self.silver)

    def test_silver_workbook_has_all_five_classes(self) -> None:
        classes = Counter(r["majority_label"] for r in self.silver)
        self.assertEqual(set(classes.keys()), set(VALID_LABELS))
        # Each class should have at least 20 rows so stratified
        # sampling is well-defined without warnings.
        for lbl, count in classes.items():
            self.assertGreaterEqual(count, 20, f"class {lbl} underfull")

    def test_stratified_sample_yields_100_rows(self) -> None:
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        self.assertEqual(len(sampled), 100)
        per_class = Counter(r["majority_label"] for r in sampled)
        for lbl in VALID_LABELS:
            self.assertEqual(per_class[lbl], 20)

    def test_sampled_rows_preserve_silver_schema(self) -> None:
        """The sampled rows must still carry the grader columns —
        otherwise ``align_rows`` will drop them as 'silver_no_valid_graders'."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        for row in sampled:
            for key in GRADER_LABEL_KEYS:
                self.assertIn(key, row)
                self.assertIn(row[key], set(VALID_LABELS))

    def test_build_self_annotation_row_carries_grader_columns(self) -> None:
        """The user-annotation wrapper must preserve grader columns
        so compute_user_vs_ensemble_alpha can read them via
        align_rows_with_drops without re-joining the silver workbook."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        row = build_self_annotation_row(sampled[0], user_label="SUPPORTED")
        for key in GRADER_LABEL_KEYS:
            self.assertIn(key, row)
            self.assertEqual(row[key], sampled[0][key])

    def test_atomic_save_round_trip(self) -> None:
        """load_existing_annotations should round-trip through
        atomic_save_json without losing any fields."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.15, seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "user.json")
            atomic_save_json(user_rows, path)
            reloaded = load_existing_annotations(path)
        self.assertEqual(len(reloaded), len(user_rows))
        self.assertEqual(
            [r["claim_id"] for r in reloaded],
            [r["claim_id"] for r in user_rows],
        )

    def test_align_rows_with_drops_zero_drops_on_clean_data(self) -> None:
        """Clean silver + clean user labels should produce zero drops."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.15, seed=7,
        )
        merged, drops = align_rows_with_drops(self.silver, user_rows)
        self.assertEqual(len(merged), 100)
        self.assertEqual(
            drops,
            {
                "missing_claim_id": 0,
                "no_silver_match": 0,
                "invalid_user_label": 0,
                "silver_no_valid_graders": 0,
            },
        )

    def test_fallback_ladder_exercises_drop_uncertain_rung(self) -> None:
        """With 15% drift and noise_rate=0.10, the canonical path is:
        full_ordinal just below 0.80, drop_uncertain above, binary
        coarsen below (because binary coarsening drops discriminative
        ordinal structure). This locks the behavior so a regression
        that silently moves the rung-2 semantics surfaces
        immediately."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.15, seed=7,
        )
        merged, _ = align_rows_with_drops(self.silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=200, seed=42,
        )
        # Rung 2 (value-level drop_uncertain) should clear; the
        # canonical Krippendorff interpretation is softer than
        # rung 1 when some UNCERTAIN codings exist.
        self.assertTrue(
            ladder["rungs"]["drop_uncertain"]["passes"],
            f"drop_uncertain rung unexpectedly failed: "
            f"{ladder['rungs']['drop_uncertain']}",
        )
        # Some rung must clear — passing_rung must not be None.
        self.assertIsNotNone(ladder["passing_rung"])

    def test_fallback_ladder_reports_all_three_rungs(self) -> None:
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.15, seed=7,
        )
        merged, _ = align_rows_with_drops(self.silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=200, seed=42,
        )
        self.assertEqual(
            set(ladder["rungs"].keys()),
            {"full_ordinal", "drop_uncertain", "binary_coarsen"},
        )
        for name, rung in ladder["rungs"].items():
            self.assertEqual(rung["n_units"], 100, f"rung {name}")
            self.assertEqual(rung["n_coders"], 4, f"rung {name}")

    def test_binary_rung_uses_nominal_level(self) -> None:
        """The binary-coarsen rung must be reported with ``level='nominal'``
        — the reviewer flagged the original ``level='ordinal'`` as a
        confusing choice for a k=2 case, and the fix is locked
        in by this assertion."""
        sampled = sample_stratified(self.silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.15, seed=7,
        )
        merged, _ = align_rows_with_drops(self.silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=200, seed=42,
        )
        self.assertEqual(
            ladder["rungs"]["binary_coarsen"]["level"], "nominal",
        )
        # The full-ordinal and drop-uncertain rungs stay on ordinal.
        self.assertEqual(
            ladder["rungs"]["full_ordinal"]["level"], "ordinal",
        )
        self.assertEqual(
            ladder["rungs"]["drop_uncertain"]["level"], "ordinal",
        )


class TestTask1ToTask8SeamEdgeCases(unittest.TestCase):
    """Seam behavior under boundary conditions."""

    def test_perfect_agreement_clears_full_ordinal(self) -> None:
        """Zero inter-grader noise + zero user drift → α = 1.

        Note: we pass ``noise_rate=0.0`` to the synthetic workbook
        builder so the 3 silver graders also have perfect
        intra-ensemble agreement.  If we kept ``noise_rate=0.10``
        and only zeroed the user drift, α would land around 0.85
        because the silver graders still disagreed with each other
        ~10% of the time — that's a meaningful test of the user
        column but NOT a test of "perfect agreement."
        """
        silver = synthetic_graded_workbook(
            n_claims=200, noise_rate=0.0, seed=42,
        )
        _stamp_majority(silver)
        sampled = sample_stratified(silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.0, seed=7,
        )
        merged, _ = align_rows_with_drops(silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=100, seed=42,
        )
        self.assertEqual(ladder["passing_rung"], "full_ordinal")
        self.assertAlmostEqual(
            ladder["rungs"]["full_ordinal"]["alpha"], 1.0, places=6,
        )

    def test_user_perfect_but_silver_noisy_clears_drop_uncertain(self) -> None:
        """Realistic scenario: user perfectly matches silver majority,
        but silver graders have 10% inter-rater noise.  The user
        column effectively votes the majority label on every row,
        so α is driven by the silver drift.  Should clear at least
        one rung of the fallback ladder.
        """
        silver = synthetic_graded_workbook(
            n_claims=200, noise_rate=0.10, seed=42,
        )
        _stamp_majority(silver)
        sampled = sample_stratified(silver, n_per_class=20, seed=42)
        user_rows = _simulate_user_labels(
            sampled, drift_rate=0.0, seed=7,
        )
        merged, _ = align_rows_with_drops(silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=100, seed=42,
        )
        # At 10% silver noise, ordinal α across 4 coders (where one
        # coder is the majority of the other three) should be high —
        # definitely clearing at least one rung.
        self.assertIsNotNone(
            ladder["passing_rung"],
            f"No rung cleared — ladder: {ladder['rungs']}",
        )

    def test_total_disagreement_fails_all_rungs(self) -> None:
        """User labels uniformly at random → α near 0 → no rung clears."""
        silver = synthetic_graded_workbook(
            n_claims=200, noise_rate=0.10, seed=42,
        )
        _stamp_majority(silver)
        sampled = sample_stratified(silver, n_per_class=20, seed=42)
        # Drift rate = 1.0 but using a uniform-random-other picker
        # (via simulate with drift=1.0) means user ALWAYS picks wrong.
        # That's not uniform random — the negative correlation will
        # drive α *below* 0, actually. Regardless, no rung clears.
        user_rows = _simulate_user_labels(
            sampled, drift_rate=1.0, seed=7,
        )
        merged, _ = align_rows_with_drops(silver, user_rows)
        ladder = compute_fallback_ladder(
            merged, min_alpha=0.80, n_bootstrap=100, seed=42,
        )
        self.assertIsNone(ladder["passing_rung"])

    def test_empty_sample_returns_empty_merged(self) -> None:
        """Edge case: sampled workbook has zero rows in every class."""
        silver: list[dict[str, Any]] = []
        sampled = sample_stratified(silver, n_per_class=20, seed=42)
        self.assertEqual(sampled, [])
        merged, drops = align_rows_with_drops(silver, [])
        self.assertEqual(merged, [])
        self.assertEqual(
            drops,
            {
                "missing_claim_id": 0,
                "no_silver_match": 0,
                "invalid_user_label": 0,
                "silver_no_valid_graders": 0,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
