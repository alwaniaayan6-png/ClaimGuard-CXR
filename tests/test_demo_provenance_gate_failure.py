"""Tests for pure helpers in ``scripts.demo_provenance_gate_failure``.

The heavy dual-run experiment (``_run_demo``) needs Modal + H100 + the
v1 verifier checkpoint + CheXagent + OpenI image data.  It is only
exercised by the real remote run.  This test file pins down the
pure-Python helpers that drive the provenance-gate logic and the
downgrade-rate accounting — everything a CPU-only laptop can validate:

* Module constants locked to the plan-mandated strings.
* ``GateDemoConfig`` — JSON round-trip + defaults.
* ``_conformal_label_from_score`` — green/yellow/red boundaries.
* ``build_gate_demo_row`` — trust-tier classification + gate result.
* ``compute_gate_demo_stats`` — per-condition downgrade counts.
* ``synthetic_gate_demo_workbook`` — determinism + plan-mandated
  ``downgrade_rate > 0.5`` assertion on the synthetic fixture.
* ``load_workbook_claims`` — JSON schema guards.
* ``pair_claims_with_evidence`` — image intersection + same/cross
  pairing, ordering, missing-field tolerance.

Note: the Modal app / volume / remote function may or may not exist at
import time depending on whether ``modal`` is installed; that path is
guarded in the module itself and is not tested here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.provenance import ProvenanceTriageLabel, TrustTier  # noqa: E402
from scripts.demo_provenance_gate_failure import (  # noqa: E402
    APP_NAME,
    DEFAULT_HIGH_SCORE_THRESHOLD,
    RUN_A_GENERATOR_ID,
    RUN_A_ID,
    RUN_B_GENERATOR_ID,
    RUN_B_ID,
    VOLUME_NAME,
    GateDemoConfig,
    GateDemoRow,
    _conformal_label_from_score,
    _extract_generator_id,
    _is_error_sentinel,
    build_gate_demo_row,
    compute_gate_demo_stats,
    load_workbook_claims,
    pair_claims_with_evidence,
    synthetic_gate_demo_workbook,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants(unittest.TestCase):
    def test_app_name(self) -> None:
        self.assertEqual(APP_NAME, "claimguard-provenance-gate-demo")

    def test_volume_name(self) -> None:
        self.assertEqual(VOLUME_NAME, "claimguard-data")

    def test_run_ids_distinct(self) -> None:
        self.assertEqual(RUN_A_ID, "run-a")
        self.assertEqual(RUN_B_ID, "run-b")
        self.assertNotEqual(RUN_A_ID, RUN_B_ID)

    def test_generator_ids_distinct(self) -> None:
        self.assertEqual(RUN_A_GENERATOR_ID, "chexagent-8b-run-a")
        self.assertEqual(RUN_B_GENERATOR_ID, "chexagent-8b-run-b")
        self.assertNotEqual(RUN_A_GENERATOR_ID, RUN_B_GENERATOR_ID)

    def test_default_threshold_in_unit_interval(self) -> None:
        self.assertGreater(DEFAULT_HIGH_SCORE_THRESHOLD, 0.0)
        self.assertLess(DEFAULT_HIGH_SCORE_THRESHOLD, 1.0)


# ---------------------------------------------------------------------------
# GateDemoConfig
# ---------------------------------------------------------------------------


class TestGateDemoConfig(unittest.TestCase):
    def test_defaults_match_plan(self) -> None:
        c = GateDemoConfig()
        self.assertEqual(c.max_images, 100)
        self.assertEqual(c.temperature, 0.7)
        self.assertEqual(c.top_p, 0.9)
        self.assertEqual(c.hf_backbone, "roberta-large")
        self.assertEqual(c.high_score_threshold, DEFAULT_HIGH_SCORE_THRESHOLD)
        self.assertNotEqual(c.seed_run_a, c.seed_run_b)
        self.assertFalse(c.skip_generation)

    def test_json_round_trip(self) -> None:
        c = GateDemoConfig(
            max_images=50,
            seed_run_a=11,
            seed_run_b=22,
            high_score_threshold=0.9,
        )
        c2 = GateDemoConfig.from_json(c.to_json())
        self.assertEqual(c, c2)

    def test_json_valid(self) -> None:
        parsed = json.loads(GateDemoConfig().to_json())
        self.assertIn("verifier_checkpoint", parsed)
        self.assertIn("image_seed", parsed)


# ---------------------------------------------------------------------------
# _conformal_label_from_score
# ---------------------------------------------------------------------------


class TestConformalLabelFromScore(unittest.TestCase):
    def test_high_score_is_green(self) -> None:
        self.assertEqual(
            _conformal_label_from_score(0.9, high_score_threshold=0.5),
            "green",
        )

    def test_score_exactly_at_threshold_is_green(self) -> None:
        # Strict ``>=`` so the threshold boundary is inclusive.
        self.assertEqual(
            _conformal_label_from_score(0.5, high_score_threshold=0.5),
            "green",
        )

    def test_medium_score_is_yellow(self) -> None:
        # 0.3 >= 0.5 * 0.5 = 0.25, but < 0.5
        self.assertEqual(
            _conformal_label_from_score(0.3, high_score_threshold=0.5),
            "yellow",
        )

    def test_low_score_is_red(self) -> None:
        self.assertEqual(
            _conformal_label_from_score(0.1, high_score_threshold=0.5),
            "red",
        )

    def test_score_at_yellow_threshold_is_yellow(self) -> None:
        self.assertEqual(
            _conformal_label_from_score(0.25, high_score_threshold=0.5),
            "yellow",
        )

    def test_custom_threshold(self) -> None:
        # Tighter threshold: 0.9 → green needs >=0.9, yellow needs >=0.45
        self.assertEqual(
            _conformal_label_from_score(0.95, high_score_threshold=0.9),
            "green",
        )
        self.assertEqual(
            _conformal_label_from_score(0.6, high_score_threshold=0.9),
            "yellow",
        )
        self.assertEqual(
            _conformal_label_from_score(0.1, high_score_threshold=0.9),
            "red",
        )


# ---------------------------------------------------------------------------
# build_gate_demo_row
# ---------------------------------------------------------------------------


class TestBuildGateDemoRow(unittest.TestCase):
    def _row(
        self, *,
        score: float,
        claim_gen: str,
        evidence_gen: str,
        condition: str = "same_run",
    ) -> GateDemoRow:
        return build_gate_demo_row(
            claim_id="c1",
            claim_text="heart is enlarged",
            evidence_text="cardiomegaly noted",
            claim_generator_id=claim_gen,
            evidence_generator_id=evidence_gen,
            verifier_score=score,
            condition=condition,
        )

    def test_same_model_green_is_downgraded(self) -> None:
        # Same generator id on both sides + high score → SAME_MODEL,
        # green gets overridden to SUPPORTED_UNCERTIFIED.
        row = self._row(
            score=0.95,
            claim_gen=RUN_A_GENERATOR_ID,
            evidence_gen=RUN_A_GENERATOR_ID,
        )
        self.assertEqual(row.trust_tier, TrustTier.SAME_MODEL)
        self.assertEqual(row.conformal_label, "green")
        self.assertEqual(
            row.gate_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED,
        )
        self.assertTrue(row.was_overridden)
        self.assertTrue(row.high_score)

    def test_independent_green_is_certified(self) -> None:
        # Different generator ids + high score → INDEPENDENT,
        # green passes through as SUPPORTED_TRUSTED.
        row = self._row(
            score=0.95,
            claim_gen=RUN_A_GENERATOR_ID,
            evidence_gen=RUN_B_GENERATOR_ID,
            condition="cross_run",
        )
        self.assertEqual(row.trust_tier, TrustTier.INDEPENDENT)
        self.assertEqual(row.conformal_label, "green")
        self.assertEqual(
            row.gate_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED,
        )
        self.assertFalse(row.was_overridden)
        self.assertTrue(row.high_score)

    def test_low_score_is_contradicted_regardless_of_tier(self) -> None:
        # A red claim → CONTRADICTED whether SAME_MODEL or INDEPENDENT.
        same = self._row(
            score=0.05,
            claim_gen=RUN_A_GENERATOR_ID,
            evidence_gen=RUN_A_GENERATOR_ID,
        )
        cross = self._row(
            score=0.05,
            claim_gen=RUN_A_GENERATOR_ID,
            evidence_gen=RUN_B_GENERATOR_ID,
            condition="cross_run",
        )
        self.assertEqual(
            same.gate_label, ProvenanceTriageLabel.CONTRADICTED,
        )
        self.assertEqual(
            cross.gate_label, ProvenanceTriageLabel.CONTRADICTED,
        )
        self.assertFalse(same.high_score)
        self.assertFalse(cross.high_score)

    def test_medium_score_is_review_required(self) -> None:
        row = self._row(
            score=0.3,
            claim_gen=RUN_A_GENERATOR_ID,
            evidence_gen=RUN_A_GENERATOR_ID,
        )
        self.assertEqual(row.conformal_label, "yellow")
        self.assertEqual(
            row.gate_label, ProvenanceTriageLabel.REVIEW_REQUIRED,
        )
        self.assertFalse(row.was_overridden)

    def test_optional_audit_fields_default_empty(self) -> None:
        row = self._row(
            score=0.9,
            claim_gen="a", evidence_gen="a",
        )
        self.assertEqual(row.image_file, "")
        self.assertEqual(row.pathology, "")
        self.assertEqual(row.ground_truth_label, "")

    def test_optional_audit_fields_forwarded(self) -> None:
        row = build_gate_demo_row(
            claim_id="c1",
            claim_text="x",
            evidence_text="y",
            claim_generator_id="a",
            evidence_generator_id="a",
            verifier_score=0.9,
            condition="same_run",
            image_file="img_001.png",
            pathology="Cardiomegaly",
            ground_truth_label="SUPPORTED",
        )
        self.assertEqual(row.image_file, "img_001.png")
        self.assertEqual(row.pathology, "Cardiomegaly")
        self.assertEqual(row.ground_truth_label, "SUPPORTED")

    def test_score_coerced_to_float(self) -> None:
        # Integer input is gracefully converted.
        row = self._row(
            score=1, claim_gen="a", evidence_gen="a",
        )
        self.assertIsInstance(row.verifier_score, float)
        self.assertEqual(row.verifier_score, 1.0)


# ---------------------------------------------------------------------------
# compute_gate_demo_stats
# ---------------------------------------------------------------------------


class TestComputeGateDemoStats(unittest.TestCase):
    def test_empty_input(self) -> None:
        stats = compute_gate_demo_stats([])
        self.assertEqual(stats["n_rows_total"], 0)
        for cond in ("same_run", "cross_run"):
            self.assertEqual(stats["by_condition"][cond]["n_claims"], 0)
            self.assertEqual(
                stats["by_condition"][cond]["n_high_score"], 0,
            )
            self.assertEqual(
                stats["by_condition"][cond]["downgrade_rate"], 0.0,
            )
        self.assertEqual(stats["downgrade_rate_diff"], 0.0)

    def test_synthetic_plan_assertion(self) -> None:
        # THE plan requirement: downgrade_rate > 0.5 on a synthetic
        # fixture.  With the built-in synthetic workbook, the same_run
        # downgrade_rate should be exactly 1.0 (every SAME_MODEL green
        # is overridden) and cross_run should be 0.0 (no overrides).
        rows = synthetic_gate_demo_workbook(n_rows=10, seed=0)
        stats = compute_gate_demo_stats(rows)
        same = stats["by_condition"]["same_run"]
        cross = stats["by_condition"]["cross_run"]
        self.assertGreater(same["downgrade_rate"], 0.5)
        self.assertEqual(cross["downgrade_rate"], 0.0)
        self.assertGreater(stats["downgrade_rate_diff"], 0.5)

    def test_synthetic_counts_are_balanced(self) -> None:
        rows = synthetic_gate_demo_workbook(n_rows=15, seed=0)
        stats = compute_gate_demo_stats(rows)
        self.assertEqual(
            stats["by_condition"]["same_run"]["n_claims"], 15,
        )
        self.assertEqual(
            stats["by_condition"]["cross_run"]["n_claims"], 15,
        )
        self.assertEqual(stats["n_rows_total"], 30)

    def test_all_green_same_run_gives_full_downgrade(self) -> None:
        rows = [
            build_gate_demo_row(
                claim_id=f"c{i}",
                claim_text="x",
                evidence_text="y",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_A_GENERATOR_ID,
                verifier_score=0.95,
                condition="same_run",
            )
            for i in range(5)
        ]
        stats = compute_gate_demo_stats(rows)
        by = stats["by_condition"]["same_run"]
        self.assertEqual(by["n_claims"], 5)
        self.assertEqual(by["n_high_score"], 5)
        self.assertEqual(by["n_certified_pre_gate"], 5)
        self.assertEqual(by["n_certified_post_gate"], 0)
        self.assertEqual(by["downgrade_rate"], 1.0)

    def test_all_green_cross_run_gives_zero_downgrade(self) -> None:
        rows = [
            build_gate_demo_row(
                claim_id=f"c{i}",
                claim_text="x",
                evidence_text="y",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_B_GENERATOR_ID,
                verifier_score=0.95,
                condition="cross_run",
            )
            for i in range(5)
        ]
        stats = compute_gate_demo_stats(rows)
        by = stats["by_condition"]["cross_run"]
        self.assertEqual(by["n_claims"], 5)
        self.assertEqual(by["n_certified_pre_gate"], 5)
        self.assertEqual(by["n_certified_post_gate"], 5)
        self.assertEqual(by["downgrade_rate"], 0.0)

    def test_no_green_claims_downgrade_rate_is_zero(self) -> None:
        # Edge case: if nothing is green pre-gate, the denominator is
        # zero — stats should report 0.0, not NaN.
        rows = [
            build_gate_demo_row(
                claim_id=f"c{i}",
                claim_text="x",
                evidence_text="y",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_A_GENERATOR_ID,
                verifier_score=0.02,  # red
                condition="same_run",
            )
            for i in range(3)
        ]
        stats = compute_gate_demo_stats(rows)
        self.assertEqual(
            stats["by_condition"]["same_run"]["downgrade_rate"], 0.0,
        )

    def test_only_one_condition_other_is_zero(self) -> None:
        rows = [
            build_gate_demo_row(
                claim_id="c1",
                claim_text="x",
                evidence_text="y",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_A_GENERATOR_ID,
                verifier_score=0.95,
                condition="same_run",
            )
        ]
        stats = compute_gate_demo_stats(rows)
        self.assertEqual(
            stats["by_condition"]["same_run"]["n_claims"], 1,
        )
        self.assertEqual(
            stats["by_condition"]["cross_run"]["n_claims"], 0,
        )


# ---------------------------------------------------------------------------
# synthetic_gate_demo_workbook
# ---------------------------------------------------------------------------


class TestSyntheticGateDemoWorkbook(unittest.TestCase):
    def test_deterministic_under_seed(self) -> None:
        a = synthetic_gate_demo_workbook(n_rows=5, seed=7)
        b = synthetic_gate_demo_workbook(n_rows=5, seed=7)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(x.verifier_score, y.verifier_score)
            self.assertEqual(x.gate_label, y.gate_label)

    def test_different_seeds_give_different_scores(self) -> None:
        a = synthetic_gate_demo_workbook(n_rows=5, seed=1)
        b = synthetic_gate_demo_workbook(n_rows=5, seed=2)
        # Not required to differ row-by-row, but the totals should.
        self.assertNotEqual(
            sum(r.verifier_score for r in a),
            sum(r.verifier_score for r in b),
        )

    def test_emits_both_conditions(self) -> None:
        rows = synthetic_gate_demo_workbook(n_rows=4, seed=0)
        conditions = {r.condition for r in rows}
        self.assertEqual(conditions, {"same_run", "cross_run"})
        same = [r for r in rows if r.condition == "same_run"]
        cross = [r for r in rows if r.condition == "cross_run"]
        self.assertEqual(len(same), 4)
        self.assertEqual(len(cross), 4)

    def test_all_rows_have_valid_labels(self) -> None:
        rows = synthetic_gate_demo_workbook(n_rows=3, seed=0)
        for r in rows:
            self.assertIn(
                r.gate_label, ProvenanceTriageLabel.ALL,
            )
            self.assertIn(r.trust_tier, TrustTier.ALL)


# ---------------------------------------------------------------------------
# load_workbook_claims
# ---------------------------------------------------------------------------


class TestLoadWorkbookClaims(unittest.TestCase):
    def _write(self, obj: Any) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(obj, f)
        f.close()
        self.addCleanup(lambda: os.unlink(f.name))
        return f.name

    def test_valid_list(self) -> None:
        path = self._write([{"claim_id": "c1"}, {"claim_id": "c2"}])
        self.assertEqual(len(load_workbook_claims(path)), 2)

    def test_non_list_raises(self) -> None:
        path = self._write({"not": "a list"})
        with self.assertRaises(ValueError):
            load_workbook_claims(path)

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_workbook_claims("/nonexistent/workbook.json")

    def test_empty_list_is_ok(self) -> None:
        path = self._write([])
        self.assertEqual(load_workbook_claims(path), [])


# ---------------------------------------------------------------------------
# _extract_generator_id — silent-overwrite bug fence (reviewer bug 2)
# ---------------------------------------------------------------------------


class TestExtractGeneratorId(unittest.TestCase):
    def test_empty_list_returns_fallback(self) -> None:
        self.assertEqual(
            _extract_generator_id([], fallback="default-id"),
            "default-id",
        )

    def test_all_rows_missing_id_returns_fallback(self) -> None:
        claims = [
            {"claim_id": "c1", "extracted_claim": "x"},
            {"claim_id": "c2", "extracted_claim": "y"},
        ]
        self.assertEqual(
            _extract_generator_id(claims, fallback="chex-a"),
            "chex-a",
        )

    def test_all_rows_empty_string_id_returns_fallback(self) -> None:
        claims = [
            {"claim_generator_id": ""},
            {"claim_generator_id": ""},
        ]
        self.assertEqual(
            _extract_generator_id(claims, fallback="chex-a"),
            "chex-a",
        )

    def test_all_rows_non_string_id_returns_fallback(self) -> None:
        # Legacy workbooks may stamp an int or None; we treat those as
        # "missing" and fall back to the constant, not silently accept.
        claims = [
            {"claim_generator_id": 42},
            {"claim_generator_id": None},
        ]
        self.assertEqual(
            _extract_generator_id(claims, fallback="chex-a"),
            "chex-a",
        )

    def test_unanimous_id_returns_that_id(self) -> None:
        claims = [
            {"claim_generator_id": "chexagent-8b-run-a"},
            {"claim_generator_id": "chexagent-8b-run-a"},
            {"claim_generator_id": "chexagent-8b-run-a"},
        ]
        self.assertEqual(
            _extract_generator_id(claims, fallback="fallback-id"),
            "chexagent-8b-run-a",
        )

    def test_unanimous_with_some_missing_returns_the_id(self) -> None:
        # Some rows missing the field entirely — the present ones agree.
        claims = [
            {"claim_generator_id": "chexagent-8b-run-a"},
            {"extracted_claim": "x"},  # no generator id
            {"claim_generator_id": "chexagent-8b-run-a"},
        ]
        self.assertEqual(
            _extract_generator_id(claims, fallback="fallback-id"),
            "chexagent-8b-run-a",
        )

    def test_conflicting_ids_raise(self) -> None:
        # This is the whole point of the helper: silently overwriting
        # with the last row's id (the prior bug) is replaced by a
        # loud ValueError.
        claims = [
            {"claim_generator_id": "chexagent-8b-run-a"},
            {"claim_generator_id": "chexagent-8b-run-silver"},  # stale
        ]
        with self.assertRaises(ValueError) as ctx:
            _extract_generator_id(claims, fallback="fallback-id")
        self.assertIn("conflicting", str(ctx.exception))
        self.assertIn("chexagent-8b-run-a", str(ctx.exception))
        self.assertIn("chexagent-8b-run-silver", str(ctx.exception))

    def test_three_way_conflict_raises(self) -> None:
        claims = [
            {"claim_generator_id": "a"},
            {"claim_generator_id": "b"},
            {"claim_generator_id": "c"},
        ]
        with self.assertRaises(ValueError):
            _extract_generator_id(claims, fallback="fallback-id")


# ---------------------------------------------------------------------------
# _is_error_sentinel — CheXagent failure sentinel detection
# ---------------------------------------------------------------------------


class TestIsErrorSentinel(unittest.TestCase):
    def test_chexagent_unavailable_sentinel_detected(self) -> None:
        self.assertTrue(_is_error_sentinel("[CheXagent unavailable]"))

    def test_generation_error_sentinel_detected(self) -> None:
        self.assertTrue(
            _is_error_sentinel("[Generation error: OOM on image 42]")
        )

    def test_empty_string_is_sentinel(self) -> None:
        self.assertTrue(_is_error_sentinel(""))

    def test_whitespace_only_is_sentinel(self) -> None:
        self.assertTrue(_is_error_sentinel("   \n\t  "))

    def test_real_report_with_brackets_not_sentinel(self) -> None:
        # Radiologists sometimes write "[1]" or "[no prior]" in reports.
        # We must not drop those as sentinels.
        self.assertFalse(
            _is_error_sentinel(
                "Findings: Cardiomegaly [1]. Impression: Enlarged heart."
            )
        )

    def test_real_report_without_brackets_not_sentinel(self) -> None:
        self.assertFalse(
            _is_error_sentinel(
                "Findings: The lungs are clear. Impression: No acute "
                "cardiopulmonary process."
            )
        )

    def test_bracketed_non_error_string_not_sentinel(self) -> None:
        # A bracketed string that doesn't contain "error" or
        # "unavailable" must not be dropped.
        self.assertFalse(_is_error_sentinel("[no prior]"))

    def test_sentinel_with_leading_whitespace_detected(self) -> None:
        self.assertTrue(_is_error_sentinel("  [Generation error: foo]  "))

    def test_case_insensitive_detection(self) -> None:
        self.assertTrue(_is_error_sentinel("[GENERATION ERROR: upstream]"))
        self.assertTrue(_is_error_sentinel("[chexagent UNAVAILABLE]"))


# ---------------------------------------------------------------------------
# pair_claims_with_evidence
# ---------------------------------------------------------------------------


class TestPairClaimsWithEvidence(unittest.TestCase):
    def _claim(
        self, *,
        claim_id: str,
        image_file: str,
        extracted_claim: str,
        generated_report: str,
        generator_id: str,
    ) -> dict:
        return {
            "claim_id": claim_id,
            "image_file": image_file,
            "extracted_claim": extracted_claim,
            "generated_report": generated_report,
            "claim_generator_id": generator_id,
        }

    def test_matched_image_emits_both_conditions(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="heart is enlarged",
                generated_report="Report A: heart is enlarged.",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="cardiac outline prominent",
                generated_report="Report B: cardiac outline prominent.",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        # Exactly 2 pairs: one same_run + one cross_run for the same
        # claim from run A.
        self.assertEqual(len(pairs), 2)
        conds = [p["condition"] for p in pairs]
        self.assertIn("same_run", conds)
        self.assertIn("cross_run", conds)

    def test_pair_uses_full_generated_report_as_evidence(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="claim text",
                generated_report="FULL REPORT A",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="claim text other",
                generated_report="FULL REPORT B",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        same = next(p for p in pairs if p["condition"] == "same_run")
        cross = next(p for p in pairs if p["condition"] == "cross_run")
        self.assertEqual(same["evidence_text"], "FULL REPORT A")
        self.assertEqual(cross["evidence_text"], "FULL REPORT B")

    def test_generator_ids_pulled_from_workbook(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="c",
                generated_report="r",
                generator_id="custom-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="c2",
                generated_report="r2",
                generator_id="custom-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        for p in pairs:
            self.assertEqual(p["claim_generator_id"], "custom-run-a")
        cross = next(p for p in pairs if p["condition"] == "cross_run")
        self.assertEqual(cross["evidence_generator_id"], "custom-run-b")
        same = next(p for p in pairs if p["condition"] == "same_run")
        self.assertEqual(same["evidence_generator_id"], "custom-run-a")

    def test_image_only_in_run_a_is_skipped(self) -> None:
        # If run B never processed this image, we cannot form a matched
        # cross_run comparison — both rows are dropped.
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_A_only.png",
                extracted_claim="claim",
                generated_report="A report",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_other.png",
                extracted_claim="c",
                generated_report="B report",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(pairs, [])

    def test_empty_extracted_claim_is_skipped(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="   ",  # whitespace-only
                generated_report="A",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="valid",
                generated_report="B",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(pairs, [])

    def test_empty_image_file_is_skipped(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="",
                extracted_claim="valid",
                generated_report="A",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="valid",
                generated_report="B",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(pairs, [])

    def test_generator_ids_default_when_missing_from_workbook(self) -> None:
        # Workbook entries lacking ``claim_generator_id`` fall back to
        # the canonical ``RUN_A_GENERATOR_ID`` / ``RUN_B_GENERATOR_ID``
        # strings.  This is the graceful path for legacy workbooks.
        claims_a = [{
            "claim_id": "a1",
            "image_file": "img_001.png",
            "extracted_claim": "claim",
            "generated_report": "A report",
        }]
        claims_b = [{
            "claim_id": "b1",
            "image_file": "img_001.png",
            "extracted_claim": "other",
            "generated_report": "B report",
        }]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(len(pairs), 2)
        for p in pairs:
            self.assertEqual(p["claim_generator_id"], RUN_A_GENERATOR_ID)
        cross = next(p for p in pairs if p["condition"] == "cross_run")
        self.assertEqual(cross["evidence_generator_id"], RUN_B_GENERATOR_ID)

    def test_multiple_claims_same_image(self) -> None:
        # If run A has 3 claims for one image and run B has that image,
        # each claim should get a full same+cross pair → 6 rows.
        claims_a = [
            self._claim(
                claim_id=f"a{i}",
                image_file="img_001.png",
                extracted_claim=f"claim {i}",
                generated_report="A report",
                generator_id="chexagent-8b-run-a",
            )
            for i in range(3)
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="anything",
                generated_report="B report",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(len(pairs), 6)
        same_rows = [p for p in pairs if p["condition"] == "same_run"]
        cross_rows = [p for p in pairs if p["condition"] == "cross_run"]
        self.assertEqual(len(same_rows), 3)
        self.assertEqual(len(cross_rows), 3)

    def test_pair_ids_are_unique(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="c1",
                generated_report="A",
                generator_id="chexagent-8b-run-a",
            )
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="c2",
                generated_report="B",
                generator_id="chexagent-8b-run-b",
            )
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        ids = [p["claim_id"] for p in pairs]
        self.assertEqual(len(set(ids)), len(ids))
        # Suffixed with __same / __cross for clarity.
        self.assertTrue(any("__same" in i for i in ids))
        self.assertTrue(any("__cross" in i for i in ids))

    def test_error_sentinel_evidence_is_filtered_out(self) -> None:
        """Reviewer bug — a CheXagent failure sentinel must NOT feed
        into the verifier as evidence.  The image should be treated
        as missing in both directions, skipping the pair entirely."""
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="heart enlarged",
                generated_report="[Generation error: OOM]",
                generator_id="chexagent-8b-run-a",
            ),
            self._claim(
                claim_id="a2",
                image_file="img_002.png",
                extracted_claim="normal lungs",
                generated_report="Real report content.",
                generator_id="chexagent-8b-run-a",
            ),
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="heart enlarged",
                generated_report="Real run-B content.",
                generator_id="chexagent-8b-run-b",
            ),
            self._claim(
                claim_id="b2",
                image_file="img_002.png",
                extracted_claim="normal lungs",
                generated_report="Real run-B content.",
                generator_id="chexagent-8b-run-b",
            ),
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        # img_001 has an error sentinel in run A → dropped entirely.
        # img_002 has real content in both → emits 2 pair rows.
        self.assertEqual(len(pairs), 2)
        image_files = {p["image_file"] for p in pairs}
        self.assertEqual(image_files, {"img_002.png"})

    def test_chexagent_unavailable_sentinel_is_filtered_out(self) -> None:
        """The '[CheXagent unavailable]' sentinel must also be treated
        as missing evidence."""
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="heart enlarged",
                generated_report="Real content.",
                generator_id="chexagent-8b-run-a",
            ),
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="heart enlarged",
                generated_report="[CheXagent unavailable]",
                generator_id="chexagent-8b-run-b",
            ),
        ]
        pairs = pair_claims_with_evidence(
            claims_run_a=claims_a, claims_run_b=claims_b
        )
        self.assertEqual(pairs, [])

    def test_conflicting_run_a_ids_raise_value_error(self) -> None:
        """Integration check: the strict generator-id helper must
        propagate its ValueError out of pair_claims_with_evidence so
        corrupt workbooks fail loudly instead of silently producing
        wrong tier classifications."""
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="c1",
                generated_report="A",
                generator_id="chexagent-8b-run-a",
            ),
            self._claim(
                claim_id="a2",
                image_file="img_002.png",
                extracted_claim="c2",
                generated_report="A",
                generator_id="chexagent-8b-run-silver",  # stale leak
            ),
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="c1",
                generated_report="B",
                generator_id="chexagent-8b-run-b",
            ),
        ]
        with self.assertRaises(ValueError):
            pair_claims_with_evidence(
                claims_run_a=claims_a, claims_run_b=claims_b
            )

    def test_conflicting_run_b_ids_raise_value_error(self) -> None:
        claims_a = [
            self._claim(
                claim_id="a1",
                image_file="img_001.png",
                extracted_claim="c1",
                generated_report="A",
                generator_id="chexagent-8b-run-a",
            ),
        ]
        claims_b = [
            self._claim(
                claim_id="b1",
                image_file="img_001.png",
                extracted_claim="c1",
                generated_report="B",
                generator_id="chexagent-8b-run-b",
            ),
            self._claim(
                claim_id="b2",
                image_file="img_002.png",
                extracted_claim="c2",
                generated_report="B",
                generator_id="chexagent-8b-run-b-STALE",
            ),
        ]
        with self.assertRaises(ValueError):
            pair_claims_with_evidence(
                claims_run_a=claims_a, claims_run_b=claims_b
            )


# ---------------------------------------------------------------------------
# End-to-end: synthetic fixture passes the plan assertion
# ---------------------------------------------------------------------------


class TestPlanAssertion(unittest.TestCase):
    """Explicit end-to-end test mirroring the Task 9 plan requirement.

    From ``/Users/aayanalwani/.claude/plans/rosy-discovering-bubble.md``:

        > Tests: ``tests/test_provenance_gate_demo.py`` — stub assertion
        > that ``downgrade_rate > 0.5`` on a synthetic fixture.

    This test class pins the plan-mandated acceptance criterion so any
    future refactor that breaks the gate logic fails loudly here.
    """

    def test_synthetic_downgrade_rate_above_half(self) -> None:
        rows = synthetic_gate_demo_workbook(n_rows=25, seed=42)
        stats = compute_gate_demo_stats(rows)
        self.assertGreater(
            stats["by_condition"]["same_run"]["downgrade_rate"],
            0.5,
            msg="Plan requires same_run downgrade_rate > 0.5",
        )

    def test_synthetic_diff_is_positive(self) -> None:
        rows = synthetic_gate_demo_workbook(n_rows=25, seed=42)
        stats = compute_gate_demo_stats(rows)
        # Gate should downgrade strictly more same-model claims than
        # cross-model claims.
        self.assertGreater(stats["downgrade_rate_diff"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
