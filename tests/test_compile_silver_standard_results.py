"""Tests for ``scripts.compile_silver_standard_results`` helpers.

Focuses on the pure functions:

* ``majority_vote`` — tie-breaking + invalid-label handling
* ``build_coder_matrix`` — ordinal encoding + missing-label → NaN
* ``stamp_regex_flags`` — Task 4 wiring
* ``per_grader_accuracy_vs_majority``
* ``majority_label_distribution``
* ``baseline_comparison`` — with and without verifier scores
* ``compile_results`` end-to-end on the dry-run synthetic workbook

The verifier inference path is smoke-tested with a non-existent
checkpoint (expected to return ``status == skipped``).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.compile_silver_standard_results import (  # noqa: E402
    GRADER_LABEL_KEYS,
    LABEL_TO_ORDINAL,
    RADFLAG_PRECISION,
    VALID_LABELS,
    baseline_comparison,
    build_coder_matrix,
    compile_results,
    majority_label_distribution,
    majority_vote,
    per_grader_accuracy_vs_majority,
    run_verifier_inference,
    stamp_regex_flags,
    synthetic_graded_workbook,
)


class TestMajorityVote(unittest.TestCase):
    def test_unanimous(self) -> None:
        self.assertEqual(
            majority_vote(["SUPPORTED", "SUPPORTED", "SUPPORTED"]),
            "SUPPORTED",
        )

    def test_two_of_three(self) -> None:
        self.assertEqual(
            majority_vote(["SUPPORTED", "SUPPORTED", "CONTRADICTED"]),
            "SUPPORTED",
        )

    def test_all_different_uncertain(self) -> None:
        self.assertEqual(
            majority_vote(
                ["SUPPORTED", "CONTRADICTED", "NOVEL_PLAUSIBLE"]
            ),
            "UNCERTAIN",
        )

    def test_empty_uncertain(self) -> None:
        self.assertEqual(majority_vote([]), "UNCERTAIN")

    def test_all_invalid_uncertain(self) -> None:
        self.assertEqual(majority_vote(["", "BOGUS", "x"]), "UNCERTAIN")

    def test_two_valid_one_invalid_still_votes(self) -> None:
        # The invalid one is dropped; two valid same-label wins.
        self.assertEqual(
            majority_vote(["SUPPORTED", "SUPPORTED", ""]),
            "SUPPORTED",
        )

    def test_one_valid_one_different_uncertain(self) -> None:
        # Only two valid votes, disagree → no plurality ≥ 2 → UNCERTAIN.
        self.assertEqual(
            majority_vote(["SUPPORTED", "CONTRADICTED", ""]),
            "UNCERTAIN",
        )


class TestBuildCoderMatrix(unittest.TestCase):
    def test_shape_and_dtype(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "SUPPORTED",
                "grader_claude_label": "SUPPORTED",
                "grader_medgemma_label": "SUPPORTED",
            },
            {
                "grader_chexbert_label": "CONTRADICTED",
                "grader_claude_label": "UNCERTAIN",
                "grader_medgemma_label": "CONTRADICTED",
            },
        ]
        m = build_coder_matrix(wb)
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m.dtype, np.float64)

    def test_ordinal_encoding(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "SUPPORTED",
                "grader_claude_label": "CONTRADICTED",
                "grader_medgemma_label": "NOVEL_PLAUSIBLE",
            }
        ]
        m = build_coder_matrix(wb)
        self.assertEqual(m[0, 0], LABEL_TO_ORDINAL["SUPPORTED"])
        self.assertEqual(m[1, 0], LABEL_TO_ORDINAL["CONTRADICTED"])
        self.assertEqual(m[2, 0], LABEL_TO_ORDINAL["NOVEL_PLAUSIBLE"])

    def test_missing_labels_become_nan(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "",
                "grader_claude_label": "BOGUS",
                "grader_medgemma_label": "UNCERTAIN",
            }
        ]
        m = build_coder_matrix(wb)
        self.assertTrue(np.isnan(m[0, 0]))
        self.assertTrue(np.isnan(m[1, 0]))
        self.assertEqual(m[2, 0], LABEL_TO_ORDINAL["UNCERTAIN"])


class TestStampRegexFlags(unittest.TestCase):
    def test_stamps_in_place(self) -> None:
        wb = [
            {"extracted_claim": "A 3 mm nodule is present."},
            {"extracted_claim": "No pneumothorax."},
        ]
        counts = stamp_regex_flags(wb)
        self.assertIn("regex_flags", wb[0])
        self.assertIn("fabricated_measurement", wb[0]["regex_flags"])
        self.assertEqual(wb[1]["regex_flags"], [])
        self.assertEqual(counts["fabricated_measurement"], 1)

    def test_empty_claim_ok(self) -> None:
        wb = [{"extracted_claim": ""}]
        counts = stamp_regex_flags(wb)
        self.assertEqual(wb[0]["regex_flags"], [])
        self.assertEqual(sum(counts.values()), 0)


class TestPerGraderAccuracy(unittest.TestCase):
    def test_all_graders_match_majority(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "SUPPORTED",
                "grader_claude_label": "SUPPORTED",
                "grader_medgemma_label": "SUPPORTED",
                "majority_label": "SUPPORTED",
            }
        ]
        acc = per_grader_accuracy_vs_majority(wb)
        for key in GRADER_LABEL_KEYS:
            self.assertEqual(acc[key], 1.0)

    def test_one_grader_off(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "SUPPORTED",
                "grader_claude_label": "SUPPORTED",
                "grader_medgemma_label": "CONTRADICTED",
                "majority_label": "SUPPORTED",
            },
            {
                "grader_chexbert_label": "CONTRADICTED",
                "grader_claude_label": "CONTRADICTED",
                "grader_medgemma_label": "CONTRADICTED",
                "majority_label": "CONTRADICTED",
            },
        ]
        acc = per_grader_accuracy_vs_majority(wb)
        self.assertEqual(acc["grader_chexbert_label"], 1.0)
        self.assertEqual(acc["grader_claude_label"], 1.0)
        self.assertEqual(acc["grader_medgemma_label"], 0.5)

    def test_invalid_rows_ignored(self) -> None:
        wb = [
            {
                "grader_chexbert_label": "",  # missing grader
                "grader_claude_label": "SUPPORTED",
                "grader_medgemma_label": "SUPPORTED",
                "majority_label": "SUPPORTED",
            }
        ]
        acc = per_grader_accuracy_vs_majority(wb)
        self.assertTrue(math.isnan(acc["grader_chexbert_label"]))
        self.assertEqual(acc["grader_claude_label"], 1.0)


class TestMajorityDistribution(unittest.TestCase):
    def test_counts_every_label(self) -> None:
        wb = [
            {"majority_label": "SUPPORTED"},
            {"majority_label": "SUPPORTED"},
            {"majority_label": "CONTRADICTED"},
        ]
        dist = majority_label_distribution(wb)
        self.assertEqual(dist["SUPPORTED"], 2)
        self.assertEqual(dist["CONTRADICTED"], 1)
        self.assertEqual(dist["UNCERTAIN"], 0)
        # Every valid label must be present even with zero count.
        for label in VALID_LABELS:
            self.assertIn(label, dist)


class TestBaselineComparison(unittest.TestCase):
    def test_no_verifier_scores(self) -> None:
        wb = [
            {
                "verifier_score": None,
                "majority_label": "SUPPORTED",
            }
        ]
        block = baseline_comparison(wb)
        self.assertEqual(block["radflag_precision"], RADFLAG_PRECISION)
        self.assertIsNone(block["our_precision"])
        self.assertIsNone(block["delta"])
        self.assertEqual(block["n_scored"], 0)

    def test_all_accepted_all_supported(self) -> None:
        wb = [
            {"verifier_score": 0.9, "majority_label": "SUPPORTED"},
            {"verifier_score": 0.8, "majority_label": "SUPPORTED"},
        ]
        block = baseline_comparison(wb)
        self.assertEqual(block["our_precision"], 1.0)
        self.assertAlmostEqual(block["delta"], 1.0 - RADFLAG_PRECISION)

    def test_mixed(self) -> None:
        wb = [
            {"verifier_score": 0.9, "majority_label": "SUPPORTED"},
            {"verifier_score": 0.9, "majority_label": "CONTRADICTED"},
            {"verifier_score": 0.3, "majority_label": "SUPPORTED"},  # rejected
        ]
        block = baseline_comparison(wb)
        self.assertEqual(block["n_accepted"], 2)
        self.assertEqual(block["our_precision"], 0.5)


class TestVerifierInferenceSkip(unittest.TestCase):
    def test_skip_when_no_checkpoint_path(self) -> None:
        wb = [{"extracted_claim": "x", "ground_truth_report": "y"}]
        meta = run_verifier_inference(wb, None)
        self.assertEqual(meta["status"], "skipped")

    def test_skip_when_checkpoint_missing(self) -> None:
        wb = [{"extracted_claim": "x", "ground_truth_report": "y"}]
        meta = run_verifier_inference(wb, "/nonexistent/ckpt.pt")
        self.assertEqual(meta["status"], "skipped")
        self.assertIsNone(wb[0].get("verifier_score"))


class TestCompileResultsDryRun(unittest.TestCase):
    def test_compile_synthetic_workbook_end_to_end(self) -> None:
        wb = synthetic_graded_workbook(
            n_claims=60, noise_rate=0.03, seed=123
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "out.json"
            results, passed = compile_results(
                wb,
                output_json=out_json,
                expect_alpha_min=0.80,
                verifier_checkpoint=None,
                n_bootstrap=200,
                seed=7,
            )
            # Assertions inside the with-block so the tempdir still exists.
            self.assertTrue(
                passed, f"α too low: {results['krippendorff_alpha']}"
            )
            self.assertEqual(results["n_claims"], 60)
            self.assertEqual(
                sum(results["majority_label_distribution"].values()), 60
            )
            csv_path = out_json.with_suffix(".csv")
            self.assertTrue(csv_path.exists())

    def test_high_noise_fails_gate(self) -> None:
        wb = synthetic_graded_workbook(
            n_claims=60, noise_rate=0.60, seed=123
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "out.json"
            results, passed = compile_results(
                wb,
                output_json=out_json,
                expect_alpha_min=0.80,
                verifier_checkpoint=None,
                n_bootstrap=100,
                seed=7,
            )
        self.assertFalse(passed)
        self.assertLess(results["krippendorff_alpha"]["point"], 0.80)

    def test_regex_flags_merged(self) -> None:
        wb = synthetic_graded_workbook(
            n_claims=60, noise_rate=0.05, seed=42
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "out.json"
            results, _ = compile_results(
                wb,
                output_json=out_json,
                expect_alpha_min=0.0,  # always pass
                verifier_checkpoint=None,
                n_bootstrap=100,
                seed=7,
            )
        counts = results["regex_flag_counts"]
        # The synthetic NOVEL_HALLUCINATED templates include a 7 mm
        # nodule and a "compared to the prior study" phrase, so the
        # measurement + relative-time flags must both appear.
        self.assertGreater(counts["fabricated_measurement"], 0)
        self.assertGreater(counts["fabricated_relative_time"], 0)


class TestCliExitCodes(unittest.TestCase):
    def test_dry_run_exit_code_is_zero_on_pass(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_REPO_ROOT, "scripts", "compile_silver_standard_results.py"),
                "--dry-run",
                "--output-json",
                "/tmp/silver_cli_pass.json",
                "--no-verifier",
                "--expect-alpha-min",
                "0.50",
                "--dry-run-noise-rate",
                "0.05",
                "--dry-run-n-claims",
                "80",
                "--n-bootstrap",
                "100",
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_dry_run_exit_code_is_two_on_fail(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_REPO_ROOT, "scripts", "compile_silver_standard_results.py"),
                "--dry-run",
                "--output-json",
                "/tmp/silver_cli_fail.json",
                "--no-verifier",
                "--expect-alpha-min",
                "0.99",
                "--dry-run-noise-rate",
                "0.30",
                "--dry-run-n-claims",
                "40",
                "--n-bootstrap",
                "100",
            ],
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr.decode())


if __name__ == "__main__":
    unittest.main(verbosity=2)
