"""Tests for ``evaluation.regex_error_annotator``.

Task 4 (demoted) of the v3 sprint.  The annotator is diagnostic
metadata only — the tests verify that each regex fires on at least
12 positives and stays silent on at least 12 negatives per flag.

Run:
    python3 tests/test_regex_error_annotator.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.regex_error_annotator import (  # noqa: E402
    FLAG_NAMES,
    PATTERNS,
    annotate,
    annotate_all,
    count_flags,
    has_any_flag,
)


# ---------------------------------------------------------------------------
# Positive / negative fixtures per flag
# ---------------------------------------------------------------------------

MEAS_POS = [
    "3 mm pulmonary nodule in the right upper lobe.",
    "A 1.2 cm left upper lobe mass.",
    "Lesion measures 5 mm.",
    "Effusion approximately 2.4 cm in depth.",
    "7 mm soft tissue density.",
    "Nodule measuring 4 x 6 mm.",
    "6-mm opacity in the left lung.",
    "Mass 1.5 cm in diameter.",
    "Pulmonary nodule 9 mm diameter.",
    "0.8 cm lesion in the right lower lobe.",
    "Measures 3.0 mm in the apex.",
    "1 x 2 cm mass lesion.",
]

MEAS_NEG = [
    "Moderate left pleural effusion.",
    "No pneumothorax.",
    "Heart size is normal.",
    "Clear lungs bilaterally.",
    "Patchy opacity in the right lower lobe.",
    "Bibasilar atelectasis.",
    "Right upper lobe consolidation.",
    "Endotracheal tube in good position.",
    "Nasogastric tube coursing into the stomach.",
    "Cardiomegaly, otherwise unremarkable.",
    "New infiltrate noted.",
    "Stable findings.",
]

PRIOR_POS = [
    "Compared to the prior study, findings are unchanged.",
    "Unchanged compared to the prior exam.",
    "Since the previous study, the opacity has resolved.",
    "From the last film, no interval change.",
    "Compared with the prior radiograph.",
    "Interval change relative to the prior study.",
    "Compared to the prior chest x-ray, no new findings.",
    "Since the previous exam, mild improvement.",
    "Compared to the prior imaging, stable.",
    "Relative to the prior CXR, worsened.",
    "Compared to the last report, progression of disease.",
    "From the prior study, new opacity.",
]

PRIOR_NEG = [
    "Moderate left pleural effusion.",
    "3 mm pulmonary nodule in the right upper lobe.",
    "Heart size is at the upper limit of normal.",
    "Endotracheal tube terminates 3 cm above the carina.",
    "No focal consolidation.",
    "No pneumothorax identified.",
    "Stable cardiomediastinal silhouette.",
    "Low lung volumes with bibasilar atelectasis.",
    "Compared to normal, the heart is enlarged.",
    # ^ "compared to normal" does not mention prior-exam nouns
    "The ET tube has been repositioned.",
    "Bilateral pleural effusions noted.",
    "Unremarkable chest radiograph.",
]

DATE_POS = [
    "Opacity present since 01/15/2026.",
    "Finding first noted on 12/03/25.",
    "Study from 3/14/2024.",
    "Comparison with January 10, 2025.",
    "Compared with March 5 film.",
    "Since the 02/20/2025 exam.",
    "Noted on 11/30/2024.",
    "December 12, 2025 study.",
    "Changes since 4/1/2026.",
    "Last imaged on 9/15/25.",
    "Baseline from August 3.",
    "First identified 5/22/2024.",
]

DATE_NEG = [
    "Moderate left pleural effusion.",
    "3 mm pulmonary nodule.",
    "Compared to the prior exam.",
    "Since 3 days ago.",
    "Yesterday's film showed no change.",
    "Last week's study.",
    "No pneumothorax.",
    "Heart size is normal.",
    "ET tube 3 cm above the carina.",
    "Endotracheal tube in position.",
    "No acute cardiopulmonary disease.",
    "Low lung volumes.",
]

RELTIME_POS = [
    "Since 3 days ago, the opacity has enlarged.",
    "New since 5 weeks ago.",
    "Present for 2 months.",  # Will be skipped — no "ago"
    # (one of the negatives — keep 12 true positives)
    "Yesterday's film showed normal findings.",
    "Compared to yesterday's film, improved.",
    "Last week, no opacity.",
    "Last month's study showed pneumonia.",
    "Last year, findings were similar.",
    "Over the past 2 weeks, worsening.",
    "Over the last 5 days, progressive change.",
    "Present for 4 days ago.",
    "Worsened over the past 3 weeks.",
    "Change noted 7 days ago.",
    "12 months ago the finding was new.",
]

RELTIME_NEG = [
    "Moderate left pleural effusion.",
    "3 mm pulmonary nodule.",
    "Since the prior study.",
    "From the last exam.",
    "Compared to the prior exam.",
    "Opacity present since 01/15/2026.",
    "December 12, 2025 study.",
    "No change.",
    "Heart size normal.",
    "Endotracheal tube in position.",
    "Unremarkable chest radiograph.",
    "Patchy opacity in the left lower lobe.",
]


# ---------------------------------------------------------------------------
# Per-flag positive / negative tests
# ---------------------------------------------------------------------------


class TestMeasurementFlag(unittest.TestCase):
    def test_positives(self):
        for claim in MEAS_POS:
            flags = annotate(claim)["regex_flags"]
            self.assertIn(
                "fabricated_measurement",
                flags,
                f"MEAS false negative on: {claim!r}",
            )

    def test_negatives(self):
        for claim in MEAS_NEG:
            flags = annotate(claim)["regex_flags"]
            self.assertNotIn(
                "fabricated_measurement",
                flags,
                f"MEAS false positive on: {claim!r}",
            )


class TestPriorFlag(unittest.TestCase):
    def test_positives(self):
        for claim in PRIOR_POS:
            flags = annotate(claim)["regex_flags"]
            self.assertIn(
                "fabricated_prior",
                flags,
                f"PRIOR false negative on: {claim!r}",
            )

    def test_negatives(self):
        for claim in PRIOR_NEG:
            flags = annotate(claim)["regex_flags"]
            self.assertNotIn(
                "fabricated_prior",
                flags,
                f"PRIOR false positive on: {claim!r}",
            )


class TestDateFlag(unittest.TestCase):
    def test_positives(self):
        for claim in DATE_POS:
            flags = annotate(claim)["regex_flags"]
            self.assertIn(
                "fabricated_date",
                flags,
                f"DATE false negative on: {claim!r}",
            )

    def test_negatives(self):
        for claim in DATE_NEG:
            flags = annotate(claim)["regex_flags"]
            self.assertNotIn(
                "fabricated_date",
                flags,
                f"DATE false positive on: {claim!r}",
            )


class TestRelTimeFlag(unittest.TestCase):
    def test_positives(self):
        """Require at least 12 matches from the positive fixture.
        (Some fixtures are intentionally tricky and may miss; we only
        require a floor, not 100% recall.)"""
        matched = 0
        for claim in RELTIME_POS:
            flags = annotate(claim)["regex_flags"]
            if "fabricated_relative_time" in flags:
                matched += 1
        self.assertGreaterEqual(
            matched, 12,
            f"RELTIME only matched {matched}/{len(RELTIME_POS)}",
        )

    def test_negatives(self):
        for claim in RELTIME_NEG:
            flags = annotate(claim)["regex_flags"]
            self.assertNotIn(
                "fabricated_relative_time",
                flags,
                f"RELTIME false positive on: {claim!r}",
            )


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


class TestAPIContract(unittest.TestCase):
    def test_annotate_empty_input(self):
        self.assertEqual(annotate("")["regex_flags"], [])
        self.assertEqual(annotate(None)["regex_flags"], [])  # type: ignore[arg-type]
        self.assertEqual(annotate(123)["regex_flags"], [])  # type: ignore[arg-type]

    def test_annotate_all_shape(self):
        claims = ["3 mm nodule", "no opacity", "compared to the prior study"]
        results = annotate_all(claims)
        self.assertEqual(len(results), len(claims))
        for r in results:
            self.assertIn("regex_flags", r)
            self.assertIsInstance(r["regex_flags"], list)

    def test_flag_names_matches_patterns_keys(self):
        self.assertEqual(set(FLAG_NAMES), set(PATTERNS.keys()))
        # Order must match too
        self.assertEqual(list(FLAG_NAMES), list(PATTERNS.keys()))

    def test_has_any_flag(self):
        self.assertTrue(has_any_flag("3 mm nodule"))
        self.assertFalse(has_any_flag("Moderate left pleural effusion."))

    def test_count_flags_shape_stable(self):
        counts = count_flags([])
        self.assertEqual(set(counts.keys()), set(FLAG_NAMES))
        for v in counts.values():
            self.assertEqual(v, 0)

    def test_count_flags_totals(self):
        claims = MEAS_POS[:3] + PRIOR_POS[:2] + MEAS_NEG[:4]
        counts = count_flags(claims)
        self.assertGreaterEqual(counts["fabricated_measurement"], 3)
        self.assertGreaterEqual(counts["fabricated_prior"], 2)

    def test_multiple_flags_on_same_claim(self):
        """A claim can fire more than one flag — e.g., a fabricated
        measurement with a fabricated prior reference."""
        claim = (
            "3 mm nodule, compared to the prior study from 01/15/2026."
        )
        flags = annotate(claim)["regex_flags"]
        self.assertIn("fabricated_measurement", flags)
        self.assertIn("fabricated_prior", flags)
        self.assertIn("fabricated_date", flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
