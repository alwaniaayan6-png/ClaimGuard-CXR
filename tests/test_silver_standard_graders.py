"""Tests for the pure helpers in ``scripts.generate_silver_standard_graders``.

We only test the parts that don't need Modal, GPU, or network access:

* ``parse_grader_json`` — robust-to-messy-VLM-output JSON parser
* ``claim_to_pathology`` — free-text → CheXbert class keyword lookup
* ``label_from_chexbert_diff`` — 14-dim diff → 5-class schema mapping
* ``LABEL_TO_ORDINAL`` — consistency with ``VALID_LABELS``
* ``CHEXBERT_PATHOLOGIES`` — exactly 14 unique classes

The Modal/GPU entrypoints and the Anthropic API path are smoke-checked
by the live Modal run, not here.
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.generate_silver_standard_graders import (  # noqa: E402
    CHEXBERT_PATHOLOGIES,
    LABEL_TO_ORDINAL,
    VALID_LABELS,
    claim_to_pathology,
    label_from_chexbert_diff,
    parse_grader_json,
)


# Index constants for readability in the diff tests.
IDX_CARDIOMEGALY = CHEXBERT_PATHOLOGIES.index("Cardiomegaly")
IDX_PNEUMOTHORAX = CHEXBERT_PATHOLOGIES.index("Pneumothorax")
IDX_EFFUSION = CHEXBERT_PATHOLOGIES.index("Pleural Effusion")

# CheXbert label codes.
CX_BLANK, CX_POS, CX_NEG, CX_UNC = 0, 1, 2, 3


class TestParseGraderJson(unittest.TestCase):
    def test_clean_json(self) -> None:
        out = parse_grader_json(
            '{"label": "SUPPORTED", "confidence": "high", "rationale": "both agree"}'
        )
        self.assertEqual(out, ("SUPPORTED", "high", "both agree"))

    def test_preamble_json(self) -> None:
        raw = (
            "Sure, here is my assessment:\n"
            '{"label": "CONTRADICTED", "confidence": "medium", '
            '"rationale": "laterality mismatch"}'
        )
        label, conf, rat = parse_grader_json(raw)
        self.assertEqual(label, "CONTRADICTED")
        self.assertEqual(conf, "medium")
        self.assertIn("laterality", rat)

    def test_regex_fallback_on_unparseable_json(self) -> None:
        raw = "I think this claim is NOVEL_HALLUCINATED because..."
        label, conf, _ = parse_grader_json(raw)
        self.assertEqual(label, "NOVEL_HALLUCINATED")
        self.assertEqual(conf, "low")

    def test_totally_garbage_returns_uncertain(self) -> None:
        out = parse_grader_json("who knows")
        self.assertEqual(out, ("UNCERTAIN", "low", "unparseable"))

    def test_empty_returns_uncertain(self) -> None:
        out = parse_grader_json("")
        self.assertEqual(out, ("UNCERTAIN", "low", "empty response"))

    def test_none_returns_uncertain(self) -> None:
        out = parse_grader_json(None)  # type: ignore[arg-type]
        self.assertEqual(out, ("UNCERTAIN", "low", "empty response"))

    def test_invalid_label_falls_through(self) -> None:
        # Valid JSON but label is not in VALID_LABELS.  Should drop to
        # the regex fallback; the regex won't find any valid token
        # either, so we get UNCERTAIN.
        out = parse_grader_json(
            '{"label": "BOGUS", "confidence": "high", "rationale": "x"}'
        )
        self.assertEqual(out[0], "UNCERTAIN")

    def test_invalid_confidence_coerced_to_medium(self) -> None:
        out = parse_grader_json(
            '{"label": "UNCERTAIN", "confidence": "bogus", "rationale": "x"}'
        )
        self.assertEqual(out, ("UNCERTAIN", "medium", "x"))

    def test_rationale_clipped_at_500_chars(self) -> None:
        long_rationale = "a" * 2000
        raw = (
            '{"label": "SUPPORTED", "confidence": "high", '
            f'"rationale": "{long_rationale}"}}'
        )
        _, _, rat = parse_grader_json(raw)
        self.assertLessEqual(len(rat), 500)


class TestClaimToPathology(unittest.TestCase):
    def test_cardiomegaly_keyword(self) -> None:
        self.assertEqual(
            claim_to_pathology("There is mild cardiomegaly."),
            "Cardiomegaly",
        )

    def test_pneumothorax_keyword(self) -> None:
        self.assertEqual(
            claim_to_pathology("No pneumothorax is seen."),
            "Pneumothorax",
        )

    def test_pleural_effusion_keyword(self) -> None:
        self.assertEqual(
            claim_to_pathology("A small pleural effusion at the left base."),
            "Pleural Effusion",
        )

    def test_no_finding_keyword(self) -> None:
        self.assertEqual(
            claim_to_pathology("normal chest radiograph"),
            "No Finding",
        )

    def test_unknown_claim_returns_none(self) -> None:
        self.assertIsNone(claim_to_pathology("The patient has frosty hair."))

    def test_case_insensitive(self) -> None:
        self.assertEqual(
            claim_to_pathology("ATELECTASIS at the left base"),
            "Atelectasis",
        )


class TestLabelFromChexbertDiff(unittest.TestCase):
    def _vec(self, idx: int, code: int) -> list[int]:
        v = [CX_BLANK] * 14
        v[idx] = code
        return v

    def test_both_positive_supported(self) -> None:
        v = self._vec(IDX_CARDIOMEGALY, CX_POS)
        self.assertEqual(
            label_from_chexbert_diff("cardiomegaly seen", v, v),
            ("SUPPORTED", "high"),
        )

    def test_both_negative_supported(self) -> None:
        v = self._vec(IDX_PNEUMOTHORAX, CX_NEG)
        self.assertEqual(
            label_from_chexbert_diff("no pneumothorax", v, v),
            ("SUPPORTED", "high"),
        )

    def test_pos_vs_neg_contradicted(self) -> None:
        claim = self._vec(IDX_CARDIOMEGALY, CX_POS)
        gt = self._vec(IDX_CARDIOMEGALY, CX_NEG)
        self.assertEqual(
            label_from_chexbert_diff("cardiomegaly", claim, gt),
            ("CONTRADICTED", "high"),
        )

    def test_neg_vs_pos_contradicted(self) -> None:
        claim = self._vec(IDX_EFFUSION, CX_NEG)
        gt = self._vec(IDX_EFFUSION, CX_POS)
        self.assertEqual(
            label_from_chexbert_diff("no pleural effusion", claim, gt),
            ("CONTRADICTED", "high"),
        )

    def test_claim_pos_gt_blank_hallucinated(self) -> None:
        claim = self._vec(IDX_PNEUMOTHORAX, CX_POS)
        gt = [CX_BLANK] * 14
        self.assertEqual(
            label_from_chexbert_diff("pneumothorax", claim, gt),
            ("NOVEL_HALLUCINATED", "low"),
        )

    def test_claim_neg_gt_blank_supported_low(self) -> None:
        claim = self._vec(IDX_PNEUMOTHORAX, CX_NEG)
        gt = [CX_BLANK] * 14
        self.assertEqual(
            label_from_chexbert_diff("no pneumothorax", claim, gt),
            ("SUPPORTED", "low"),
        )

    def test_chexbert_uncertain_propagates(self) -> None:
        claim = self._vec(IDX_CARDIOMEGALY, CX_UNC)
        gt = self._vec(IDX_CARDIOMEGALY, CX_POS)
        self.assertEqual(
            label_from_chexbert_diff("cardiomegaly", claim, gt),
            ("UNCERTAIN", "medium"),
        )

    def test_no_keyword_match_returns_uncertain(self) -> None:
        self.assertEqual(
            label_from_chexbert_diff("blob", [CX_BLANK] * 14, [CX_BLANK] * 14),
            ("UNCERTAIN", "low"),
        )

    def test_none_vectors_return_uncertain(self) -> None:
        self.assertEqual(
            label_from_chexbert_diff("cardiomegaly", None, [CX_BLANK] * 14),
            ("UNCERTAIN", "low"),
        )
        self.assertEqual(
            label_from_chexbert_diff("cardiomegaly", [CX_BLANK] * 14, None),
            ("UNCERTAIN", "low"),
        )


class TestSchemaInvariants(unittest.TestCase):
    def test_chexbert_pathologies_count(self) -> None:
        self.assertEqual(len(CHEXBERT_PATHOLOGIES), 14)

    def test_chexbert_pathologies_unique(self) -> None:
        self.assertEqual(
            len(set(CHEXBERT_PATHOLOGIES)), len(CHEXBERT_PATHOLOGIES)
        )

    def test_label_ordinal_matches_valid_labels(self) -> None:
        self.assertEqual(set(LABEL_TO_ORDINAL), VALID_LABELS)

    def test_label_ordinals_are_contiguous_0_to_4(self) -> None:
        self.assertEqual(
            sorted(LABEL_TO_ORDINAL.values()), [0, 1, 2, 3, 4]
        )

    def test_supported_is_zero(self) -> None:
        self.assertEqual(LABEL_TO_ORDINAL["SUPPORTED"], 0)

    def test_uncertain_is_four(self) -> None:
        self.assertEqual(LABEL_TO_ORDINAL["UNCERTAIN"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
