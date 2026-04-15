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
    NEGATION_CUES,
    PATHOLOGY_KEYWORDS,
    UNCERTAINTY_CUES,
    VALID_LABELS,
    _rule_based_chexpert_label_vector,
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


# ---------------------------------------------------------------------------
# Rule-based CheXpert labeler (2026-04-14 reviewer fix — replaces the
# broken HF CheXbert path that stamped every row UNCERTAIN because
# the 14-head classifier wasn't on HF).
# ---------------------------------------------------------------------------


class TestRuleBasedChexpertLabeler(unittest.TestCase):
    """Lock the rule-based labeler behaviour.

    The labeler replaces the broken HF CheXbert load that the pre-
    flight reviewer flagged as a silent ghost grader.  Every row in
    Task 1 passes through this labeler on both the claim text and
    the GT report, producing 14-dim label vectors that feed into
    ``label_from_chexbert_diff``.  These tests pin the main
    behaviours:

        (1) Empty input → all zeros
        (2) Positive mentions → label 1
        (3) Negated mentions → label 2
        (4) Uncertainty cues → label 3
        (5) Multi-pathology reports → correct per-class assignment
        (6) "No Finding" inference — positive only when no other
            pathology is positive AND the text contains a
            no-finding phrase
        (7) Word-order variants (e.g., "heart is enlarged" vs
            "enlarged heart") both hit Cardiomegaly
    """

    def _vec(self, text: str) -> dict[str, int]:
        """Return a {pathology: label} dict, dropping zeros."""
        labels = _rule_based_chexpert_label_vector(text)
        return {
            p: l for p, l in zip(CHEXBERT_PATHOLOGIES, labels) if l != 0
        }

    def test_empty_input_all_zeros(self) -> None:
        vec = _rule_based_chexpert_label_vector("")
        self.assertEqual(vec, [0] * 14)

    def test_whitespace_input_all_zeros(self) -> None:
        vec = _rule_based_chexpert_label_vector("   \n\t  ")
        self.assertEqual(vec, [0] * 14)

    def test_positive_pneumothorax(self) -> None:
        d = self._vec("Large right-sided pneumothorax.")
        self.assertEqual(d.get("Pneumothorax"), 1)

    def test_negated_pneumothorax(self) -> None:
        d = self._vec("No pneumothorax seen.")
        self.assertEqual(d.get("Pneumothorax"), 2)

    def test_negation_without_cue(self) -> None:
        d = self._vec("Lungs without pneumothorax.")
        self.assertEqual(d.get("Pneumothorax"), 2)

    def test_uncertain_pneumonia(self) -> None:
        d = self._vec("Possible left lower lobe pneumonia.")
        self.assertEqual(d.get("Pneumonia"), 3)

    def test_uncertain_may_represent(self) -> None:
        d = self._vec("Opacity in the left base may represent atelectasis.")
        self.assertEqual(d.get("Atelectasis"), 3)

    def test_positive_cardiomegaly_word_order_variants(self) -> None:
        """Regression: 'heart is enlarged' must hit Cardiomegaly even
        though the literal keyword 'enlarged heart' has the opposite
        word order. Fixed in the 2026-04-14 reviewer pass."""
        for claim in (
            "The heart is enlarged.",
            "Cardiomegaly is present.",
            "Enlarged cardiac silhouette noted.",
            "Cardiac enlargement is seen.",
        ):
            d = self._vec(claim)
            self.assertEqual(
                d.get("Cardiomegaly"), 1,
                f"expected Cardiomegaly=1 in {claim!r}, got {d}",
            )

    def test_no_finding_inference_positive_when_clear_lungs(self) -> None:
        d = self._vec("Clear lungs. No acute cardiopulmonary process.")
        self.assertEqual(d.get("No Finding"), 1)

    def test_no_finding_inference_suppressed_by_positive_pathology(self) -> None:
        """'No Finding' must be 0 if ANY other pathology is positive,
        even when a no-finding phrase is present."""
        d = self._vec("The lungs are clear but cardiomegaly is present.")
        self.assertEqual(d.get("Cardiomegaly"), 1)
        # "No Finding" should NOT be 1 because cardiomegaly is positive.
        self.assertNotEqual(d.get("No Finding"), 1)

    def test_multi_pathology_report(self) -> None:
        text = (
            "Cardiomegaly is present with mild pulmonary edema. "
            "No pleural effusion. Possible left lower lobe atelectasis."
        )
        d = self._vec(text)
        self.assertEqual(d.get("Cardiomegaly"), 1)
        self.assertEqual(d.get("Edema"), 1)
        self.assertEqual(d.get("Pleural Effusion"), 2)
        self.assertEqual(d.get("Atelectasis"), 3)

    def test_negation_far_from_keyword_not_applied(self) -> None:
        """The negation window is ~30 chars before the keyword. A
        negation cue far earlier in the sentence must NOT negate a
        later-mentioned pathology."""
        text = (
            "There is no cardiomegaly, but a large pneumothorax is "
            "seen on the right with significant mediastinal shift."
        )
        d = self._vec(text)
        # "no cardiomegaly" → Cardiomegaly=2 (negated)
        self.assertEqual(d.get("Cardiomegaly"), 2)
        # "large pneumothorax" (30+ chars after "no") → Pneumothorax=1 (positive)
        self.assertEqual(d.get("Pneumothorax"), 1)

    def test_returns_list_of_14_integers(self) -> None:
        vec = _rule_based_chexpert_label_vector("some claim text")
        self.assertEqual(len(vec), 14)
        for v in vec:
            self.assertIsInstance(v, int)
            self.assertIn(v, {0, 1, 2, 3})

    def test_all_14_pathologies_have_keywords(self) -> None:
        """Every CheXbert pathology class should have at least one
        keyword in the lookup table."""
        for pathology in CHEXBERT_PATHOLOGIES:
            self.assertIn(pathology, PATHOLOGY_KEYWORDS)
            self.assertGreater(len(PATHOLOGY_KEYWORDS[pathology]), 0)

    def test_negation_cues_nonempty(self) -> None:
        self.assertGreater(len(NEGATION_CUES), 5)

    def test_uncertainty_cues_nonempty(self) -> None:
        self.assertGreater(len(UNCERTAINTY_CUES), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
