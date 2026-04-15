"""Tests for the ``LLMClaimExtractor`` wiring in prepare_eval_data / demo
and for the extractor-fidelity driver.

Scope (network-free):
    * Rule-based fallback path in ``LLMClaimExtractor.extract_claims``
      is deterministic on fixed golden inputs.
    * ``prepare_eval_data.extract_claims(..., use_llm_extractor=False)``
      preserves v1 behaviour.
    * ``prepare_eval_data.extract_claims(..., use_llm_extractor=True)``
      routes through the LLMClaimExtractor class (we monkey-patch a
      deterministic stub to avoid loading Phi-3-mini).
    * The ``extractor_fidelity.compute_fidelity`` driver works end-to-end
      with a trivial extractor and gracefully degrades when
      ``sacrebleu`` / ``bert_score`` / NLI are unavailable.

Run:
    python3 tests/test_llm_claim_extractor.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.decomposer.llm_claim_extractor import LLMClaimExtractor  # noqa: E402


GOLDEN_REPORTS = [
    (
        "FINDINGS: The lungs are clear. There is no pleural effusion. "
        "Heart size is normal. No acute osseous abnormality.",
        # Expected: 4 atomic claims, all covering core findings
        {"lungs", "pleural effusion", "heart", "osseous"},
    ),
    (
        "IMPRESSION: 1. Moderate right-sided pleural effusion with "
        "associated atelectasis. 2. No pneumothorax.",
        {"pleural effusion", "atelectasis", "pneumothorax"},
    ),
    (
        "New patchy opacity in the left lower lobe, concerning for "
        "pneumonia. Compared to the prior study from 2 weeks ago.",
        {"opacity", "pneumonia", "prior"},
    ),
    (
        "Endotracheal tube terminates 3 cm above the carina. "
        "Nasogastric tube coursing into the stomach. "
        "No pneumothorax.",
        {"endotracheal", "nasogastric", "pneumothorax"},
    ),
    (
        "Unchanged small left apical pneumothorax. "
        "Bibasilar atelectasis. Mild cardiomegaly.",
        {"pneumothorax", "atelectasis", "cardiomegaly"},
    ),
]


class TestRuleBasedFallback(unittest.TestCase):
    """The rule-based path must work without any model weights."""

    def setUp(self) -> None:
        self.extractor = LLMClaimExtractor(use_llm=False)

    def test_all_golden_reports_extract_expected_keywords(self):
        for report, expected_tokens in GOLDEN_REPORTS:
            claims = self.extractor.extract_claims(report)
            joined = " ".join(c["claim"].lower() for c in claims)
            for token in expected_tokens:
                self.assertIn(
                    token,
                    joined,
                    f"Expected token {token!r} missing from extracted "
                    f"claims for report: {report[:60]!r}",
                )

    def test_empty_report_returns_empty(self):
        self.assertEqual(self.extractor.extract_claims(""), [])
        self.assertEqual(self.extractor.extract_claims("  "), [])
        self.assertEqual(self.extractor.extract_claims("abc"), [])

    def test_pathology_labels_present(self):
        for report, _ in GOLDEN_REPORTS:
            for c in self.extractor.extract_claims(report):
                self.assertIn("pathology", c)
                self.assertIsInstance(c["pathology"], str)


class TestPrepareEvalDataWiring(unittest.TestCase):
    """Verify the --use-llm-extractor plumb in ``scripts/prepare_eval_data.py``."""

    def test_naive_path_matches_v1_behaviour(self):
        # Import here so the module-level LLM singleton stays cold
        from scripts.prepare_eval_data import extract_claims

        report = (
            "Moderate left pleural effusion. "
            "No pneumothorax. Heart size within normal limits."
        )
        claims = extract_claims(report, use_llm_extractor=False)
        self.assertGreater(len(claims), 0)
        # Should include all three findings
        joined = " ".join(claims).lower()
        self.assertIn("pleural effusion", joined)
        self.assertIn("pneumothorax", joined)
        self.assertIn("heart", joined)

    def test_llm_path_goes_through_extractor_class(self):
        """Monkey-patch the shared singleton so this runs network-free."""
        import scripts.prepare_eval_data as ped

        class _StubExtractor:
            def extract_claims(self, report_text, max_claims=20):
                return [
                    {"claim": "stub: left pleural effusion", "pathology": "Pleural Effusion"},
                    {"claim": "stub: no pneumothorax", "pathology": "Pneumothorax"},
                ]

        saved = ped._LLM_EXTRACTOR
        try:
            ped._LLM_EXTRACTOR = _StubExtractor()
            claims = ped.extract_claims(
                "Moderate left pleural effusion. No pneumothorax.",
                use_llm_extractor=True,
            )
            self.assertEqual(claims, [
                "stub: left pleural effusion",
                "stub: no pneumothorax",
            ])
        finally:
            ped._LLM_EXTRACTOR = saved

    def test_llm_path_with_short_report_returns_empty(self):
        import scripts.prepare_eval_data as ped

        self.assertEqual(
            ped.extract_claims("", use_llm_extractor=True),
            [],
        )
        # Under 10 chars should short-circuit before loading the LLM
        self.assertEqual(
            ped.extract_claims("abc", use_llm_extractor=True),
            [],
        )


class TestExtractorFidelityDriver(unittest.TestCase):
    def test_compute_fidelity_runs_end_to_end(self):
        """compute_fidelity must not crash when bert_score / sacrebleu
        are unavailable — it gracefully returns 0 for those metrics."""
        from evaluation.extractor_fidelity import (
            compute_fidelity,
            _rule_based_extractor,
        )

        reports = [r for r, _ in GOLDEN_REPORTS]
        metrics = compute_fidelity(reports, _rule_based_extractor)
        self.assertEqual(metrics.n_reports, len(reports))
        self.assertGreaterEqual(metrics.mean_claims_per_report, 0.0)
        # BLEU / BERTScore / NLI can be 0 if their optional deps are
        # missing, but must be finite floats in [0, 1].
        for field in (
            "bleu4",
            "bertscore_f1",
            "nli_entailment_rate",
            "roundtrip_bertscore_f1",
        ):
            val = getattr(metrics, field)
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_compute_fidelity_empty_reports(self):
        from evaluation.extractor_fidelity import (
            compute_fidelity,
            _rule_based_extractor,
        )

        metrics = compute_fidelity([], _rule_based_extractor)
        self.assertEqual(metrics.n_reports, 0)
        self.assertEqual(metrics.mean_claims_per_report, 0.0)

    def test_report_loader_json_list_of_strings(self):
        import json
        import tempfile

        from evaluation.extractor_fidelity import _load_reports

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(["report A", "report B"], f)
            path = Path(f.name)
        try:
            reports = _load_reports(path, n=None)
            self.assertEqual(reports, ["report A", "report B"])
            reports = _load_reports(path, n=1)
            self.assertEqual(reports, ["report A"])
        finally:
            path.unlink()

    def test_report_loader_json_list_of_dicts(self):
        import json
        import tempfile

        from evaluation.extractor_fidelity import _load_reports

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {"report": "A"},
                    {"text": "B"},
                    {"section_findings": "C"},
                ],
                f,
            )
            path = Path(f.name)
        try:
            reports = _load_reports(path, n=None)
            self.assertEqual(reports, ["A", "B", "C"])
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
