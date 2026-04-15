"""Tests for evidence-provenance gating in ClaimGuard-CXR.

These tests lock in the safety rule: the verifier's trust certification must
depend on where the evidence came from, not just on the verifier's score.

Run with either:

    pytest tests/test_provenance.py -v
    python  tests/test_provenance.py

The second form runs the same tests without requiring pytest, so the CI /
Modal image doesn't need an extra dependency.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make `inference/` importable when the test file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.conformal_triage import (
    TriageResult,
    gate_triage_with_provenance,
)
from inference.provenance import (
    PROVENANCE_FIELDS,
    EvidenceSourceType,
    ProvenanceGateResult,
    ProvenanceTriageLabel,
    TrustTier,
    apply_provenance_gate,
    apply_provenance_gate_batch,
    classify_trust_tier,
    default_provenance,
    ensure_provenance_fields,
    is_certifiable,
    summarize_by_trust_tier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_example(
    claim: str,
    evidence: list[str],
    label: int,
    **provenance,
) -> dict:
    """Build a fake claim-evidence example for a test."""
    example = {
        "claim": claim,
        "evidence": evidence,
        "label": label,
        "pathology": "Pleural Effusion",
        "negative_type": "none",
        "patient_id": "patient_0001",
    }
    if provenance:
        example.update(provenance)
    return example


TRUSTED_EXAMPLE = _make_example(
    claim="There is a small left pleural effusion.",
    evidence=["Small left pleural effusion noted."],
    label=0,
    **default_provenance(source_type=EvidenceSourceType.ORACLE_REPORT_TEXT),
)

INDEPENDENT_RETRIEVED_EXAMPLE = _make_example(
    claim="Heart size is normal.",
    evidence=["Cardiothoracic ratio within normal limits."],
    label=0,
    **default_provenance(source_type=EvidenceSourceType.RETRIEVED_REPORT_TEXT),
)

SAME_MODEL_EXAMPLE = _make_example(
    claim="No evidence of pneumothorax.",
    evidence=["Lungs are clear bilaterally without pneumothorax."],
    label=0,
    **default_provenance(
        source_type=EvidenceSourceType.GENERATOR_OUTPUT,
        claim_generator_id="CheXagent-8b",
        evidence_generator_id="CheXagent-8b",
    ),
)

CROSS_MODEL_EXAMPLE = _make_example(
    claim="No evidence of pneumothorax.",
    evidence=["Lungs are clear bilaterally without pneumothorax."],
    label=0,
    **default_provenance(
        source_type=EvidenceSourceType.GENERATOR_OUTPUT,
        claim_generator_id="CheXagent-8b",
        evidence_generator_id="RadFM",
    ),
)

UNKNOWN_EXAMPLE = _make_example(
    claim="Mild cardiomegaly.",
    evidence=["unknown source"],
    label=0,
    # No provenance fields at all — simulates legacy data.
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyTrustTier(unittest.TestCase):
    def test_oracle_is_trusted(self):
        self.assertEqual(
            classify_trust_tier(EvidenceSourceType.ORACLE_REPORT_TEXT),
            TrustTier.TRUSTED,
        )

    def test_retrieved_is_independent(self):
        self.assertEqual(
            classify_trust_tier(EvidenceSourceType.RETRIEVED_REPORT_TEXT),
            TrustTier.INDEPENDENT,
        )

    def test_generator_same_model_is_same_model(self):
        self.assertEqual(
            classify_trust_tier(
                EvidenceSourceType.GENERATOR_OUTPUT,
                claim_generator_id="gpt-4",
                evidence_generator_id="gpt-4",
            ),
            TrustTier.SAME_MODEL,
        )

    def test_generator_cross_model_is_independent(self):
        self.assertEqual(
            classify_trust_tier(
                EvidenceSourceType.GENERATOR_OUTPUT,
                claim_generator_id="gpt-4",
                evidence_generator_id="claude",
            ),
            TrustTier.INDEPENDENT,
        )

    def test_generator_missing_id_is_same_model(self):
        """Conservative default: if we can't prove independence, assume self-loop."""
        self.assertEqual(
            classify_trust_tier(
                EvidenceSourceType.GENERATOR_OUTPUT,
                claim_generator_id="gpt-4",
                evidence_generator_id=None,
            ),
            TrustTier.SAME_MODEL,
        )
        self.assertEqual(
            classify_trust_tier(
                EvidenceSourceType.GENERATOR_OUTPUT,
                claim_generator_id=None,
                evidence_generator_id="gpt-4",
            ),
            TrustTier.SAME_MODEL,
        )

    def test_unknown_source_is_unknown(self):
        self.assertEqual(
            classify_trust_tier(EvidenceSourceType.UNKNOWN),
            TrustTier.UNKNOWN,
        )
        self.assertEqual(
            classify_trust_tier("garbage_source_string"),
            TrustTier.UNKNOWN,
        )

    def test_certifiable_tiers(self):
        self.assertTrue(is_certifiable(TrustTier.TRUSTED))
        self.assertTrue(is_certifiable(TrustTier.INDEPENDENT))
        self.assertFalse(is_certifiable(TrustTier.SAME_MODEL))
        self.assertFalse(is_certifiable(TrustTier.UNKNOWN))


# ---------------------------------------------------------------------------
# default_provenance / ensure_provenance_fields
# ---------------------------------------------------------------------------


class TestProvenanceDefaults(unittest.TestCase):
    def test_default_provenance_has_all_fields(self):
        prov = default_provenance()
        for field in PROVENANCE_FIELDS:
            self.assertIn(field, prov)

    def test_default_provenance_tier_matches_source(self):
        self.assertEqual(
            default_provenance(EvidenceSourceType.ORACLE_REPORT_TEXT)["evidence_trust_tier"],
            TrustTier.TRUSTED,
        )
        self.assertEqual(
            default_provenance(EvidenceSourceType.RETRIEVED_REPORT_TEXT)["evidence_trust_tier"],
            TrustTier.INDEPENDENT,
        )
        self.assertEqual(
            default_provenance(EvidenceSourceType.GENERATOR_OUTPUT)["evidence_trust_tier"],
            TrustTier.SAME_MODEL,
        )
        self.assertEqual(
            default_provenance(EvidenceSourceType.UNKNOWN)["evidence_trust_tier"],
            TrustTier.UNKNOWN,
        )

    def test_is_independent_matches_tier(self):
        oracle = default_provenance(EvidenceSourceType.ORACLE_REPORT_TEXT)
        retrieved = default_provenance(EvidenceSourceType.RETRIEVED_REPORT_TEXT)
        same_model = default_provenance(EvidenceSourceType.GENERATOR_OUTPUT)
        unknown = default_provenance(EvidenceSourceType.UNKNOWN)
        self.assertTrue(oracle["evidence_is_independent"])
        self.assertTrue(retrieved["evidence_is_independent"])
        self.assertFalse(same_model["evidence_is_independent"])
        self.assertFalse(unknown["evidence_is_independent"])

    def test_ensure_provenance_fields_backfills_legacy(self):
        legacy = {"claim": "x", "evidence": ["y"], "label": 0}
        enriched = ensure_provenance_fields(legacy)
        # Original object must not be mutated.
        self.assertNotIn("evidence_trust_tier", legacy)
        # Enriched has all fields, tier is UNKNOWN (not certifiable).
        self.assertEqual(enriched["evidence_trust_tier"], TrustTier.UNKNOWN)
        self.assertFalse(enriched["evidence_is_independent"])

    def test_ensure_provenance_fields_respects_existing(self):
        """If legacy data already has source_type, recompute the tier from it."""
        partial = {
            "claim": "x",
            "evidence": ["y"],
            "label": 0,
            "evidence_source_type": EvidenceSourceType.RETRIEVED_REPORT_TEXT,
        }
        enriched = ensure_provenance_fields(partial)
        self.assertEqual(enriched["evidence_trust_tier"], TrustTier.INDEPENDENT)
        self.assertTrue(enriched["evidence_is_independent"])


# ---------------------------------------------------------------------------
# apply_provenance_gate — the full label x tier truth table
# ---------------------------------------------------------------------------


class TestProvenanceGate(unittest.TestCase):
    def test_green_plus_trusted_is_supported_trusted(self):
        gate = apply_provenance_gate("green", TrustTier.TRUSTED)
        self.assertEqual(gate.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertFalse(gate.was_overridden)

    def test_green_plus_independent_is_supported_trusted(self):
        gate = apply_provenance_gate("green", TrustTier.INDEPENDENT)
        self.assertEqual(gate.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertFalse(gate.was_overridden)

    def test_green_plus_same_model_is_uncertified(self):
        gate = apply_provenance_gate("green", TrustTier.SAME_MODEL)
        self.assertEqual(gate.final_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)
        self.assertTrue(gate.was_overridden)

    def test_green_plus_unknown_is_uncertified(self):
        gate = apply_provenance_gate("green", TrustTier.UNKNOWN)
        self.assertEqual(gate.final_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)
        self.assertTrue(gate.was_overridden)

    def test_yellow_is_always_review(self):
        for tier in TrustTier.ALL:
            gate = apply_provenance_gate("yellow", tier)
            self.assertEqual(gate.final_label, ProvenanceTriageLabel.REVIEW_REQUIRED)
            self.assertFalse(gate.was_overridden)

    def test_red_is_always_contradicted(self):
        for tier in TrustTier.ALL:
            gate = apply_provenance_gate("red", tier)
            self.assertEqual(gate.final_label, ProvenanceTriageLabel.CONTRADICTED)
            self.assertFalse(gate.was_overridden)

    def test_batch_helper_matches_scalar(self):
        labels = ["green", "green", "yellow", "red"]
        tiers = [TrustTier.TRUSTED, TrustTier.SAME_MODEL,
                 TrustTier.UNKNOWN, TrustTier.INDEPENDENT]
        batch = apply_provenance_gate_batch(labels, tiers)
        scalars = [apply_provenance_gate(l, t) for l, t in zip(labels, tiers)]
        for b, s in zip(batch, scalars):
            self.assertEqual(b.final_label, s.final_label)
            self.assertEqual(b.was_overridden, s.was_overridden)


# ---------------------------------------------------------------------------
# gate_triage_with_provenance — integration with TriageResult
# ---------------------------------------------------------------------------


def _make_triage(label: str, claim: str = "test claim") -> TriageResult:
    return TriageResult(
        claim_text=claim,
        pathology_group="Pleural Effusion",
        faithfulness_score=0.95,
        conformal_pvalue=0.01,
        label=label,
        is_accepted=(label == "green"),
    )


class TestGateTriageWithProvenance(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gate_triage_with_provenance(
                [_make_triage("green")],
                [TrustTier.TRUSTED, TrustTier.SAME_MODEL],
            )

    def test_trusted_flow_end_to_end(self):
        r = _make_triage("green")
        [r_out] = gate_triage_with_provenance([r], [TrustTier.TRUSTED])
        self.assertEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertEqual(r_out.trust_tier, TrustTier.TRUSTED)
        # Original conformal label preserved for analysis.
        self.assertEqual(r_out.label, "green")

    def test_independent_retrieved_flow_end_to_end(self):
        r = _make_triage("green")
        [r_out] = gate_triage_with_provenance([r], [TrustTier.INDEPENDENT])
        self.assertEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)

    def test_same_model_never_certified(self):
        r = _make_triage("green")
        [r_out] = gate_triage_with_provenance([r], [TrustTier.SAME_MODEL])
        self.assertNotEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)

    def test_unknown_never_certified(self):
        r = _make_triage("green")
        [r_out] = gate_triage_with_provenance([r], [TrustTier.UNKNOWN])
        self.assertNotEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertEqual(r_out.final_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)

    def test_mixed_batch(self):
        rs = [_make_triage(l) for l in ["green", "green", "yellow", "red"]]
        tiers = [TrustTier.TRUSTED, TrustTier.SAME_MODEL,
                 TrustTier.TRUSTED, TrustTier.SAME_MODEL]
        out = gate_triage_with_provenance(rs, tiers)
        self.assertEqual(out[0].final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)
        self.assertEqual(out[1].final_label, ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED)
        self.assertEqual(out[2].final_label, ProvenanceTriageLabel.REVIEW_REQUIRED)
        self.assertEqual(out[3].final_label, ProvenanceTriageLabel.CONTRADICTED)


# ---------------------------------------------------------------------------
# Realistic fixture examples
# ---------------------------------------------------------------------------


class TestFixtureExamplesFlowThrough(unittest.TestCase):
    """Lock in the four cases the task explicitly called out."""

    def test_trusted_oracle_example_flows_normally(self):
        self.assertEqual(
            TRUSTED_EXAMPLE["evidence_trust_tier"], TrustTier.TRUSTED
        )
        r = _make_triage("green")
        [out] = gate_triage_with_provenance(
            [r], [TRUSTED_EXAMPLE["evidence_trust_tier"]]
        )
        self.assertEqual(out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)

    def test_independent_retrieved_example_flows_normally(self):
        self.assertEqual(
            INDEPENDENT_RETRIEVED_EXAMPLE["evidence_trust_tier"], TrustTier.INDEPENDENT
        )
        r = _make_triage("green")
        [out] = gate_triage_with_provenance(
            [r], [INDEPENDENT_RETRIEVED_EXAMPLE["evidence_trust_tier"]]
        )
        self.assertEqual(out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)

    def test_same_model_generator_example_is_never_trusted(self):
        self.assertEqual(
            SAME_MODEL_EXAMPLE["evidence_trust_tier"], TrustTier.SAME_MODEL
        )
        r = _make_triage("green")
        [out] = gate_triage_with_provenance(
            [r], [SAME_MODEL_EXAMPLE["evidence_trust_tier"]]
        )
        self.assertNotEqual(out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)

    def test_cross_model_generator_example_is_independent(self):
        """Claim by GPT, evidence by Claude -> independent, can be certified."""
        self.assertEqual(
            CROSS_MODEL_EXAMPLE["evidence_trust_tier"], TrustTier.INDEPENDENT
        )

    def test_unknown_provenance_example_is_never_trusted(self):
        # The legacy example has no provenance fields yet. Run it through the
        # backfill helper the eval pipeline uses, then verify it lands in
        # the UNKNOWN tier.
        enriched = ensure_provenance_fields(UNKNOWN_EXAMPLE)
        self.assertEqual(enriched["evidence_trust_tier"], TrustTier.UNKNOWN)
        r = _make_triage("green")
        [out] = gate_triage_with_provenance(
            [r], [enriched["evidence_trust_tier"]]
        )
        self.assertNotEqual(out.final_label, ProvenanceTriageLabel.SUPPORTED_TRUSTED)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


class TestSummarizeByTrustTier(unittest.TestCase):
    def test_counts_examples_by_tier(self):
        examples = [
            TRUSTED_EXAMPLE,
            TRUSTED_EXAMPLE,
            INDEPENDENT_RETRIEVED_EXAMPLE,
            SAME_MODEL_EXAMPLE,
            UNKNOWN_EXAMPLE,  # missing tier -> counted as UNKNOWN
        ]
        counts = summarize_by_trust_tier(examples)
        self.assertEqual(counts[TrustTier.TRUSTED], 2)
        self.assertEqual(counts[TrustTier.INDEPENDENT], 1)
        self.assertEqual(counts[TrustTier.SAME_MODEL], 1)
        self.assertEqual(counts[TrustTier.UNKNOWN], 1)


# ---------------------------------------------------------------------------
# Cross-check: canonical strings used in Modal retrieval eval script
# ---------------------------------------------------------------------------


class TestLiteralStringsMatchModalScript(unittest.TestCase):
    """The Modal retrieval eval builder hardcodes provenance strings (it
    cannot import inference.provenance from inside the container). Make sure
    the hardcoded literals still match the canonical values from this module.
    If this test fails, update modal_build_retrieval_eval.py."""

    def test_retrieved_report_text_literal(self):
        self.assertEqual(EvidenceSourceType.RETRIEVED_REPORT_TEXT, "retrieved_report_text")

    def test_independent_tier_literal(self):
        self.assertEqual(TrustTier.INDEPENDENT, "independent")

    def test_trusted_tier_literal(self):
        self.assertEqual(TrustTier.TRUSTED, "trusted")

    def test_unknown_tier_literal(self):
        self.assertEqual(TrustTier.UNKNOWN, "unknown")

    def test_same_model_tier_literal(self):
        self.assertEqual(TrustTier.SAME_MODEL, "same_model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
