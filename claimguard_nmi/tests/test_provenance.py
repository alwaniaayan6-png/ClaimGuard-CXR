"""Unit tests for the provenance gate."""
from __future__ import annotations

from claimguard_nmi.provenance import (
    GateInput,
    PostGateLabel,
    TrustTier,
    apply_gate,
    compute_trust_tier,
    downgrade_rate,
)


def test_oracle_evidence_is_trusted():
    inp = GateInput(
        claim_generator_id="chexagent",
        evidence_generator_id=None,
        evidence_source_type="oracle",
        pre_gate_label="supported",
    )
    assert compute_trust_tier(inp) is TrustTier.TRUSTED
    assert apply_gate(inp) is PostGateLabel.SUPPORTED_TRUSTED


def test_retrieved_evidence_is_independent():
    inp = GateInput(
        claim_generator_id="chexagent",
        evidence_generator_id=None,
        evidence_source_type="retrieval",
        pre_gate_label="supported",
    )
    assert compute_trust_tier(inp) is TrustTier.INDEPENDENT
    assert apply_gate(inp) is PostGateLabel.SUPPORTED_TRUSTED


def test_same_model_evidence_is_downgraded():
    inp = GateInput(
        claim_generator_id="chexagent",
        evidence_generator_id="chexagent",
        evidence_source_type="vlm_generated",
        pre_gate_label="supported",
    )
    assert compute_trust_tier(inp) is TrustTier.SAME_MODEL
    assert apply_gate(inp) is PostGateLabel.SUPPORTED_UNCERTIFIED


def test_cross_model_evidence_is_independent():
    inp = GateInput(
        claim_generator_id="chexagent",
        evidence_generator_id="medgemma",
        evidence_source_type="vlm_generated",
        pre_gate_label="supported",
    )
    assert compute_trust_tier(inp) is TrustTier.INDEPENDENT
    assert apply_gate(inp) is PostGateLabel.SUPPORTED_TRUSTED


def test_unknown_generator_is_downgraded():
    inp = GateInput(
        claim_generator_id=None,
        evidence_generator_id="chexagent",
        evidence_source_type="vlm_generated",
        pre_gate_label="supported",
    )
    assert compute_trust_tier(inp) is TrustTier.UNKNOWN
    assert apply_gate(inp) is PostGateLabel.SUPPORTED_UNCERTIFIED


def test_contradicted_always_stays_contradicted():
    inp = GateInput(
        claim_generator_id="chexagent",
        evidence_generator_id=None,
        evidence_source_type="oracle",
        pre_gate_label="contradicted",
    )
    assert apply_gate(inp) is PostGateLabel.CONTRADICTED


def test_downgrade_rate_matches_expectation():
    inputs = [
        # 2 same-model supported -> downgraded
        GateInput("a", "a", "vlm_generated", "supported"),
        GateInput("a", "a", "vlm_generated", "supported"),
        # 2 independent supported -> NOT downgraded
        GateInput("a", "b", "vlm_generated", "supported"),
        GateInput("a", "b", "vlm_generated", "supported"),
        # 1 contradicted (ignored by downgrade_rate)
        GateInput("a", "a", "vlm_generated", "contradicted"),
    ]
    # Of 4 supported, 2 downgraded -> rate = 0.5.
    assert downgrade_rate(inputs) == 0.5


def test_downgrade_rate_zero_when_no_supported():
    assert downgrade_rate([GateInput("a", "a", "vlm_generated", "contradicted")]) == 0.0
