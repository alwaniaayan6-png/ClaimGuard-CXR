"""Provenance-tier gate for claim certification.

Port of the v1 provenance module to Path B, aligned with the Path B
Claim schema (which carries ``source_model_id``) and the new naming
convention.

A claim is post-gate certifiable only when the evidence it was scored
against came from a *different* generator than the claim itself (or from
a human). The canonical tiers, in decreasing trust:

    TRUSTED       — oracle, human-authored reference report
    INDEPENDENT   — retrieved human-authored passage, or generator-distinct VLM
    SAME_MODEL    — evidence came from the same model that wrote the claim
    UNKNOWN       — provenance unavailable

Same-model + unknown are always downgraded. See §3.7 of MANUSCRIPT_MINI
for the empirical motivation (same-model dual-run downgrade rate = 1.00).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrustTier(str, Enum):
    TRUSTED = "trusted"
    INDEPENDENT = "independent"
    SAME_MODEL = "same_model"
    UNKNOWN = "unknown"


class PostGateLabel(str, Enum):
    SUPPORTED_TRUSTED = "supported_trusted"
    SUPPORTED_UNCERTIFIED = "supported_uncertified"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class GateInput:
    claim_generator_id: Optional[str]
    evidence_generator_id: Optional[str]
    evidence_source_type: str   # 'oracle' | 'retrieval' | 'vlm_generated'
    pre_gate_label: str         # 'supported' | 'contradicted'


def compute_trust_tier(inp: GateInput) -> TrustTier:
    """Decide the evidence trust tier for a scored (claim, evidence) pair."""
    if inp.evidence_source_type == "oracle":
        return TrustTier.TRUSTED

    if inp.evidence_source_type == "retrieval":
        # Retrieved human-authored passages are independent by construction.
        return TrustTier.INDEPENDENT

    if inp.evidence_source_type == "vlm_generated":
        if inp.claim_generator_id is None or inp.evidence_generator_id is None:
            return TrustTier.UNKNOWN
        if inp.claim_generator_id == inp.evidence_generator_id:
            return TrustTier.SAME_MODEL
        return TrustTier.INDEPENDENT

    return TrustTier.UNKNOWN


def apply_gate(inp: GateInput) -> PostGateLabel:
    """Apply the provenance gate. Contradicted always stays contradicted."""
    if inp.pre_gate_label == "contradicted":
        return PostGateLabel.CONTRADICTED
    tier = compute_trust_tier(inp)
    if tier in (TrustTier.TRUSTED, TrustTier.INDEPENDENT):
        return PostGateLabel.SUPPORTED_TRUSTED
    return PostGateLabel.SUPPORTED_UNCERTIFIED


def downgrade_rate(inputs) -> float:
    """Fraction of 'supported' pre-gate labels that got downgraded post-gate."""
    n_supp = 0
    n_down = 0
    for inp in inputs:
        if inp.pre_gate_label != "supported":
            continue
        n_supp += 1
        if apply_gate(inp) == PostGateLabel.SUPPORTED_UNCERTIFIED:
            n_down += 1
    if n_supp == 0:
        return 0.0
    return n_down / n_supp
