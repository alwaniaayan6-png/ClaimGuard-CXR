"""Evidence provenance metadata and trust-tier gating for ClaimGuard-CXR.

This module fixes a core limitation of the text-only verifier: the binary
RoBERTa-large cross-encoder scores claim-evidence consistency, but it cannot
tell whether the evidence is actually independent from the claim. If the same
model that generated the claim also generated the "evidence," a high
consistency score is meaningless — the model just agrees with itself.

The fix is a pipeline-level policy, not a model change. Every claim-evidence
example now carries structured provenance metadata, and the final triage
step refuses to certify any claim whose evidence is same-model-generated or
whose provenance is unknown. The binary verifier is not retrained. Only the
trust certification step becomes provenance-aware.

Contract (every example in the eval / training / demo pipelines):

    {
        ...claim/evidence/label/pathology/patient_id fields...,
        "evidence_source_type": str,   # see EvidenceSourceType
        "evidence_is_independent": bool,
        "evidence_generator_id": Optional[str],
        "claim_generator_id": Optional[str],
        "evidence_trust_tier": str,    # see TrustTier
    }

Policy (see `apply_provenance_gate`):

    trusted           -> normal verifier + conformal path (can be certified safe)
    independent       -> normal verifier + conformal path (can be certified safe,
                         annotated as independent-retrieval)
    same_model        -> NEVER certified safe. Forced to review_required regardless
                         of verifier score or conformal acceptance.
    unknown           -> NEVER certified safe. Forced to review_required.

Final provenance-aware triage labels (five-tier, replacing the three-tier
green/yellow/red scheme for anything that needs to be displayed to a
clinician):

    supported_trusted       — verifier accepted + trusted/independent provenance
    supported_uncertified   — verifier accepted but provenance is same_model or unknown
    review_required         — verifier non-accepted but not clearly contradicted
    contradicted            — verifier non-accepted with low score
    provenance_blocked      — shown only when the provenance gate overrode an
                              otherwise-green decision (for audit/debugging)

The green/yellow/red labels in `inference/conformal_triage.py::assign_triage_labels`
are preserved unchanged for backward compatibility with existing analysis
scripts. The new labels are a post-processing layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class EvidenceSourceType:
    """Canonical strings for the `evidence_source_type` field.

    These are the only accepted values. Any other value is treated as UNKNOWN
    by `classify_trust_tier`.
    """

    ORACLE_REPORT_TEXT = "oracle_report_text"
    """Evidence pulled from the same radiologist-written ground-truth report
    the claim was extracted from. This is the standard oracle setting used
    to isolate verifier quality from retrieval quality. Fully trusted."""

    RETRIEVED_REPORT_TEXT = "retrieved_report_text"
    """Evidence retrieved from the training-patient corpus by MedCPT dense
    retrieval. Not from the same report as the claim. Independent as long as
    the training corpus is human-written."""

    GENERATOR_OUTPUT = "generator_output"
    """Evidence that was itself produced by an LLM / VLM. Potentially
    same-model with the claim. Never certified safe unless we can prove
    evidence_generator_id != claim_generator_id AND the generator is
    explicitly trusted."""

    UNKNOWN = "unknown"
    """Provenance was never annotated. Default for legacy data. Never
    certified safe."""

    ALL = (
        ORACLE_REPORT_TEXT,
        RETRIEVED_REPORT_TEXT,
        GENERATOR_OUTPUT,
        UNKNOWN,
    )


class TrustTier:
    """Canonical strings for the `evidence_trust_tier` field.

    Computed from `evidence_source_type` and the generator IDs by
    `classify_trust_tier`. Stored explicitly on each example so downstream
    code can filter without re-running the classification logic.
    """

    TRUSTED = "trusted"
    """Oracle, radiologist-written, same-report evidence. Fully independent
    of any generator."""

    INDEPENDENT = "independent"
    """Evidence that was not produced by the same model that produced the
    claim. Includes dense-retrieved human-written passages and
    cross-generator evidence (claim_generator_id != evidence_generator_id)."""

    SAME_MODEL = "same_model"
    """Evidence produced by the same generator as the claim. Certifying
    this as safe would be a self-consistency loop. Blocked by policy."""

    UNKNOWN = "unknown"
    """Provenance was never annotated or source_type was UNKNOWN. Cannot
    be certified safe."""

    ALL = (TRUSTED, INDEPENDENT, SAME_MODEL, UNKNOWN)

    CERTIFIABLE = frozenset({TRUSTED, INDEPENDENT})
    """The only two tiers from which a claim is allowed to be labeled
    supported_trusted."""


class ProvenanceTriageLabel:
    """Provenance-aware triage labels shown to downstream consumers.

    These replace the three-tier green/yellow/red output at the *certification*
    layer. The underlying green/yellow/red from the conformal procedure is
    preserved for analysis compatibility — these labels are a post-processing
    decision on top.
    """

    SUPPORTED_TRUSTED = "supported_trusted"
    SUPPORTED_UNCERTIFIED = "supported_uncertified"
    REVIEW_REQUIRED = "review_required"
    CONTRADICTED = "contradicted"
    PROVENANCE_BLOCKED = "provenance_blocked"

    ALL = (
        SUPPORTED_TRUSTED,
        SUPPORTED_UNCERTIFIED,
        REVIEW_REQUIRED,
        CONTRADICTED,
        PROVENANCE_BLOCKED,
    )


# ---------------------------------------------------------------------------
# Default provenance metadata
# ---------------------------------------------------------------------------

# The fields we expect every example to carry after this patch. Existing eval
# data that predates this patch will be backfilled with UNKNOWN defaults by
# `ensure_provenance_fields`, which is safe because UNKNOWN is the maximally
# restrictive tier and cannot be certified safe.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "evidence_source_type",
    "evidence_is_independent",
    "evidence_generator_id",
    "claim_generator_id",
    "evidence_trust_tier",
)


def default_provenance(
    source_type: str = EvidenceSourceType.UNKNOWN,
    claim_generator_id: Optional[str] = None,
    evidence_generator_id: Optional[str] = None,
) -> dict:
    """Build a provenance-metadata dict ready to merge into an example.

    The `evidence_is_independent` flag and `evidence_trust_tier` are derived
    from `source_type` and the two generator IDs using the rules below.
    Callers should NOT set these fields manually — always go through this
    function so the invariants are enforced in one place.
    """
    tier = classify_trust_tier(
        source_type=source_type,
        claim_generator_id=claim_generator_id,
        evidence_generator_id=evidence_generator_id,
    )
    is_independent = tier in (TrustTier.TRUSTED, TrustTier.INDEPENDENT)
    return {
        "evidence_source_type": source_type,
        "evidence_is_independent": bool(is_independent),
        "evidence_generator_id": evidence_generator_id,
        "claim_generator_id": claim_generator_id,
        "evidence_trust_tier": tier,
    }


def ensure_provenance_fields(example: dict) -> dict:
    """Return a copy of `example` with all provenance fields populated.

    Missing fields are filled with UNKNOWN defaults. This is the backward-
    compatibility path for eval / training data generated before this patch
    was added. Because UNKNOWN maps to TrustTier.UNKNOWN, which is not
    certifiable, stamping legacy examples this way cannot turn a previously-
    unsafe decision into a safe one. It only makes the legacy data usable
    without regeneration.

    The returned dict is a shallow copy with provenance fields overwritten,
    so the caller's original dict is not mutated.
    """
    out = dict(example)
    existing = {k: example.get(k) for k in PROVENANCE_FIELDS if k in example}

    # If the caller provided source_type + generator IDs, re-derive the tier
    # so the invariants hold. Otherwise stamp UNKNOWN defaults.
    if "evidence_source_type" in existing:
        source_type = existing["evidence_source_type"]
        claim_gen = existing.get("claim_generator_id")
        evidence_gen = existing.get("evidence_generator_id")
        prov = default_provenance(
            source_type=source_type,
            claim_generator_id=claim_gen,
            evidence_generator_id=evidence_gen,
        )
    else:
        prov = default_provenance()

    out.update(prov)
    return out


# ---------------------------------------------------------------------------
# Trust-tier classification
# ---------------------------------------------------------------------------


def classify_trust_tier(
    source_type: str,
    claim_generator_id: Optional[str] = None,
    evidence_generator_id: Optional[str] = None,
) -> str:
    """Return the TrustTier for a claim/evidence pair.

    Rules (checked in order):

      1. source_type == ORACLE_REPORT_TEXT -> TRUSTED
         Oracle text is radiologist-written ground truth. Always trusted.
      2. source_type == RETRIEVED_REPORT_TEXT -> INDEPENDENT
         Retrieved from a human-written corpus that excludes the claim's
         source report. Independent by construction.
      3. source_type == GENERATOR_OUTPUT -> depends on generator IDs:
         - If claim_generator_id and evidence_generator_id are both known
           and different, INDEPENDENT.
         - If they are equal, or either is None, SAME_MODEL (treated as
           same-model for safety).
      4. Anything else -> UNKNOWN.

    The precedence is deliberately conservative: if we cannot prove the
    evidence is independent, we refuse to call it independent.
    """
    if source_type == EvidenceSourceType.ORACLE_REPORT_TEXT:
        return TrustTier.TRUSTED

    if source_type == EvidenceSourceType.RETRIEVED_REPORT_TEXT:
        return TrustTier.INDEPENDENT

    if source_type == EvidenceSourceType.GENERATOR_OUTPUT:
        if claim_generator_id is None or evidence_generator_id is None:
            return TrustTier.SAME_MODEL
        if str(claim_generator_id) == str(evidence_generator_id):
            return TrustTier.SAME_MODEL
        return TrustTier.INDEPENDENT

    return TrustTier.UNKNOWN


def is_certifiable(tier: str) -> bool:
    """Return True if a claim with this trust tier can ever be supported_trusted."""
    return tier in TrustTier.CERTIFIABLE


# ---------------------------------------------------------------------------
# Gating (post-processing over the conformal triage output)
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceGateResult:
    """Output of the provenance gate for a single claim.

    - `final_label`: the provenance-aware label (ProvenanceTriageLabel).
    - `underlying_conformal_label`: the green/yellow/red label the conformal
      procedure assigned before the gate. Preserved for debugging and for
      analyses that want to compare gated vs ungated outcomes.
    - `trust_tier`: the trust tier that drove the decision.
    - `was_overridden`: True if the gate downgraded a green claim to
      supported_uncertified or provenance_blocked.
    """

    final_label: str
    underlying_conformal_label: str
    trust_tier: str
    was_overridden: bool


def apply_provenance_gate(
    conformal_label: Literal["green", "yellow", "red"],
    trust_tier: str,
) -> ProvenanceGateResult:
    """Map a (conformal_label, trust_tier) pair to a provenance-aware label.

    Rules:

        conformal    | tier              | final
        -------------|-------------------|----------------------------
        green        | trusted           | supported_trusted
        green        | independent       | supported_trusted
        green        | same_model        | supported_uncertified  (OVERRIDE)
        green        | unknown           | supported_uncertified  (OVERRIDE)
        yellow       | *                 | review_required
        red          | *                 | contradicted

    Note: a "green" claim with non-certifiable provenance is downgraded to
    `supported_uncertified`, not to `provenance_blocked`. The `provenance_blocked`
    label is reserved for code paths that want to surface the override
    explicitly in an audit log. The gate here uses the softer downgrade so
    downstream consumers can still see the verifier thought the claim was
    fine — it just can't be certified.
    """
    if conformal_label == "green":
        if is_certifiable(trust_tier):
            return ProvenanceGateResult(
                final_label=ProvenanceTriageLabel.SUPPORTED_TRUSTED,
                underlying_conformal_label=conformal_label,
                trust_tier=trust_tier,
                was_overridden=False,
            )
        # Non-certifiable provenance -> override
        return ProvenanceGateResult(
            final_label=ProvenanceTriageLabel.SUPPORTED_UNCERTIFIED,
            underlying_conformal_label=conformal_label,
            trust_tier=trust_tier,
            was_overridden=True,
        )

    if conformal_label == "yellow":
        return ProvenanceGateResult(
            final_label=ProvenanceTriageLabel.REVIEW_REQUIRED,
            underlying_conformal_label=conformal_label,
            trust_tier=trust_tier,
            was_overridden=False,
        )

    # Everything else (including "red" and any unrecognized value) -> contradicted.
    return ProvenanceGateResult(
        final_label=ProvenanceTriageLabel.CONTRADICTED,
        underlying_conformal_label=conformal_label,
        trust_tier=trust_tier,
        was_overridden=False,
    )


def apply_provenance_gate_batch(
    conformal_labels: Iterable[str],
    trust_tiers: Iterable[str],
) -> list[ProvenanceGateResult]:
    """Vectorised convenience: apply `apply_provenance_gate` to each pair."""
    pairs = list(zip(conformal_labels, trust_tiers))
    return [apply_provenance_gate(cl, tt) for cl, tt in pairs]


# ---------------------------------------------------------------------------
# Summary helper for evaluation / reporting
# ---------------------------------------------------------------------------


def summarize_by_trust_tier(examples: list[dict]) -> dict[str, int]:
    """Count examples grouped by their `evidence_trust_tier` field.

    Useful for provenance-aware evaluation reports: we want to see what
    fraction of a test set is trusted, independent, same_model, or unknown.
    Examples without a tier are counted as UNKNOWN.
    """
    counts: dict[str, int] = {t: 0 for t in TrustTier.ALL}
    for ex in examples:
        tier = ex.get("evidence_trust_tier", TrustTier.UNKNOWN)
        if tier not in counts:
            tier = TrustTier.UNKNOWN
        counts[tier] += 1
    return counts
