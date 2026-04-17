"""Unit tests for the baselines registry."""
from __future__ import annotations

import json

from claimguard_nmi.baselines import (
    LLMJudgeBaseline,
    MajorityClassBaseline,
    RadFlagBaseline,
    RuleBasedNegationBaseline,
    get_baseline,
    list_baselines,
)
from claimguard_nmi.extraction import StubBackend
from claimguard_nmi.grounding.claim_schema import (
    Claim,
    ClaimCertainty,
    ClaimType,
    Laterality,
    Region,
)


def _supp_claim() -> Claim:
    return Claim(
        raw_text="no pleural effusion",
        finding="pleural_effusion",
        claim_type=ClaimType.NEGATION,
        certainty=ClaimCertainty.ABSENT,
        laterality=Laterality.UNSPECIFIED,
        region=Region.UNSPECIFIED,
    )


def _present_claim() -> Claim:
    return Claim(
        raw_text="small left pleural effusion",
        finding="pleural_effusion",
        claim_type=ClaimType.FINDING,
        certainty=ClaimCertainty.PRESENT,
        laterality=Laterality.LEFT,
        region=Region.LOWER,
    )


def test_registry_enumerates_all_expected():
    assert set(list_baselines()) >= {
        "majority_class", "rule_based_negation", "llm_judge",
        "vlm_judge", "radflag",
    }


def test_get_baseline_instantiates_by_name():
    b = get_baseline("majority_class")
    assert isinstance(b, MajorityClassBaseline)


def test_get_baseline_unknown_raises():
    try:
        get_baseline("not_a_real_baseline")
        assert False, "should have raised"
    except KeyError:
        pass


def test_majority_class_always_predicts_supported():
    b = MajorityClassBaseline()
    s = b.score_claim(_supp_claim(), "any evidence")
    assert s.pred_label == 0
    assert s.supported_prob == 1.0


def test_rule_based_detects_negation_contradiction():
    b = RuleBasedNegationBaseline()
    # Claim asserts pleural effusion is present; evidence says "no pleural effusion"
    s = b.score_claim(_present_claim(), "There is no pleural effusion.")
    assert s.pred_label == 1


def test_rule_based_consistent_when_no_negation_cue():
    b = RuleBasedNegationBaseline()
    s = b.score_claim(_present_claim(), "There is a small pleural effusion.")
    assert s.pred_label == 0


def test_llm_judge_parses_well_formed_response():
    canned = json.dumps({"contradicted_prob": 0.85, "rationale": "negation flip"})
    b = LLMJudgeBaseline(backend=StubBackend(canned=canned))
    s = b.score_claim(_present_claim(), "No pleural effusion")
    assert abs(s.contradicted_prob - 0.85) < 1e-6
    assert s.pred_label == 1


def test_llm_judge_falls_back_on_malformed_response():
    b = LLMJudgeBaseline(backend=StubBackend(canned="not json"))
    s = b.score_claim(_present_claim(), "anything")
    assert s.contradicted_prob == 0.5


def test_radflag_flags_missing_finding_when_claim_present():
    b = RadFlagBaseline(consistency_threshold=0.5)
    # Claim asserts pleural effusion is present. K samples never mention it.
    s = b.score_claim(
        _present_claim(),
        evidence_text="",
        k_samples=["report A with no mention", "report B with no mention"],
    )
    assert s.contradicted_prob == 1.0


def test_radflag_supports_finding_when_consistent():
    b = RadFlagBaseline()
    s = b.score_claim(
        _present_claim(),
        evidence_text="",
        k_samples=[
            "There is pleural effusion.",
            "Pleural effusion noted.",
            "Pleural effusion is present.",
        ],
    )
    assert s.contradicted_prob < 0.5
