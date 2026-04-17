"""Unit tests for the claim extractor."""
from __future__ import annotations

import json

from claimguard_nmi.extraction import ClaimExtractor, StubBackend
from claimguard_nmi.grounding.claim_schema import (
    ClaimCertainty,
    ClaimType,
    Laterality,
    Region,
)


def test_extractor_parses_well_formed_json():
    canned = json.dumps([
        {
            "raw_text": "small left pleural effusion",
            "finding": "pleural_effusion",
            "claim_type": "finding",
            "certainty": "present",
            "laterality": "left",
            "region": "lower",
            "severity": "small",
        },
        {
            "raw_text": "no pneumothorax",
            "finding": "pneumothorax",
            "claim_type": "negation",
            "certainty": "absent",
        },
    ])
    ext = ClaimExtractor(backend=StubBackend(canned=canned))
    claims = ext.extract("dummy report")
    assert len(claims) == 2
    first = claims[0]
    assert first.finding == "pleural_effusion"
    assert first.claim_type is ClaimType.FINDING
    assert first.laterality is Laterality.LEFT
    assert first.region is Region.LOWER
    assert first.severity == "small"

    second = claims[1]
    assert second.certainty is ClaimCertainty.ABSENT
    assert second.claim_type is ClaimType.NEGATION


def test_extractor_strips_code_fences():
    fenced = "```json\n" + json.dumps([
        {"raw_text": "x", "finding": "nodule"},
    ]) + "\n```"
    ext = ClaimExtractor(backend=StubBackend(canned=fenced))
    claims = ext.extract("dummy")
    assert len(claims) == 1
    assert claims[0].finding == "nodule"


def test_extractor_returns_empty_on_bad_json():
    ext = ClaimExtractor(backend=StubBackend(canned="not json at all"))
    assert ext.extract("dummy") == []


def test_extractor_unknown_enum_falls_back_to_default():
    canned = json.dumps([
        {
            "raw_text": "x", "finding": "nodule",
            "claim_type": "banana",   # not a valid enum
            "certainty": "confusion", # not valid either
            "laterality": "nowhere",
        }
    ])
    ext = ClaimExtractor(backend=StubBackend(canned=canned))
    claims = ext.extract("dummy")
    assert len(claims) == 1
    assert claims[0].claim_type is ClaimType.FINDING
    assert claims[0].certainty is ClaimCertainty.PRESENT
    assert claims[0].laterality is Laterality.UNSPECIFIED


def test_extractor_handles_claims_object_wrapper():
    canned = json.dumps({"claims": [{"raw_text": "x", "finding": "mass"}]})
    ext = ClaimExtractor(backend=StubBackend(canned=canned))
    claims = ext.extract("dummy")
    assert len(claims) == 1
    assert claims[0].finding == "mass"
