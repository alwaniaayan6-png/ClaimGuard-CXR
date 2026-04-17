"""Tests for the rule-based claim parser. Deterministic, no API calls."""

from __future__ import annotations

import pytest

from v5.data.claim_parser import rule_parse, load_ontology


@pytest.fixture(scope="module")
def ontology():
    return load_ontology()


def test_detects_left_pleural_effusion(ontology):
    claim = rule_parse(
        "There is a small left pleural effusion.",
        claim_id="c1", report_id="r1", ontology=ontology,
    )
    assert claim.finding == "pleural_effusion"
    assert claim.laterality == "L"
    assert claim.severity == "mild"
    assert "size_small" in claim.modifier_tags


def test_detects_pneumothorax(ontology):
    claim = rule_parse(
        "Large right pneumothorax is present.",
        claim_id="c2", report_id="r2", ontology=ontology,
    )
    assert claim.finding == "pneumothorax"
    assert claim.laterality == "R"
    assert claim.severity == "severe"


def test_negation_maps_to_no_finding(ontology):
    claim = rule_parse(
        "No pneumothorax.",
        claim_id="c3", report_id="r3", ontology=ontology,
    )
    assert "polarity_negated" in claim.modifier_tags
    # rule-parser rewrites unknown negations to no_finding;
    # when the finding is explicitly present, keep it and tag as negated
    assert claim.finding in {"pneumothorax", "no_finding"}


def test_comparison_detection(ontology):
    claim = rule_parse(
        "Increased opacity compared to the prior exam.",
        claim_id="c4", report_id="r4", ontology=ontology,
    )
    assert claim.comparison == "present"


def test_support_device(ontology):
    claim = rule_parse(
        "Endotracheal tube in good position.",
        claim_id="c5", report_id="r5", ontology=ontology,
    )
    assert claim.finding == "support_device"


def test_lobe_locations(ontology):
    for text, exp_loc in [
        ("Left upper lobe consolidation.", "left_upper_lobe"),
        ("Right lower lobe atelectasis.", "right_lower_lobe"),
        ("Right middle lobe opacity.", "right_middle_lobe"),
    ]:
        c = rule_parse(text, claim_id="x", report_id="x", ontology=ontology)
        assert c.location == exp_loc, f"{text} -> {c.location}"
