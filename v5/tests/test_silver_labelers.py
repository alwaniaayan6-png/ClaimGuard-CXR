"""Unit tests for green/radfact/vert silver labelers.

LLM calls are mocked. The goal is to verify:

* ``GreenLabeler._parse_response`` correctly extracts score + error categories
  from varied GREEN model output shapes.
* ``RadFactLabeler.verify`` correctly parses Opus ENTAILED/CONTRADICTED/RATIONALE
  responses and handles minor whitespace/capitalization noise.
* ``RadFactLabeler.decompose`` correctly splits Haiku output into atomic phrases
  with bullet, numbered, and mixed prefixes.
* ``VertLabeler.label_claim`` correctly parses VERDICT/SEVERITY/RATIONALE output.
* Decision rules map parsed fields -> SUPPORTED/CONTRADICTED/UNCERTAIN correctly.

No network access; no model loading. The tests instantiate the labelers and
substitute the ``_call`` method (for RadFact/VERT) or the parsing helper
directly (for GREEN, which runs locally on GPU and therefore can't be fully
instantiated here).
"""

from __future__ import annotations

import pytest

from v5.eval.green_labeler import GreenLabeler, GreenLabel, _SIGNIFICANT_CATEGORIES, _NONSIG_CATEGORIES
from v5.eval.radfact_labeler import RadFactLabeler, RadFactLabel
from v5.eval.vert_labeler import VertLabeler, VertLabel


# ---------------------------------------------------------------------------
# GREEN — test the parser without loading the 7B model
# ---------------------------------------------------------------------------


class TestGreenParseResponse:
    def _labeler(self) -> GreenLabeler:
        obj = GreenLabeler.__new__(GreenLabeler)
        return obj

    def test_clean_response_all_zero(self):
        resp = (
            "1. false finding: 0 errors\n"
            "2. missing finding: 0 errors\n"
            "3. anatomic location error: 0 errors\n"
            "4. severity error: 0 errors\n"
            "5. extraneous comparison: 0 errors\n"
            "6. omitted prior comparison: 0 errors\n"
            "GREEN score: 0.95"
        )
        score, n_sig, cats = self._labeler()._parse_response(resp)
        assert score == pytest.approx(0.95)
        assert n_sig == 0
        assert cats == []

    def test_false_finding_flags_sig(self):
        resp = (
            "1. false finding: 2 errors\n"
            "2. missing finding: 0\n3. anatomic location error: 0\n"
            "4. severity error: 0\n5. extraneous comparison: 0\n"
            "6. omitted prior comparison: 0\n"
            "GREEN score = 0.40"
        )
        score, n_sig, cats = self._labeler()._parse_response(resp)
        assert score == pytest.approx(0.40)
        assert n_sig == 2
        assert "false_finding" in cats

    def test_missing_finding_is_nonsig_per_claim_mode(self):
        """Per pre-flight B2 decision: missing_finding cannot count as significant
        in per-claim mode because every single-sentence candidate omits the rest
        of the reference."""
        assert "missing_finding" in _NONSIG_CATEGORIES
        assert "missing_finding" not in _SIGNIFICANT_CATEGORIES
        resp = (
            "1. false finding: 0\n2. missing finding: 5\n"
            "3. anatomic location error: 0\n4. severity error: 0\n"
            "5. extraneous comparison: 0\n6. omitted prior comparison: 0\n"
            "score: 0.5"
        )
        score, n_sig, cats = self._labeler()._parse_response(resp)
        assert n_sig == 0          # missing_finding does not inflate n_sig
        assert "missing_finding" in cats

    def test_severity_and_anatomic_both_counted(self):
        resp = (
            "1. false finding: 0\n2. missing finding: 0\n"
            "3. anatomic location error: 1\n4. severity error: 2\n"
            "5. extraneous comparison: 0\n6. omitted prior comparison: 0\n"
            "Score: 0.3"
        )
        score, n_sig, cats = self._labeler()._parse_response(resp)
        assert n_sig == 3
        assert "anatomic_location_error" in cats
        assert "severity_error" in cats

    def test_score_out_of_range_clamped(self):
        score, _, _ = self._labeler()._parse_response("GREEN score: 1.7")
        assert score == 1.0
        score, _, _ = self._labeler()._parse_response("GREEN score: -0.1")
        assert score == 0.0

    def test_no_score_defaults(self):
        score, n_sig, cats = self._labeler()._parse_response("completely unparseable output")
        assert score == 0.5
        assert n_sig == 0
        assert cats == []


# ---------------------------------------------------------------------------
# RadFact — mock the Anthropic call
# ---------------------------------------------------------------------------


class TestRadFactDecompose:
    def test_bullet_list(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: (
            "- cardiomegaly present\n"
            "- no pleural effusion\n"
            "- mild atelectasis in left lower lobe\n"
        )
        phrases = labeler.decompose("Some report text.")
        assert phrases == [
            "cardiomegaly present",
            "no pleural effusion",
            "mild atelectasis in left lower lobe",
        ]

    def test_numbered_list(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: (
            "1. cardiomegaly present\n"
            "2. no pleural effusion\n"
            "3. mild atelectasis\n"
        )
        phrases = labeler.decompose("Some report.")
        assert phrases == ["cardiomegaly present", "no pleural effusion", "mild atelectasis"]

    def test_mixed_bullets_and_stars(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: "* phrase one\n- phrase two\n"
        phrases = labeler.decompose("text")
        assert phrases == ["phrase one", "phrase two"]

    def test_empty_input_returns_empty(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: "should not be called"
        assert labeler.decompose("") == []
        assert labeler.decompose("   ") == []


class TestRadFactVerify:
    def test_entailed_not_contradicted_supported(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: (
            "ENTAILED: yes\nCONTRADICTED: no\nRATIONALE: reference directly asserts the claim"
        )
        entailed, contradicted, rationale = labeler.verify(
            reference_phrases=["cardiomegaly present"],
            candidate_claim="there is cardiomegaly",
        )
        assert entailed is True
        assert contradicted is False
        assert "directly asserts" in rationale

    def test_contradicted_wins_over_entailed(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: (
            "ENTAILED: no\nCONTRADICTED: yes\nRATIONALE: reference says absent"
        )
        labeled = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="cardiomegaly present",
            reference_phrases=["no cardiomegaly"],
        )
        assert labeled.label == "CONTRADICTED"

    def test_neither_is_uncertain(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: (
            "ENTAILED: no\nCONTRADICTED: no\nRATIONALE: insufficient info"
        )
        labeled = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="incidental finding of unclear significance",
            reference_phrases=["unremarkable"],
        )
        assert labeled.label == "UNCERTAIN"

    def test_case_insensitive_parsing(self):
        labeler = RadFactLabeler()
        labeler._call = lambda *a, **kw: "entailed: YES\ncontradicted: NO\nrationale: ok"
        entailed, contradicted, _ = labeler.verify(
            reference_phrases=["x"], candidate_claim="x",
        )
        assert entailed is True
        assert contradicted is False

    def test_missing_reference_raises(self):
        labeler = RadFactLabeler()
        with pytest.raises(ValueError):
            labeler.label_claim(
                claim_id="c1", image_id="i1", rrg_model="maira",
                claim_text="x",
            )


# ---------------------------------------------------------------------------
# VERT — mock the Anthropic call
# ---------------------------------------------------------------------------


class TestVertLabelClaim:
    def test_supported_minor(self):
        labeler = VertLabeler()
        labeler._call = lambda *a, **kw: (
            "VERDICT: SUPPORTED\nSEVERITY: minor\nRATIONALE: consistent with reference"
        )
        label = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="cardiomegaly", reference_report="mild cardiomegaly present",
        )
        assert label.label == "SUPPORTED"
        assert label.severity == "minor"
        assert "consistent" in label.rationale

    def test_contradicted_major(self):
        labeler = VertLabeler()
        labeler._call = lambda *a, **kw: (
            "VERDICT: CONTRADICTED\nSEVERITY: major\nRATIONALE: opposite laterality"
        )
        label = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="right pleural effusion", reference_report="left pleural effusion",
        )
        assert label.label == "CONTRADICTED"
        assert label.severity == "major"

    def test_uncertain_none(self):
        labeler = VertLabeler()
        labeler._call = lambda *a, **kw: (
            "VERDICT: UNCERTAIN\nSEVERITY: none\nRATIONALE: ambiguous"
        )
        label = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="borderline finding", reference_report="reference unclear",
        )
        assert label.label == "UNCERTAIN"
        assert label.severity == "none"

    def test_malformed_response_defaults_to_uncertain(self):
        labeler = VertLabeler()
        labeler._call = lambda *a, **kw: "completely unstructured response"
        label = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="x", reference_report="y",
        )
        assert label.label == "UNCERTAIN"
        assert label.severity == "none"

    def test_case_insensitive_verdict(self):
        labeler = VertLabeler()
        labeler._call = lambda *a, **kw: "verdict: supported\nseverity: minor\nrationale: ok"
        label = labeler.label_claim(
            claim_id="c1", image_id="i1", rrg_model="maira",
            claim_text="x", reference_report="y",
        )
        assert label.label == "SUPPORTED"
