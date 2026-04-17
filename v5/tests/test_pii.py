"""Tests for PII scrubber regex layer (Presidio layer tested separately)."""

from __future__ import annotations

from v5.data.pii_scrubber import PIIScrubber


def test_scrubs_date():
    s = PIIScrubber()
    out = s.scrub("Comparison is made to prior exam on 03/14/2023.")
    assert "03/14/2023" not in out.text
    assert "[DATE]" in out.text
    assert out.n_redactions >= 1


def test_scrubs_mrn_and_accession():
    s = PIIScrubber()
    out = s.scrub("MRN: 1234567, Accession # 987654321")
    assert "[MRN]" in out.text
    assert "[ACCESSION]" in out.text


def test_scrubs_phone_and_email():
    s = PIIScrubber()
    out = s.scrub("Contact Dr. Smith at 555-123-4567 or smith@hospital.edu.")
    assert "[PHONE]" in out.text
    assert "[EMAIL]" in out.text
    assert "[PROVIDER]" in out.text


def test_scrubs_institution_name():
    s = PIIScrubber()
    out = s.scrub("Imaging performed at Stanford hospital.")
    assert "Stanford" not in out.text
    assert "[INSTITUTION]" in out.text
