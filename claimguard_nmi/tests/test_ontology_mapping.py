"""Unit tests for canonical-finding alias resolution."""
from __future__ import annotations

from claimguard_nmi.data import load_ontology


def test_english_aliases_canonicalize():
    m = load_ontology()
    assert m.canonicalize("lung opacity") == "lung_opacity"
    assert m.canonicalize("Pulmonary Opacity") == "lung_opacity"
    assert m.canonicalize("consolidation") == "lung_opacity"


def test_spanish_alias_canonicalizes():
    m = load_ontology()
    assert m.canonicalize("patron alveolar pulmonar") == "lung_opacity"
    assert m.canonicalize("derrame pleural") == "pleural_effusion"


def test_unknown_string_returns_none():
    m = load_ontology()
    assert m.canonicalize("unicorn lesion") is None


def test_diffuse_flag():
    m = load_ontology()
    assert m.is_diffuse("cardiomegaly") is True
    assert m.is_diffuse("nodule") is False


def test_ungroundable_claim_type_flag():
    m = load_ontology()
    assert m.is_ungroundable_claim_type("prior_comparison") is True
    assert m.is_ungroundable_claim_type("finding") is False
