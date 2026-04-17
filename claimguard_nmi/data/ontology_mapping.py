"""Canonical-ontology mapping for cross-dataset claim grounding.

Every dataset uses a different label schema:
  * PadChest-GR uses Spanish medical terminology.
  * ChestX-Det10 uses 10 disease classes.
  * RSNA uses one class (opacity).
  * Object-CXR uses device / foreign-object categories.

We harmonize all of them into the 13-ish canonical findings defined in
``configs/ontology.yaml``. A claim whose finding string does not map to
any canonical id is marked UNGROUNDED downstream.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONTOLOGY_YAML = _REPO_ROOT / "configs" / "ontology.yaml"


class OntologyMapper:
    """Map surface-form strings to canonical finding ids.

    Matching is case-insensitive, whitespace-normalized, and punctuation-stripped.
    Multi-word aliases take precedence over single-word fallbacks.
    """

    def __init__(self, ontology: dict):
        self._ontology = ontology
        self._alias_to_canonical: Dict[str, str] = {}
        self._ungroundable_types = set(ontology.get("ungroundable_claim_types", []))
        self._diffuse = set(ontology.get("diffuse_findings_override", []))
        self._build_alias_index()

    def _build_alias_index(self) -> None:
        for canonical_id, meta in self._ontology.get("canonical_findings", {}).items():
            for name in meta.get("common_names", []):
                key = self._normalize(name)
                if key in self._alias_to_canonical:
                    existing = self._alias_to_canonical[key]
                    if existing != canonical_id:
                        raise ValueError(
                            f"Ontology alias collision: '{name}' maps to both "
                            f"'{existing}' and '{canonical_id}'"
                        )
                self._alias_to_canonical[key] = canonical_id

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def canonicalize(self, surface_form: str) -> Optional[str]:
        """Return the canonical finding id for a surface string, or None if unmapped."""
        if not surface_form:
            return None
        key = self._normalize(surface_form)
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]
        # Fallback: try each multi-word alias as a substring match.
        for alias_key, canonical in self._alias_to_canonical.items():
            if len(alias_key.split()) >= 2 and alias_key in key:
                return canonical
        return None

    def is_diffuse(self, canonical_id: str) -> bool:
        return canonical_id in self._diffuse

    def is_ungroundable_claim_type(self, claim_type: str) -> bool:
        return claim_type in self._ungroundable_types

    def canonical_findings(self) -> Iterable[str]:
        return self._ontology.get("canonical_findings", {}).keys()


def load_ontology(path: Path = _ONTOLOGY_YAML) -> OntologyMapper:
    with open(path, "r") as fh:
        ontology = yaml.safe_load(fh)
    return OntologyMapper(ontology)
