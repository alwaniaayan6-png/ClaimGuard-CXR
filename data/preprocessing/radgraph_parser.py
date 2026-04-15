"""RadGraph-XL parser for ClaimGuard-CXR.

Parses RadGraph entity-relation annotations into atomic claims and provides
utilities for hard negative construction and pathology ontology mapping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# CheXpert 14 + No Finding + Support Devices + Rare/Other = 17 categories
CHEXPERT_ONTOLOGY = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "Rare/Other",
]

# Mapping from RadGraph entity labels to CheXpert ontology
_ENTITY_TO_CHEXPERT = {
    # Direct mappings
    "cardiomegaly": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "edema": "Edema",
    "pulmonary edema": "Edema",
    "consolidation": "Consolidation",
    "pneumonia": "Pneumonia",
    "atelectasis": "Atelectasis",
    "pneumothorax": "Pneumothorax",
    "pleural effusion": "Pleural Effusion",
    "effusion": "Pleural Effusion",
    "fracture": "Fracture",
    "rib fracture": "Fracture",
    "tube": "Support Devices",
    "catheter": "Support Devices",
    "line": "Support Devices",
    "pacemaker": "Support Devices",
    "device": "Support Devices",
    "drain": "Support Devices",
    "ett": "Support Devices",
    "endotracheal tube": "Support Devices",
    "picc": "Support Devices",
    "port": "Support Devices",
    "opacity": "Lung Opacity",
    "infiltrate": "Lung Opacity",
    "haziness": "Lung Opacity",
    "mass": "Lung Lesion",
    "nodule": "Lung Lesion",
    "lesion": "Lung Lesion",
    "tumor": "Lung Lesion",
    "widened mediastinum": "Enlarged Cardiomediastinum",
    "mediastinal widening": "Enlarged Cardiomediastinum",
    "pleural thickening": "Pleural Other",
    "pleural abnormality": "Pleural Other",
    "normal": "No Finding",
    "clear lungs": "No Finding",
    "no acute": "No Finding",
    "unremarkable": "No Finding",
}


@dataclass
class Entity:
    """A single entity from RadGraph annotation."""
    tokens: str
    label: str  # 'ANAT-DP', 'OBS-DP', 'OBS-U', 'OBS-DA'
    start_ix: int
    end_ix: int
    entity_id: str = ""

    @property
    def is_observation(self) -> bool:
        return self.label.startswith("OBS")

    @property
    def is_anatomy(self) -> bool:
        return self.label.startswith("ANAT")

    @property
    def is_present(self) -> bool:
        return self.label == "OBS-DP"

    @property
    def is_absent(self) -> bool:
        return self.label == "OBS-DA"

    @property
    def is_uncertain(self) -> bool:
        return self.label == "OBS-U"


@dataclass
class Relation:
    """A relation between two entities."""
    source_id: str
    target_id: str
    relation_type: str  # 'located_at', 'suggestive_of', 'modify'


@dataclass
class Claim:
    """An atomic verifiable claim extracted from a report."""
    text: str
    pathology_category: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    is_negated: bool = False
    laterality: Optional[str] = None  # 'left', 'right', 'bilateral', None
    severity: Optional[str] = None  # 'mild', 'moderate', 'severe', None
    anatomy: Optional[str] = None


def map_to_chexpert_ontology(entity_text: str) -> str:
    """Map a RadGraph entity label to a CheXpert ontology category.

    Args:
        entity_text: The entity text from RadGraph.

    Returns:
        CheXpert category string.
    """
    text_lower = entity_text.lower().strip()

    # Direct lookup
    if text_lower in _ENTITY_TO_CHEXPERT:
        return _ENTITY_TO_CHEXPERT[text_lower]

    # Substring matching (only check if key appears in text, not reverse)
    for key, category in _ENTITY_TO_CHEXPERT.items():
        if key in text_lower:
            return category

    return "Rare/Other"


def load_radgraph(json_path: str | Path) -> dict:
    """Load RadGraph or RadGraph-XL annotations from JSON.

    Args:
        json_path: Path to the RadGraph JSON file.

    Returns:
        Dict mapping report_id -> annotation data.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"RadGraph file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded RadGraph annotations for {len(data)} reports from {json_path}")
    return data


def extract_entities(report_annotation: dict) -> list[Entity]:
    """Extract entities from a single report's RadGraph annotation.

    Args:
        report_annotation: The annotation dict for one report.

    Returns:
        List of Entity objects.
    """
    entities = []
    entity_data = report_annotation.get("entities", {})

    for eid, edata in entity_data.items():
        entity = Entity(
            tokens=edata.get("tokens", ""),
            label=edata.get("label", ""),
            start_ix=edata.get("start_ix", 0),
            end_ix=edata.get("end_ix", 0),
            entity_id=str(eid),
        )
        entities.append(entity)

    return entities


def extract_relations(report_annotation: dict) -> list[Relation]:
    """Extract relations from a single report's RadGraph annotation.

    Args:
        report_annotation: The annotation dict for one report.

    Returns:
        List of Relation objects.
    """
    relations = []
    entity_data = report_annotation.get("entities", {})

    for eid, edata in entity_data.items():
        for rel in edata.get("relations", []):
            relation = Relation(
                source_id=str(eid),
                target_id=str(rel[1]) if len(rel) > 1 else "",
                relation_type=str(rel[0]) if len(rel) > 0 else "",
            )
            relations.append(relation)

    return relations


def _detect_laterality(text: str) -> Optional[str]:
    """Detect laterality from claim text."""
    text_lower = text.lower()
    has_left = any(w in text_lower for w in ["left", "left-sided", "lll", "lul"])
    has_right = any(w in text_lower for w in ["right", "right-sided", "rll", "rul"])

    if has_left and has_right:
        return "bilateral"
    elif has_left:
        return "left"
    elif has_right:
        return "right"
    return None


def _detect_severity(text: str) -> Optional[str]:
    """Detect severity from claim text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["severe", "large", "massive", "extensive"]):
        return "severe"
    elif any(w in text_lower for w in ["moderate", "medium"]):
        return "moderate"
    elif any(w in text_lower for w in ["mild", "small", "minimal", "trace", "tiny"]):
        return "mild"
    return None


def entities_to_claims(
    entities: list[Entity],
    relations: list[Relation],
    report_text: str = "",
) -> list[Claim]:
    """Convert RadGraph entities and relations into atomic claims.

    Each observation entity becomes a claim. Anatomy entities are attached
    via 'located_at' relations.

    Args:
        entities: List of extracted entities.
        relations: List of extracted relations.
        report_text: Original report text (for context).

    Returns:
        List of Claim objects.
    """
    entity_map = {e.entity_id: e for e in entities}
    claims = []

    for entity in entities:
        if not entity.is_observation:
            continue

        # Find related anatomy (collect all locations)
        anatomy_parts = []
        for rel in relations:
            if rel.source_id == entity.entity_id and rel.relation_type == "located_at":
                target = entity_map.get(rel.target_id)
                if target and target.is_anatomy:
                    anatomy_parts.append(target.tokens)
        anatomy = " and ".join(anatomy_parts) if anatomy_parts else None

        # Build claim text
        claim_parts = []
        if entity.is_absent:
            claim_parts.append("No")
        claim_parts.append(entity.tokens)
        if anatomy:
            claim_parts.append(f"in the {anatomy}")

        claim_text = " ".join(claim_parts)
        category = map_to_chexpert_ontology(entity.tokens)

        claim = Claim(
            text=claim_text,
            pathology_category=category,
            entities=[entity],
            is_negated=entity.is_absent,
            laterality=_detect_laterality(claim_text),
            severity=_detect_severity(claim_text),
            anatomy=anatomy,
        )
        claims.append(claim)

    return claims


def get_swappable_fields(claim: Claim) -> dict:
    """Get fields that can be swapped for hard negative construction.

    Args:
        claim: A Claim object.

    Returns:
        Dict with keys indicating which hard negative types are applicable.
    """
    fields = {
        "can_swap_laterality": claim.laterality is not None,
        "can_negate": True,  # any claim can be negated
        "can_swap_severity": claim.severity is not None,
        "can_swap_region": claim.anatomy is not None,
        "can_swap_finding": claim.pathology_category != "No Finding",
        "current_laterality": claim.laterality,
        "current_severity": claim.severity,
        "current_anatomy": claim.anatomy,
        "current_finding": claim.pathology_category,
        "is_negated": claim.is_negated,
    }
    return fields


def parse_all_reports(
    radgraph_data: dict,
) -> pd.DataFrame:
    """Parse all RadGraph annotations into a claims DataFrame.

    Args:
        radgraph_data: Full RadGraph JSON data.

    Returns:
        DataFrame with columns: report_id, claim_text, pathology_category,
        is_negated, laterality, severity, anatomy, entity_label.
    """
    rows = []
    for report_id, annotation in radgraph_data.items():
        entities = extract_entities(annotation)
        relations = extract_relations(annotation)
        claims = entities_to_claims(entities, relations)

        for claim in claims:
            rows.append({
                "report_id": report_id,
                "claim_text": claim.text,
                "pathology_category": claim.pathology_category,
                "is_negated": claim.is_negated,
                "laterality": claim.laterality,
                "severity": claim.severity,
                "anatomy": claim.anatomy,
                "entity_label": claim.entities[0].label if claim.entities else "",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No claims parsed from RadGraph data")
    else:
        logger.info(
            f"Parsed {len(df)} claims from {df['report_id'].nunique()} reports. "
            f"Category distribution:\n{df['pathology_category'].value_counts().to_string()}"
        )
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse RadGraph annotations")
    parser.add_argument("--radgraph-json", type=str, required=True)
    parser.add_argument("--output", type=str, default="./radgraph_claims.csv")
    args = parser.parse_args()

    data = load_radgraph(args.radgraph_json)
    claims_df = parse_all_reports(data)
    claims_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(claims_df)} claims to {args.output}")
