"""Coverage computation for ClaimGuard-CXR best-of-N selection.

Computes what fraction of CheXbert-detected findings are mentioned
in a candidate report. Used in the constrained best-of-N objective.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that map to CheXpert finding labels
_FINDING_KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion"],
    "Consolidation": ["consolidation"],
    "Pneumonia": ["pneumonia"],
    "Atelectasis": ["atelectasis", "volume loss"],
    "Pneumothorax": ["pneumothorax"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Lung Opacity": ["opacity", "infiltrate", "haziness"],
    "Lung Lesion": ["mass", "nodule", "lesion"],
    "Fracture": ["fracture"],
    "Support Devices": ["tube", "catheter", "line", "pacemaker", "device", "drain"],
    "Enlarged Cardiomediastinum": ["mediastinum", "mediastinal widening"],
    "Pleural Other": ["pleural thickening"],
}


def extract_mentioned_findings(report_text: str) -> set[str]:
    """Extract which CheXpert findings are mentioned in a report.

    Args:
        report_text: Generated report text.

    Returns:
        Set of CheXpert finding labels mentioned.
    """
    text_lower = report_text.lower()
    mentioned = set()

    for finding, keywords in _FINDING_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                mentioned.add(finding)
                break

    # Check for "No Finding" / normal
    normal_phrases = ["no acute", "unremarkable", "normal", "clear lungs", "within normal limits"]
    if any(p in text_lower for p in normal_phrases) and not mentioned:
        mentioned.add("No Finding")

    return mentioned


def compute_coverage(
    report_text: str,
    detected_findings: set[str],
) -> float:
    """Compute fraction of detected findings mentioned in a report.

    Args:
        report_text: Generated report text.
        detected_findings: CheXpert labels detected in the image (from CheXbert or ensemble).

    Returns:
        Coverage fraction in [0, 1].
    """
    if not detected_findings or detected_findings == {"No Finding"}:
        # "No Finding" image: coverage=1 if report says normal
        mentioned = extract_mentioned_findings(report_text)
        if "No Finding" in mentioned or not mentioned - {"No Finding", "Support Devices"}:
            return 1.0
        return 0.0

    # Remove meta-labels
    relevant_detected = detected_findings - {"No Finding"}
    if not relevant_detected:
        return 1.0

    mentioned = extract_mentioned_findings(report_text)
    overlap = mentioned & relevant_detected
    return len(overlap) / len(relevant_detected)


def compute_coverage_from_claims(
    claims: list[dict],
    detected_findings: set[str],
) -> float:
    """Compute coverage from decomposed claims (alternative to text-based).

    Args:
        claims: List of claim dicts with 'pathology' key.
        detected_findings: CheXpert findings detected in image.

    Returns:
        Coverage fraction in [0, 1].
    """
    if not detected_findings or detected_findings == {"No Finding"}:
        claim_pathologies = {c["pathology"] for c in claims}
        if "No Finding" in claim_pathologies or not claim_pathologies - {"No Finding", "Support Devices"}:
            return 1.0
        return 0.0

    relevant_detected = detected_findings - {"No Finding"}
    if not relevant_detected:
        return 1.0

    mentioned = {c["pathology"] for c in claims}
    overlap = mentioned & relevant_detected
    return len(overlap) / len(relevant_detected)
