"""Deterministic claim synthesis from radiologist annotations.

Used for detection/segmentation-only datasets (RSNA Pneumonia, SIIM-ACR,
Object-CXR, ChestX-Det10) that ship bounding boxes or pixel masks but no
free-text reports. Produces `(StructuredClaim, gt_label)` tuples that bypass
the LLM extractor/parser, since the structure comes directly from the
annotation metadata.

Template:
    positive: "There is {finding} in the {laterality} {location}."
    negative: "There is no {finding} in the {laterality} {location}."

For each annotated finding on an image we emit one positive claim (GT =
SUPPORTED). For label balance we optionally emit one negative claim sampled
from findings that are ABSENT on the image (GT = CONTRADICTED).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .claim_matcher import Annotation
from .claim_parser import StructuredClaim, PROMPT_VERSION
from ._common import stable_id


_FINDING_FAMILY = {
    "pneumonia": "consolidation_family",
    "consolidation": "consolidation_family",
    "pneumothorax": "pleural_family",
    "pleural_effusion": "pleural_family",
    "effusion": "pleural_family",
    "atelectasis": "lung_family",
    "edema": "lung_family",
    "cardiomegaly": "cardiac_family",
    "nodule": "lung_family",
    "mass": "lung_family",
    "opacity": "lung_family",
    "fibrosis": "lung_family",
    "support_device": "device_family",
    "foreign_object": "device_family",
}


def _laterality_phrase(lat: str) -> str:
    return {
        "L": "left ",
        "R": "right ",
        "bilateral": "bilateral ",
        "none": "",
        "unknown": "",
    }.get(lat, "")


def _location_phrase(location: str) -> str:
    # Keep location natural. If "unknown", default to "lung" for lung-family
    # findings; caller should supply a better location if available.
    if not location or location == "unknown":
        return "lung"
    return location.replace("_", " ")


def _finding_phrase(finding: str) -> str:
    return finding.replace("_", " ")


@dataclass
class SynthesizedClaim:
    structured: StructuredClaim
    gt_label: str  # "SUPPORTED" or "CONTRADICTED"


def synthesize_claims_for_image(
    image_id: str,
    annotations: list[Annotation],
    *,
    source: str,
    location_for_finding: dict[str, str] | None = None,
    all_findings_in_site: list[str] | None = None,
    emit_negatives: bool = True,
    emit_contradicted_positives: bool = True,
    seed: int = 17,
) -> list[SynthesizedClaim]:
    """Produce claims from annotations in three flavors:

    - Positive assertion, finding present: "There is {X} in the {lat} {loc}."
      where X is present on the image → GT = SUPPORTED.
    - Negative assertion, finding absent: "There is no {X} in the {loc}."
      where X is absent on the image → GT = SUPPORTED (the negation is true).
    - Positive assertion, finding absent: "There is {X} in the {loc}." where
      X is ABSENT on the image → GT = CONTRADICTED. This is the critical
      class that gives detection-only sites non-zero CONTRADICTED training data.

    Args:
        image_id: stable identifier for the image (becomes report_id).
        annotations: list of Annotation for this image.
        source: site name used in claim_id stable hash + generator_id.
        location_for_finding: optional override mapping finding -> anatomical
            location string (e.g., pneumothorax -> pleural_space). If None,
            "lung" is used as a generic fallback.
        all_findings_in_site: sample negative/contradicted claims from findings
            in this list that are absent on the image. Required when either
            negative flag is True.
        emit_negatives: whether to emit a "there is no X" claim (GT=SUPPORTED).
        emit_contradicted_positives: whether to emit a "there is X" assertion
            about an absent finding (GT=CONTRADICTED).
        seed: RNG seed for finding selection.

    Returns:
        List of SynthesizedClaim. Empty if no annotations and no negatives.
    """
    rng = random.Random(f"{source}:{image_id}:{seed}")
    out: list[SynthesizedClaim] = []
    location_for_finding = location_for_finding or {}

    present_findings: set[str] = set()
    for ann in annotations:
        finding = ann.finding
        present_findings.add(finding)
        loc = location_for_finding.get(finding, _location_phrase("lung"))
        lat = ann.laterality if ann.laterality else "unknown"
        lat_phrase = _laterality_phrase(lat)
        finding_phrase = _finding_phrase(finding)
        raw = f"There is {finding_phrase} in the {lat_phrase}{loc}.".replace("  ", " ")
        claim_id = stable_id("synth", source, image_id, finding, lat, "pos")
        structured = StructuredClaim(
            claim_id=claim_id,
            raw_text=raw,
            report_id=image_id,
            finding=finding,
            finding_family=_FINDING_FAMILY.get(finding, "unknown"),
            location=loc,
            laterality=lat,
            severity="unknown",
            temporality="unknown",
            comparison="present",
            modifier_tags=["synthesized"],
            evidence_source_type="annotation_synthesized",
            generator_id=f"synth:{source}",
            generator_temperature=None,
            generator_seed=seed,
            parser_confidence=1.0,
            parser_version=PROMPT_VERSION,
        )
        out.append(SynthesizedClaim(structured=structured, gt_label="SUPPORTED"))

    absent_findings: list[str] = []
    if all_findings_in_site:
        absent_findings = [f for f in all_findings_in_site if f not in present_findings]

    if emit_negatives and absent_findings:
        neg = rng.choice(absent_findings)
        loc = location_for_finding.get(neg, _location_phrase("lung"))
        finding_phrase = _finding_phrase(neg)
        raw = f"There is no {finding_phrase} in the {loc}."
        claim_id = stable_id("synth", source, image_id, neg, "-", "neg")
        structured = StructuredClaim(
            claim_id=claim_id,
            raw_text=raw,
            report_id=image_id,
            finding=neg,
            finding_family=_FINDING_FAMILY.get(neg, "unknown"),
            location=loc,
            laterality="unknown",
            severity="unknown",
            temporality="unknown",
            comparison="absent",
            modifier_tags=["synthesized", "negative", "polarity_negated"],
            evidence_source_type="annotation_synthesized",
            generator_id=f"synth:{source}",
            generator_temperature=None,
            generator_seed=seed,
            parser_confidence=1.0,
            parser_version=PROMPT_VERSION,
        )
        # "no X" when X is absent → claim is true → SUPPORTED
        out.append(SynthesizedClaim(structured=structured, gt_label="SUPPORTED"))

    if emit_contradicted_positives and absent_findings:
        # Pick a DIFFERENT absent finding from the "no X" negation above (if
        # both emit, avoid using the same finding twice with conflicting logic).
        pool = [f for f in absent_findings if not (emit_negatives and len(absent_findings) > 1 and f == neg)] \
            if emit_negatives else absent_findings
        if pool:
            con = rng.choice(pool)
            loc = location_for_finding.get(con, _location_phrase("lung"))
            finding_phrase = _finding_phrase(con)
            raw = f"There is {finding_phrase} in the {loc}."
            claim_id = stable_id("synth", source, image_id, con, "-", "contra")
            structured = StructuredClaim(
                claim_id=claim_id,
                raw_text=raw,
                report_id=image_id,
                finding=con,
                finding_family=_FINDING_FAMILY.get(con, "unknown"),
                location=loc,
                laterality="unknown",
                severity="unknown",
                temporality="unknown",
                comparison="present",
                modifier_tags=["synthesized"],
                evidence_source_type="annotation_synthesized",
                generator_id=f"synth:{source}",
                generator_temperature=None,
                generator_seed=seed,
                parser_confidence=1.0,
                parser_version=PROMPT_VERSION,
            )
            # Asserts presence of a finding that IS NOT on the image → CONTRADICTED
            out.append(SynthesizedClaim(structured=structured, gt_label="CONTRADICTED"))

    return out


def default_site_findings(source: str) -> list[str]:
    """Best-known finding taxonomy for each detection-only site."""
    return {
        "rsna_pneumonia": ["pneumonia"],
        "siim_acr": ["pneumothorax"],
        "object_cxr": ["foreign_object"],
        "chestx_det10": [
            "atelectasis",
            "calcification",
            "cardiomegaly",
            "consolidation",
            "effusion",
            "edema",
            "emphysema",
            "fibrosis",
            "fracture",
            "mass",
            "nodule",
            "pneumothorax",
        ],
    }.get(source, [])
