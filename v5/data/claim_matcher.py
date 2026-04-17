"""Claim-to-annotation matcher: compute image-grounded GT for (image, claim).

Given a StructuredClaim and the set of radiologist annotations on its image
(bounding boxes, segmentation masks, structured labels, anatomy masks), return
one of {SUPPORTED, CONTRADICTED, NO_GT, NO_FINDING}. See §4.2 of the
architecture doc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from .claim_parser import StructuredClaim, load_ontology

logger = logging.getLogger(__name__)


class GTLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NO_GT = "NO_GT"
    NO_FINDING = "NO_FINDING"


@dataclass
class Annotation:
    """A single radiologist annotation on an image."""

    image_id: str
    finding: str                    # unified 23-class
    laterality: str = "unknown"     # {L, R, bilateral, none, unknown}
    bbox: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2) normalized 0–1
    mask: np.ndarray | None = None  # HxW binary mask
    source: str = "unknown"         # {ms_cxr, rsna, siim, object_cxr, chestx_det10, padchest, chexpert_labeler}
    confidence: float = 1.0
    is_structured_negative: bool = False  # e.g., CheXpert "no consolidation" = 0 for this image


@dataclass
class AnatomyMask:
    """Per-image anatomy segmentation as a dict {region_id: HxW binary mask}."""

    image_id: str
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # region_id -> HxW
    height: int = 0
    width: int = 0


@dataclass
class MatcherConfig:
    tau_iou: float = 0.3
    require_laterality_match: bool = True
    enable_no_finding_rule: bool = True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _bbox_to_mask(bbox: tuple[float, float, float, float], H: int, W: int) -> np.ndarray:
    """Convert a normalized (x1, y1, x2, y2) bbox to a HxW binary mask."""
    x1, y1, x2, y2 = bbox
    x1i, y1i = int(round(x1 * W)), int(round(y1 * H))
    x2i, y2i = int(round(x2 * W)), int(round(y2 * H))
    x1i, x2i = max(0, min(x1i, W)), max(0, min(x2i, W))
    y1i, y2i = max(0, min(y1i, H)), max(0, min(y2i, H))
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1i:y2i, x1i:x2i] = 1
    return mask


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        # resample mask_b to mask_a shape via nearest
        try:
            from PIL import Image

            m = Image.fromarray(mask_b.astype(np.uint8) * 255).resize(
                (mask_a.shape[1], mask_a.shape[0]), Image.NEAREST
            )
            mask_b = (np.array(m) > 127).astype(np.uint8)
        except Exception:
            return 0.0
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# Laterality
# ---------------------------------------------------------------------------


def _ann_side_from_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Given a normalized bbox, infer side based on centroid x.

    Convention: x=0 is image left (observer-facing left, which is patient RIGHT
    for standard PA/AP radiographs with the patient facing the viewer).
    """
    cx = 0.5 * (bbox[0] + bbox[2])
    # Under PA/AP convention, anatomical LEFT is on the observer's RIGHT (cx > 0.5).
    if cx < 0.45:
        return "R"
    elif cx > 0.55:
        return "L"
    return "none"  # midline / bilateral


def _ann_side_from_mask(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return "unknown"
    W = mask.shape[1]
    cx = xs.mean() / W
    if cx < 0.45:
        return "R"
    elif cx > 0.55:
        return "L"
    return "none"


def _resolve_annotation_laterality(ann: Annotation) -> str:
    if ann.laterality not in ("unknown", ""):
        return ann.laterality
    if ann.bbox is not None:
        return _ann_side_from_bbox(ann.bbox)
    if ann.mask is not None:
        return _ann_side_from_mask(ann.mask)
    return "unknown"


def _laterality_compatible(ann_lat: str, claim_lat: str) -> bool:
    if claim_lat in ("unknown", "none"):
        return True
    if ann_lat in ("unknown", "none"):
        return True
    if ann_lat == "bilateral" or claim_lat == "bilateral":
        return True
    return ann_lat == claim_lat


# ---------------------------------------------------------------------------
# Anatomical overlap
# ---------------------------------------------------------------------------


_LOCATION_TO_ANATOMY = {
    "left_upper_lobe": ["left_upper_lung"],
    "left_lower_lobe": ["left_lower_lung"],
    "left_mid_lung": ["left_mid_lung", "left_upper_lung", "left_lower_lung"],
    "right_upper_lobe": ["right_upper_lung"],
    "right_middle_lobe": ["right_mid_lung"],
    "right_lower_lobe": ["right_lower_lung"],
    "right_mid_lung": ["right_mid_lung"],
    "lung_base_left": ["left_lower_lung"],
    "lung_base_right": ["right_lower_lung"],
    "apex_left": ["left_upper_lung"],
    "apex_right": ["right_upper_lung"],
    "lingula": ["left_mid_lung", "left_lower_lung"],
    "mediastinum": ["mediastinum", "heart"],
    "hilum_left": ["left_mid_lung"],
    "hilum_right": ["right_mid_lung"],
    "heart": ["heart", "mediastinum"],
    "diaphragm": ["left_lower_lung", "right_lower_lung"],
    "pleural_space": ["left_upper_lung", "left_lower_lung", "right_upper_lung", "right_lower_lung"],
    "unspecified": [],
}


def _anatomical_overlap(
    ann: Annotation,
    claim_location: str,
    anatomy: AnatomyMask | None,
) -> float:
    """Return an overlap score in [0, 1] between annotation and claim location."""
    if claim_location in ("unknown", "unspecified"):
        return 1.0  # no location constraint → treat as spatially compatible
    if anatomy is None or not anatomy.masks:
        return 0.5  # no anatomy info → soft-pass
    regions = _LOCATION_TO_ANATOMY.get(claim_location, [])
    if not regions:
        return 0.5
    ann_mask = None
    if ann.mask is not None:
        ann_mask = (ann.mask > 0).astype(np.uint8)
    elif ann.bbox is not None:
        ann_mask = _bbox_to_mask(ann.bbox, anatomy.height, anatomy.width)
    if ann_mask is None:
        return 0.0
    anat_union = np.zeros_like(ann_mask, dtype=np.uint8)
    for region in regions:
        m = anatomy.masks.get(region)
        if m is None:
            continue
        if m.shape != anat_union.shape:
            # resize nearest
            try:
                from PIL import Image

                m = np.array(
                    Image.fromarray(m.astype(np.uint8) * 255).resize(
                        (anat_union.shape[1], anat_union.shape[0]), Image.NEAREST
                    )
                )
                m = (m > 127).astype(np.uint8)
            except Exception:
                continue
        anat_union |= (m > 0).astype(np.uint8)
    if anat_union.sum() == 0:
        return 0.0
    # asymmetric overlap: fraction of annotation that lies in claim-anatomy
    inter = np.logical_and(ann_mask > 0, anat_union > 0).sum()
    if ann_mask.sum() == 0:
        return 0.0
    return float(inter / ann_mask.sum())


# ---------------------------------------------------------------------------
# Finding-family compatibility
# ---------------------------------------------------------------------------


def _finding_compatible(ann_finding: str, claim_finding: str, ontology: dict) -> bool:
    """Exact match or same finding-family."""
    if ann_finding == claim_finding:
        return True
    groups = ontology.get("finding_family_groups", {})
    for members in groups.values():
        if ann_finding in members and claim_finding in members:
            return True
    return False


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    label: GTLabel
    matching_annotation_id: str | None = None
    spatial_overlap: float = 0.0
    reason: str = ""
    coverage: bool = True  # False iff NO_GT


class ClaimMatcher:
    def __init__(self, cfg: MatcherConfig | None = None, ontology: dict | None = None):
        self.cfg = cfg or MatcherConfig()
        self.ontology = ontology or load_ontology()

    # -- public API ----------------------------------------------------------

    def match(
        self,
        claim: StructuredClaim,
        annotations: Iterable[Annotation],
        anatomy: AnatomyMask | None = None,
    ) -> MatchResult:
        """Compute image-grounded GT for one claim."""
        annotations = list(annotations)

        # Rule 1: explicit negation in claim ("no consolidation")
        claim_is_negation = "polarity_negated" in claim.modifier_tags or claim.finding == "no_finding"

        # Rule 2: "no finding" image + claim asserts something present → CONTRADICTED
        image_is_no_finding = any(
            a.finding == "no_finding" and not a.is_structured_negative for a in annotations
        )
        structured_negatives_for_finding = [
            a for a in annotations
            if a.is_structured_negative
            and _finding_compatible(a.finding, claim.finding, self.ontology)
        ]
        matching_cands = [
            a for a in annotations
            if (not a.is_structured_negative)
            and a.finding != "no_finding"
            and _finding_compatible(a.finding, claim.finding, self.ontology)
        ]

        # --- case: claim asserts nothing is there -----------------------------
        if claim_is_negation:
            # "no X" — supported iff no annotation of X exists on image
            if matching_cands:
                return MatchResult(
                    label=GTLabel.CONTRADICTED,
                    matching_annotation_id=matching_cands[0].source,
                    reason="claim negates a finding that is present per radiologist annotation",
                )
            if structured_negatives_for_finding or image_is_no_finding:
                return MatchResult(label=GTLabel.SUPPORTED, reason="claim-negation matches absence")
            return MatchResult(
                label=GTLabel.NO_GT,
                coverage=False,
                reason="no annotation coverage for the negated finding",
            )

        # --- case: claim asserts finding present -----------------------------
        if not matching_cands:
            if structured_negatives_for_finding:
                return MatchResult(
                    label=GTLabel.CONTRADICTED,
                    reason="claim asserts finding but structured label says absent",
                )
            if image_is_no_finding and self.cfg.enable_no_finding_rule:
                return MatchResult(
                    label=GTLabel.CONTRADICTED,
                    reason="image labeled no_finding but claim asserts a finding",
                )
            return MatchResult(
                label=GTLabel.NO_GT,
                coverage=False,
                reason="no annotation of that finding family and no structured negative",
            )

        # Candidates exist → check spatial + laterality compatibility
        best_overlap = 0.0
        best_ann: Annotation | None = None
        for ann in matching_cands:
            ann_lat = _resolve_annotation_laterality(ann)
            lat_ok = _laterality_compatible(ann_lat, claim.laterality) or not self.cfg.require_laterality_match
            if not lat_ok:
                continue
            overlap = _anatomical_overlap(ann, claim.location, anatomy)
            if overlap >= best_overlap:
                best_overlap = overlap
                best_ann = ann
        if best_ann is not None and best_overlap >= self.cfg.tau_iou:
            return MatchResult(
                label=GTLabel.SUPPORTED,
                matching_annotation_id=best_ann.source,
                spatial_overlap=best_overlap,
                reason="annotation matches finding + location + laterality",
            )

        # Finding family present but location/laterality mismatch → CONTRADICTED
        return MatchResult(
            label=GTLabel.CONTRADICTED,
            matching_annotation_id=matching_cands[0].source,
            spatial_overlap=best_overlap,
            reason="finding present but location/laterality mismatch",
        )


def save_matcher_outputs(results: list[dict], path: Path) -> None:
    import json

    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
