"""Image-grounded claim verification logic.

For each (image, claim) pair, decide whether the claim's asserted
referent overlaps a radiologist-drawn annotation on the image. This is
the core of ClaimGuard-Bench-Grounded: ground truth is anchored in
pixels, not in another radiologist's text.

See ARCHITECTURE_PATH_B.md Section 5 for the algorithm rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .claim_schema import Claim, ClaimCertainty, Laterality, Region


IOU_THRESHOLD_DEFAULT = 0.5  # headline; sensitivity curve at {0.1, 0.3, 0.5, 0.7}

DIFFUSE_FINDINGS = frozenset({
    "cardiomegaly",
    "pulmonary_edema",
    "fibrosis",
})


class GroundingOutcome(str, Enum):
    """Result of grounding a single claim against radiologist image annotations."""
    GROUNDED_SUPPORTED = "grounded_supported"          # radiologist annotation agrees
    GROUNDED_CONTRADICTED = "grounded_contradicted"    # radiologist annotation disagrees
    GROUNDED_ABSENT_OK = "grounded_absent_ok"          # claim asserts absent, no annotation — consistent
    GROUNDED_ABSENT_FAIL = "grounded_absent_fail"      # claim asserts present, no annotation — contradicts
    UNGROUNDED = "ungrounded"                          # cannot evaluate on this dataset / claim type


@dataclass(frozen=True)
class Annotation:
    """Radiologist-drawn annotation for a single (finding, image) pair.

    A single image may carry multiple annotations for the same finding
    (multiple lesions). The grounding module unions them.
    """
    finding: str                    # canonical id
    mask: np.ndarray                # boolean, shape (H, W)
    confidence: float = 1.0         # typically 1.0 for radiologist-drawn


def soft_region_prior(
    region: Region,
    laterality: Laterality,
    image_shape: Tuple[int, int],
    peak: float = 1.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Build a soft attention prior over a 2D image for a claim's asserted region.

    Unlike v1's hard binary mask, this returns a smooth weight map that
    peaks in the asserted region and decays elsewhere. Used for
    weighted_iou against radiologist annotation masks.

    Parameters
    ----------
    region : Region
    laterality : Laterality
    image_shape : (H, W)
    peak, floor : maximum and minimum prior weight

    Returns
    -------
    np.ndarray, shape (H, W), dtype float32, values in [floor, peak].
    """
    H, W = image_shape
    prior = np.full((H, W), floor, dtype=np.float32)

    # Laterality boundaries (CXR convention: viewer's right = patient's left).
    # We encode in patient-frame: LEFT = right half of image, RIGHT = left half.
    if laterality == Laterality.LEFT:
        col_mask = np.zeros(W, dtype=bool)
        col_mask[W // 2:] = True
    elif laterality == Laterality.RIGHT:
        col_mask = np.zeros(W, dtype=bool)
        col_mask[:W // 2] = True
    else:
        col_mask = np.ones(W, dtype=bool)

    # Vertical region boundaries (approximate, CXR-standard thirds).
    if region == Region.UPPER or region == Region.APICAL:
        row_hi = int(H * 0.35)
        row_lo = 0
    elif region == Region.MIDDLE or region == Region.PERIHILAR:
        row_lo = int(H * 0.30)
        row_hi = int(H * 0.65)
    elif region == Region.LOWER or region == Region.BASAL or region == Region.COSTOPHRENIC:
        row_lo = int(H * 0.60)
        row_hi = H
    elif region == Region.RETROCARDIAC:
        # Lower-medial; narrow column band
        row_lo = int(H * 0.55)
        row_hi = int(H * 0.90)
        col_mask = np.zeros(W, dtype=bool)
        col_mask[int(W * 0.35): int(W * 0.65)] = True
    elif region == Region.MEDIASTINAL:
        row_lo = int(H * 0.20)
        row_hi = int(H * 0.75)
        col_mask = np.zeros(W, dtype=bool)
        col_mask[int(W * 0.35): int(W * 0.65)] = True
    else:
        row_lo, row_hi = 0, H

    prior[row_lo:row_hi, col_mask] = peak

    # Smooth the hard rectangle into a soft map — Gaussian blur via cumulative
    # distance fall-off. Kept dependency-free; use scipy.ndimage.gaussian_filter
    # if available at runtime.
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        sigma = max(H, W) / 30.0
        prior = gaussian_filter(prior, sigma=sigma)
        # Re-normalize to [floor, peak] after blur.
        prior = floor + (peak - floor) * (prior - prior.min()) / max(prior.max() - prior.min(), 1e-6)
    except ImportError:
        pass  # return hard-rectangle prior if scipy unavailable

    return prior.astype(np.float32)


def weighted_iou(prior: np.ndarray, annotation_mask: np.ndarray) -> float:
    """IoU between a soft prior and a binary annotation mask.

    Generalizes standard IoU to soft priors:
        weighted_iou = sum(prior * ann) / sum(max(prior, ann))

    Both inputs must be the same shape. annotation_mask may be bool or float.
    """
    if prior.shape != annotation_mask.shape:
        raise ValueError(f"Shape mismatch: prior {prior.shape} vs ann {annotation_mask.shape}")

    ann = annotation_mask.astype(np.float32)
    intersection = float((prior * ann).sum())
    union = float(np.maximum(prior, ann).sum())
    if union <= 0:
        return 0.0
    return intersection / union


def union_of_masks(
    annotations: Sequence[Annotation],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Union of a list of radiologist annotation masks for the same finding."""
    union = np.zeros(image_shape, dtype=bool)
    for ann in annotations:
        if ann.mask.shape != image_shape:
            raise ValueError(
                f"Annotation shape {ann.mask.shape} does not match image {image_shape}"
            )
        union |= ann.mask.astype(bool)
    return union


def _lateralities_match(claim: Claim, annotations: Sequence[Annotation]) -> bool:
    """True iff the claim's laterality is consistent with the annotations' centroid side.

    If claim laterality is unspecified or bilateral, this returns True unconditionally.
    Otherwise, we check that at least one annotation centroid falls in the claim's
    asserted half-image.
    """
    if claim.laterality in (Laterality.UNSPECIFIED, Laterality.BILATERAL):
        return True
    for ann in annotations:
        ys, xs = np.where(ann.mask)
        if xs.size == 0:
            continue
        centroid_x = float(xs.mean())
        W = ann.mask.shape[1]
        centroid_in_left_half = centroid_x < W / 2
        # Patient LEFT = image right half (not-left-half).
        if claim.laterality == Laterality.LEFT and not centroid_in_left_half:
            return True
        if claim.laterality == Laterality.RIGHT and centroid_in_left_half:
            return True
    return False


def _size_ratio_match(
    claim: Claim,
    annotations: Sequence[Annotation],
    image_shape: Tuple[int, int],
) -> GroundingOutcome:
    """For diffuse findings, compare annotation area ratio to claim severity.

    Very rough: if the claim asserts presence and the union mask covers more than
    2% of the image, we say SUPPORTED. Below 2%, CONTRADICTED. This heuristic is
    documented as a sensitivity hyperparameter in the supplementary.
    """
    union = union_of_masks(annotations, image_shape)
    area_ratio = float(union.sum()) / (image_shape[0] * image_shape[1])
    if area_ratio >= 0.02:
        return GroundingOutcome.GROUNDED_SUPPORTED
    return GroundingOutcome.GROUNDED_CONTRADICTED


def ground_claim(
    claim: Claim,
    annotations: dict,
    image_shape: Tuple[int, int],
    dataset_label_schema: Iterable[str],
    iou_threshold: float = IOU_THRESHOLD_DEFAULT,
) -> GroundingOutcome:
    """Decide the grounded outcome of a single claim against image annotations.

    Parameters
    ----------
    claim : Claim
        Structured claim.
    annotations : dict[str, list[Annotation]]
        Map from canonical finding id to a list of radiologist annotations on this image.
    image_shape : (H, W)
    dataset_label_schema : iterable of canonical finding ids
        Which findings the annotators were asked to draw for this dataset.
        Absence of an annotation for a finding NOT in the schema means
        'unknown', NOT 'absent'.
    iou_threshold : float
        Headline threshold; default 0.5 per v2 review.

    Returns
    -------
    GroundingOutcome
    """
    schema = frozenset(dataset_label_schema)

    # Claim type gate — some claim types cannot be grounded against still images.
    if not claim.is_groundable():
        return GroundingOutcome.UNGROUNDED

    # Finding-schema gate — a dataset that does not label this finding cannot evaluate it.
    if claim.finding not in schema:
        return GroundingOutcome.UNGROUNDED

    finding_annotations = annotations.get(claim.finding, [])

    # Absence claim.
    if claim.certainty == ClaimCertainty.ABSENT:
        if len(finding_annotations) == 0:
            return GroundingOutcome.GROUNDED_ABSENT_OK
        return GroundingOutcome.GROUNDED_CONTRADICTED

    # Presence claim, no annotation.
    if len(finding_annotations) == 0:
        return GroundingOutcome.GROUNDED_ABSENT_FAIL

    # Diffuse finding branch.
    if claim.finding in DIFFUSE_FINDINGS:
        return _size_ratio_match(claim, finding_annotations, image_shape)

    # Localizable finding — IoU check.
    prior = soft_region_prior(claim.region, claim.laterality, image_shape)
    ann_mask = union_of_masks(finding_annotations, image_shape).astype(np.float32)
    iou = weighted_iou(prior, ann_mask)

    if iou >= iou_threshold and _lateralities_match(claim, finding_annotations):
        return GroundingOutcome.GROUNDED_SUPPORTED
    return GroundingOutcome.GROUNDED_CONTRADICTED
