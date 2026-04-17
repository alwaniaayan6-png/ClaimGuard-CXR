"""Unit tests for the claim-to-annotation grounding logic."""
from __future__ import annotations

import numpy as np

from claimguard_nmi.grounding import (
    Claim,
    ClaimCertainty,
    ClaimType,
    GroundingOutcome,
    ground_claim,
    soft_region_prior,
    weighted_iou,
)
from claimguard_nmi.grounding.claim_schema import Laterality, Region
from claimguard_nmi.grounding.grounding import Annotation


DATASET_SCHEMA_LUNG = {"lung_opacity", "pleural_effusion"}


def _make_mask(shape, box):
    m = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = box
    m[r0:r1, c0:c1] = True
    return m


def test_ungroundable_types_return_ungrounded():
    for claim_type in (
        ClaimType.PRIOR_COMPARISON,
        ClaimType.MEASUREMENT,
        ClaimType.IMPLICIT_GLOBAL_NEGATION,
        ClaimType.HEDGE,
        ClaimType.GLOBAL_DESCRIPTION,
    ):
        claim = Claim(
            raw_text="dummy",
            finding="lung_opacity",
            claim_type=claim_type,
        )
        out = ground_claim(claim, {}, (224, 224), DATASET_SCHEMA_LUNG)
        assert out is GroundingOutcome.UNGROUNDED, claim_type


def test_finding_not_in_schema_returns_ungrounded():
    claim = Claim(
        raw_text="no pneumothorax",
        finding="pneumothorax",  # not in DATASET_SCHEMA_LUNG
        claim_type=ClaimType.NEGATION,
        certainty=ClaimCertainty.ABSENT,
    )
    out = ground_claim(claim, {}, (224, 224), DATASET_SCHEMA_LUNG)
    assert out is GroundingOutcome.UNGROUNDED


def test_absence_claim_with_no_annotation_is_absent_ok():
    claim = Claim(
        raw_text="no pleural effusion",
        finding="pleural_effusion",
        claim_type=ClaimType.NEGATION,
        certainty=ClaimCertainty.ABSENT,
    )
    out = ground_claim(claim, {}, (224, 224), DATASET_SCHEMA_LUNG)
    assert out is GroundingOutcome.GROUNDED_ABSENT_OK


def test_absence_claim_with_annotation_is_contradicted():
    mask = _make_mask((224, 224), (100, 200, 50, 150))
    ann = Annotation(finding="pleural_effusion", mask=mask)
    claim = Claim(
        raw_text="no pleural effusion",
        finding="pleural_effusion",
        claim_type=ClaimType.NEGATION,
        certainty=ClaimCertainty.ABSENT,
    )
    out = ground_claim(
        claim, {"pleural_effusion": [ann]}, (224, 224), DATASET_SCHEMA_LUNG,
    )
    assert out is GroundingOutcome.GROUNDED_CONTRADICTED


def test_presence_claim_with_no_annotation_is_absent_fail():
    claim = Claim(
        raw_text="left lower lobe opacity",
        finding="lung_opacity",
        claim_type=ClaimType.FINDING,
        certainty=ClaimCertainty.PRESENT,
        laterality=Laterality.LEFT,
        region=Region.LOWER,
    )
    out = ground_claim(claim, {}, (224, 224), DATASET_SCHEMA_LUNG)
    assert out is GroundingOutcome.GROUNDED_ABSENT_FAIL


def test_presence_claim_with_matching_annotation_is_supported():
    # Left lower lobe = image right-half, bottom-third.
    H, W = 224, 224
    box = (int(H * 0.7), int(H * 0.95), int(W * 0.55), int(W * 0.95))
    mask = _make_mask((H, W), box)
    ann = Annotation(finding="lung_opacity", mask=mask)
    claim = Claim(
        raw_text="left lower lobe opacity",
        finding="lung_opacity",
        claim_type=ClaimType.FINDING,
        certainty=ClaimCertainty.PRESENT,
        laterality=Laterality.LEFT,
        region=Region.LOWER,
    )
    out = ground_claim(
        claim,
        {"lung_opacity": [ann]},
        (H, W),
        DATASET_SCHEMA_LUNG,
        iou_threshold=0.1,  # relaxed for the unit-test sized box
    )
    assert out is GroundingOutcome.GROUNDED_SUPPORTED


def test_presence_claim_with_wrong_laterality_is_contradicted():
    H, W = 224, 224
    # Box in image-left half = patient-RIGHT
    box = (int(H * 0.7), int(H * 0.95), int(W * 0.05), int(W * 0.45))
    mask = _make_mask((H, W), box)
    ann = Annotation(finding="lung_opacity", mask=mask)
    # But claim asserts LEFT
    claim = Claim(
        raw_text="left lower lobe opacity",
        finding="lung_opacity",
        claim_type=ClaimType.FINDING,
        certainty=ClaimCertainty.PRESENT,
        laterality=Laterality.LEFT,
        region=Region.LOWER,
    )
    out = ground_claim(
        claim,
        {"lung_opacity": [ann]},
        (H, W),
        DATASET_SCHEMA_LUNG,
        iou_threshold=0.1,
    )
    assert out is GroundingOutcome.GROUNDED_CONTRADICTED


def test_soft_region_prior_has_expected_shape_and_range():
    prior = soft_region_prior(Region.LOWER, Laterality.LEFT, (224, 224))
    assert prior.shape == (224, 224)
    assert np.all(prior >= 0.0)
    assert np.all(prior <= 1.01)


def test_weighted_iou_on_disjoint_regions_is_zero():
    prior = np.zeros((64, 64), dtype=np.float32)
    prior[:32, :32] = 1.0
    ann = np.zeros((64, 64), dtype=bool)
    ann[32:, 32:] = True
    assert weighted_iou(prior, ann) == 0.0


def test_weighted_iou_on_identical_masks_is_one():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0
    assert abs(weighted_iou(mask, (mask > 0).astype(np.float32)) - 1.0) < 1e-6
