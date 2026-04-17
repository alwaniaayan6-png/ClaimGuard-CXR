"""Tests for the image-grounded claim matcher. Uses synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from v5.data.claim_matcher import (
    Annotation,
    AnatomyMask,
    ClaimMatcher,
    GTLabel,
    MatcherConfig,
)
from v5.data.claim_parser import StructuredClaim


def _anatomy_half_image() -> AnatomyMask:
    H, W = 100, 100
    left = np.zeros((H, W), dtype=np.uint8)
    left[:, 50:] = 1  # observer-right half = anatomical LEFT
    right = np.zeros((H, W), dtype=np.uint8)
    right[:, :50] = 1
    return AnatomyMask(
        image_id="img1",
        masks={
            "left_upper_lung": left,
            "left_mid_lung": left,
            "left_lower_lung": left,
            "right_upper_lung": right,
            "right_mid_lung": right,
            "right_lower_lung": right,
            "heart": np.zeros((H, W), dtype=np.uint8),
            "mediastinum": np.zeros((H, W), dtype=np.uint8),
        },
        height=H,
        width=W,
    )


def _structured_claim(**kw) -> StructuredClaim:
    defaults = dict(
        claim_id="c", raw_text="", report_id="r", finding="pleural_effusion",
        finding_family="effusive", location="left_lower_lobe",
        laterality="L", severity="unknown", temporality="unknown",
        comparison="unknown", modifier_tags=[],
        evidence_source_type="oracle_human", generator_id="oracle",
    )
    defaults.update(kw)
    return StructuredClaim(**defaults)


def test_supported_when_annotation_matches():
    m = ClaimMatcher()
    claim = _structured_claim()
    # annotation on observer-right = anatomical LEFT lower half
    ann = Annotation(
        image_id="img1",
        finding="pleural_effusion",
        bbox=(0.6, 0.6, 0.9, 0.9),
        source="synthetic",
    )
    result = m.match(claim, [ann], _anatomy_half_image())
    assert result.label == GTLabel.SUPPORTED


def test_contradicted_when_laterality_mismatch():
    m = ClaimMatcher()
    claim = _structured_claim(laterality="L")
    # annotation on observer-LEFT = anatomical RIGHT
    ann = Annotation(
        image_id="img1",
        finding="pleural_effusion",
        bbox=(0.1, 0.6, 0.3, 0.9),
        source="synthetic",
    )
    result = m.match(claim, [ann], _anatomy_half_image())
    assert result.label == GTLabel.CONTRADICTED


def test_no_gt_when_no_annotation_of_family():
    m = ClaimMatcher()
    claim = _structured_claim(finding="pneumothorax", finding_family="air")
    ann = Annotation(
        image_id="img1",
        finding="pleural_effusion",
        bbox=(0.6, 0.6, 0.9, 0.9),
        source="synthetic",
    )
    result = m.match(claim, [ann], _anatomy_half_image())
    assert result.label == GTLabel.NO_GT


def test_structured_negative_contradicts_positive_claim():
    m = ClaimMatcher()
    claim = _structured_claim(finding="pneumothorax", finding_family="air", laterality="R")
    ann = Annotation(
        image_id="img1", finding="pneumothorax",
        source="chexpert_labeler", is_structured_negative=True,
    )
    result = m.match(claim, [ann], _anatomy_half_image())
    assert result.label == GTLabel.CONTRADICTED


def test_claim_negation_matches_absence():
    m = ClaimMatcher()
    claim = _structured_claim(
        finding="no_finding", finding_family="misc",
        modifier_tags=["polarity_negated"],
    )
    ann = Annotation(
        image_id="img1", finding="pneumothorax",
        source="chexpert_labeler", is_structured_negative=True,
    )
    result = m.match(claim, [ann], _anatomy_half_image())
    assert result.label in {GTLabel.SUPPORTED, GTLabel.NO_GT}
