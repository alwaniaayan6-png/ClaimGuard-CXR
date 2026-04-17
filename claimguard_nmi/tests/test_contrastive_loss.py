"""Unit tests for the 4-way contrastive loss.

Key property we want to hold: the loss is minimized by a model that
behaves correctly on ALL FOUR variants (V1/V2/V3/V4), and is NOT
minimized by any text-only model.
"""
from __future__ import annotations

import torch

from claimguard_nmi.training import (
    FourWayContrastiveLoss,
    VariantBatch,
    VARIANT_POSITIVE,
    VARIANT_EVIDENCE_SWAP,
    VARIANT_IMAGE_SWAP,
    VARIANT_IMAGE_MASK,
)


def _make_variant(logits: torch.Tensor, score: torch.Tensor, label: torch.Tensor) -> VariantBatch:
    return VariantBatch(verdict_logits=logits, supported_prob=score, label=label)


def test_perfect_model_has_small_loss():
    B = 4
    # V1: label 0, score near 1
    v1 = _make_variant(
        torch.tensor([[5.0, -5.0]] * B),
        torch.tensor([0.95] * B),
        torch.zeros(B),
    )
    # V2: label 1, score near 0
    v2 = _make_variant(
        torch.tensor([[-5.0, 5.0]] * B),
        torch.tensor([0.05] * B),
        torch.ones(B),
    )
    # V3: label 1 (image-swap), score near 0
    v3 = _make_variant(
        torch.tensor([[-5.0, 5.0]] * B),
        torch.tensor([0.05] * B),
        torch.ones(B),
    )
    # V4: label 1 (image-mask), score near 0
    v4 = _make_variant(
        torch.tensor([[-5.0, 5.0]] * B),
        torch.tensor([0.05] * B),
        torch.ones(B),
    )
    loss_mod = FourWayContrastiveLoss()
    out = loss_mod(
        {
            VARIANT_POSITIVE: v1,
            VARIANT_EVIDENCE_SWAP: v2,
            VARIANT_IMAGE_SWAP: v3,
            VARIANT_IMAGE_MASK: v4,
        }
    )
    assert out["loss"].item() < 1.0


def test_text_only_model_cannot_satisfy_image_margin():
    """A model that ignores the image produces identical scores on V1 and V3 -> img_margin > 0."""
    B = 4
    # V1 and V3 have identical text — a text-only model cannot distinguish.
    # Assume the text-only model correctly predicts "supported" for V1 (high score),
    # and because it cannot see the image, gives V3 the SAME high score.
    v1 = _make_variant(
        torch.tensor([[5.0, -5.0]] * B),
        torch.tensor([0.95] * B),
        torch.zeros(B),
    )
    # V2: different text (contra evidence) — text-only model can solve this correctly
    v2 = _make_variant(
        torch.tensor([[-5.0, 5.0]] * B),
        torch.tensor([0.05] * B),
        torch.ones(B),
    )
    # V3: identical text as V1; text-only model gives the same high score; TRUE label is 1.
    v3 = _make_variant(
        torch.tensor([[5.0, -5.0]] * B),   # text-only wrongly predicts "supported"
        torch.tensor([0.95] * B),
        torch.ones(B),                      # true label = 1 (contradicted)
    )
    loss_mod = FourWayContrastiveLoss()
    out = loss_mod(
        {
            VARIANT_POSITIVE: v1,
            VARIANT_EVIDENCE_SWAP: v2,
            VARIANT_IMAGE_SWAP: v3,
        }
    )
    # img_margin must be strictly positive when V1 score == V3 score.
    assert out["img_margin"].item() > 0
    # Losses we expect to remain: CE on V3 (predicts wrong class) + img_margin
    assert out["loss"].item() > 0.5


def test_only_v3_wrong_keeps_evi_margin_at_zero():
    """If text-only model gets V2 right but V3 wrong, evi_margin should be 0, img_margin > 0."""
    B = 2
    v1 = _make_variant(
        torch.tensor([[5.0, -5.0]] * B),
        torch.tensor([0.99] * B),
        torch.zeros(B),
    )
    v2 = _make_variant(
        torch.tensor([[-5.0, 5.0]] * B),
        torch.tensor([0.01] * B),
        torch.ones(B),
    )
    v3 = _make_variant(
        torch.tensor([[5.0, -5.0]] * B),
        torch.tensor([0.99] * B),
        torch.ones(B),
    )
    out = FourWayContrastiveLoss()({
        VARIANT_POSITIVE: v1,
        VARIANT_EVIDENCE_SWAP: v2,
        VARIANT_IMAGE_SWAP: v3,
    })
    assert out["evi_margin"].item() == 0.0
    assert out["img_margin"].item() > 0.0


def test_loss_raises_without_required_variants():
    B = 2
    v1 = _make_variant(
        torch.tensor([[1.0, 0.0]] * B),
        torch.tensor([0.7] * B),
        torch.zeros(B),
    )
    try:
        FourWayContrastiveLoss()({VARIANT_POSITIVE: v1})
        assert False, "should have raised"
    except KeyError:
        pass


def test_v4_weighting_matches_lambda_mask_not_lambda_ce_plus_mask():
    """Regression test for the V4 double-count bug caught in v2 review.

    When V4 is supplied, its contribution to the total loss must be
    exactly ``lambda_mask * CE(V4)``. Previous buggy code counted V4
    once in the main CE sum AND once as the mask term, so the effective
    weight was ``lambda_ce + lambda_mask = 1.05``.
    """
    B = 4
    # Build V1/V2/V3 so their CE and margin terms are zero.
    perfect_supp = torch.tensor([[10.0, -10.0]] * B)   # class 0 logits
    perfect_con = torch.tensor([[-10.0, 10.0]] * B)    # class 1 logits
    v1 = _make_variant(perfect_supp, torch.tensor([0.99] * B), torch.zeros(B))
    v2 = _make_variant(perfect_con, torch.tensor([0.01] * B), torch.ones(B))
    v3 = _make_variant(perfect_con, torch.tensor([0.01] * B), torch.ones(B))

    # V4: make its CE meaningfully nonzero so the weighting is detectable.
    v4_wrong = _make_variant(
        torch.tensor([[1.0, 0.0]] * B),  # predicts class 0 but label 1 -> CE ~ log(1+e)
        torch.tensor([0.7] * B),
        torch.ones(B),
    )
    lm = 0.05
    loss_mod = FourWayContrastiveLoss(
        lambda_ce=1.0, lambda_evi=0.3, lambda_img=0.3, lambda_mask=lm,
    )
    out_with_v4 = loss_mod({
        VARIANT_POSITIVE: v1,
        VARIANT_EVIDENCE_SWAP: v2,
        VARIANT_IMAGE_SWAP: v3,
        VARIANT_IMAGE_MASK: v4_wrong,
    })
    out_without_v4 = loss_mod({
        VARIANT_POSITIVE: v1,
        VARIANT_EVIDENCE_SWAP: v2,
        VARIANT_IMAGE_SWAP: v3,
    })
    delta = out_with_v4["loss"] - out_without_v4["loss"]
    expected = lm * out_with_v4["mask_ce"]
    # Delta should equal lambda_mask * mask_ce, NOT (1 + lambda_mask) * mask_ce.
    assert torch.allclose(delta, expected, atol=1e-5), (
        f"V4 weight wrong: delta {delta.item()} vs expected {expected.item()}"
    )
