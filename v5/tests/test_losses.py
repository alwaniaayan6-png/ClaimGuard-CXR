"""Sanity tests for the multi-objective loss."""

from __future__ import annotations

import torch

from v5.losses import LossWeights, total_loss


def test_cls_only():
    B = 8
    logits = torch.randn(B, 2)
    labels = torch.randint(0, 2, (B,))
    out = total_loss(
        weights=LossWeights(cls=1.0, ground=0.0, consist=0.0, contrast=0.0, uncert=0.0),
        verdict_logits_full=logits,
        verdict_logits_masked=None,
        support_score_matched=torch.sigmoid(logits[:, 0]),
        support_score_mismatched=None,
        labels=labels,
        grounding_logits=None,
        grounding_target=None,
        grounding_mask=None,
    )
    assert out.total.item() >= 0.0
    assert out.ground.item() == 0.0
    assert out.consist.item() == 0.0
    assert out.contrast.item() == 0.0


def test_consistency_prefers_full_over_masked():
    B = 16
    labels = torch.zeros(B, dtype=torch.long)
    # full model: high P(class 0); masked model: uniform
    full_logits = torch.zeros(B, 2)
    full_logits[:, 0] = 3.0
    masked_logits = torch.zeros(B, 2)
    out = total_loss(
        weights=LossWeights(cls=0.0, ground=0.0, consist=1.0, contrast=0.0, uncert=0.0),
        verdict_logits_full=full_logits,
        verdict_logits_masked=masked_logits,
        support_score_matched=torch.sigmoid(full_logits[:, 0]),
        support_score_mismatched=None,
        labels=labels,
        grounding_logits=None,
        grounding_target=None,
        grounding_mask=None,
    )
    # Full is confident correct, masked is uniform → margin is large (~0.5),
    # exceeding the relu threshold of 0.2, so loss should be near zero.
    assert out.consist.item() <= 0.2


def test_contrast_prefers_matched_over_mismatched():
    B = 8
    matched = torch.full((B,), 0.9)
    mismatched = torch.full((B,), 0.3)
    out = total_loss(
        weights=LossWeights(cls=0.0, ground=0.0, consist=0.0, contrast=1.0, uncert=0.0, contrast_margin=0.2),
        verdict_logits_full=torch.zeros(B, 2),
        verdict_logits_masked=None,
        support_score_matched=matched,
        support_score_mismatched=mismatched,
        labels=torch.zeros(B, dtype=torch.long),
        grounding_logits=None,
        grounding_target=None,
        grounding_mask=None,
    )
    # matched - mismatched = 0.6 > margin 0.2 → loss should be 0
    assert out.contrast.item() == 0.0
