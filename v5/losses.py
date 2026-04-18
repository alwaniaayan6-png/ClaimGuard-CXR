"""Multi-objective loss for ClaimGuard-CXR v5.

See ARCHITECTURE_V5_IMAGE_GROUNDED.md §6.1. Total loss:

    L = L_cls
      + lambda_g * L_ground    (only when grounding_target is supplied)
      + lambda_c * L_consist   (KL between full and image-masked forward)
      + lambda_e * L_contrast  (margin between matched and mismatched evidence)
      + lambda_u * L_uncert    (MC-dropout calibration regularizer)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    cls: float = 1.0
    ground: float = 0.5
    consist: float = 0.3
    contrast: float = 0.3
    uncert: float = 0.1
    contrast_margin: float = 0.2


def classification_loss(verdict_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over {not-contradicted=0, contradicted=1}."""
    return F.cross_entropy(verdict_logits, labels.long())


def grounding_loss(
    grounding_logits: torch.Tensor,
    grounding_target: torch.Tensor,
    grounding_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over a 14x14 grounding map.

    Args:
        grounding_logits: (B, H, W) logits.
        grounding_target: (B, H, W) in {0, 1}.
        grounding_mask:   (B,) bool, 1 where grounding supervision is available.
    """
    if grounding_mask.sum() == 0:
        return torch.zeros((), device=grounding_logits.device)
    sel = grounding_mask.bool()
    logits = grounding_logits[sel]
    target = grounding_target[sel].float()
    return F.binary_cross_entropy_with_logits(logits, target)


def consistency_loss(
    verdict_logits_full: torch.Tensor,
    verdict_logits_masked: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Encourage image-masked output to *diverge* from full output.

    The spec in §6.1 says L_consist should encourage use of the image. We
    implement this as a *reverse* KL: we want the full model to be confident
    and the image-masked model to be uncertain or wrong, so minimizing
    -KL(p_full || p_masked) is wrong. Instead we maximize the label-margin
    between full and masked.

    L_consist = -[p_full(correct) - p_masked(correct)]
    """
    p_full = F.softmax(verdict_logits_full, dim=-1)
    p_masked = F.softmax(verdict_logits_masked, dim=-1)
    idx = labels.long().unsqueeze(-1)
    p_full_correct = p_full.gather(1, idx).squeeze(-1)
    p_masked_correct = p_masked.gather(1, idx).squeeze(-1)
    margin = p_full_correct - p_masked_correct
    # negate so that minimizing the loss maximizes the margin; clamp to keep
    # well-separated examples from dominating the gradient
    return F.relu(0.2 - margin).mean()


def contrastive_evidence_loss(
    score_matched: torch.Tensor,
    score_mismatched: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Margin loss: supported claim should score higher with matching evidence
    than with mismatched evidence.

    Args:
        score_matched: (B,) sigmoid support probabilities with matching evidence.
        score_mismatched: (B,) with intentionally wrong evidence.
    """
    return F.relu(margin - (score_matched - score_mismatched)).mean()


def uncertainty_regularizer(
    mc_probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Bin-free MC-dropout ECE.

    Args:
        mc_probs: (S, B, C) softmax probabilities from S forward passes of the
            full model with dropout modules in training mode. The caller MUST
            obtain these via repeated forwards — applying Dropout to
            pre-computed logits is NOT MC-dropout and would defeat the purpose.
        labels: (B,) ground-truth class indices.

    Returns:
        Mean |confidence - correctness| over the batch.
    """
    if mc_probs.dim() != 3:
        raise ValueError(
            f"uncertainty_regularizer expects mc_probs of shape (S,B,C); got {tuple(mc_probs.shape)}"
        )
    mean_p = mc_probs.mean(dim=0)
    conf, pred = mean_p.max(dim=-1)
    correct = (pred == labels).float()
    return (conf - correct).abs().mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    ground: torch.Tensor
    consist: torch.Tensor
    contrast: torch.Tensor
    uncert: torch.Tensor


def total_loss(
    *,
    weights: LossWeights,
    verdict_logits_full: torch.Tensor,
    verdict_logits_masked: torch.Tensor | None,
    support_score_matched: torch.Tensor,
    support_score_mismatched: torch.Tensor | None,
    labels: torch.Tensor,
    grounding_logits: torch.Tensor | None,
    grounding_target: torch.Tensor | None,
    grounding_mask: torch.Tensor | None,
    uncertainty_mc_probs: torch.Tensor | None = None,
    example_weights: torch.Tensor | None = None,
) -> LossOutput:
    """Combined loss. `example_weights` (B,) applies per-example downweighting
    (from the adversarial HO filter) to the classification term only.
    """
    device = verdict_logits_full.device
    zero = torch.zeros((), device=device)

    if example_weights is None:
        cls = classification_loss(verdict_logits_full, labels)
    else:
        per_example = F.cross_entropy(verdict_logits_full, labels.long(), reduction="none")
        cls = (per_example * example_weights).mean()

    if (
        weights.ground > 0
        and grounding_logits is not None
        and grounding_target is not None
        and grounding_mask is not None
    ):
        ground = grounding_loss(grounding_logits, grounding_target, grounding_mask)
    else:
        ground = zero

    if weights.consist > 0 and verdict_logits_masked is not None:
        consist = consistency_loss(verdict_logits_full, verdict_logits_masked, labels)
    else:
        consist = zero

    if weights.contrast > 0 and support_score_mismatched is not None:
        contrast = contrastive_evidence_loss(
            support_score_matched,
            support_score_mismatched,
            weights.contrast_margin,
        )
    else:
        contrast = zero

    if weights.uncert > 0 and uncertainty_mc_probs is not None:
        uncert = uncertainty_regularizer(uncertainty_mc_probs, labels)
    else:
        uncert = zero

    total = (
        weights.cls * cls
        + weights.ground * ground
        + weights.consist * consist
        + weights.contrast * contrast
        + weights.uncert * uncert
    )
    return LossOutput(total=total, cls=cls, ground=ground, consist=consist, contrast=contrast, uncert=uncert)
