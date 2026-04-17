"""Four-way contrastive loss for image-grounded claim verification.

See ARCHITECTURE_PATH_B.md Section 4.2. For each claim `c`, the training
batch contains four variants that force BOTH the text and image pathways
to carry signal:

  V1 (positive):     (c, evidence_supp,   image_correct)         label 0
  V2 (evidence-swap): (c, evidence_contra, image_correct)         label 1
  V3 (image-swap):   (c, evidence_supp,   image_random_patient)  label 1
  V4 (image-mask):   (c, evidence_supp,   zeros)                 label 1

HO baseline (text-only, evidence-masked) cannot distinguish V1 from V3
because the text is identical — therefore cannot satisfy the image
margin loss by construction. That is the HO-gap fix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


VARIANT_POSITIVE = "V1"
VARIANT_EVIDENCE_SWAP = "V2"
VARIANT_IMAGE_SWAP = "V3"
VARIANT_IMAGE_MASK = "V4"


@dataclass
class VariantBatch:
    """Container for one of {V1, V2, V3, V4} forward passes.

    verdict_logits : (B, 2)
    supported_prob : (B,)
    label          : (B,) — 0 (Not Contradicted) or 1 (Contradicted)
    """
    verdict_logits: torch.Tensor
    supported_prob: torch.Tensor
    label: torch.Tensor


class FourWayContrastiveLoss(nn.Module):
    """L = CE_all + λ_evi · margin(V1 − V2) + λ_img · margin(V1 − V3) + λ_mask · CE(V4).

    The CE term is summed across all four variants.
    The evidence margin forces the model to produce a higher supported-prob on V1
    than on V2 (same image, different evidence, different label).
    The image margin forces the model to produce a higher supported-prob on V1
    than on V3 (same text, different image, different label). This is the
    HO-gap fix — a text-only model sees identical text on V1 and V3.
    The mask term is a diagnostic weight on V4.
    """

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_evi: float = 0.3,
        lambda_img: float = 0.3,
        lambda_mask: float = 0.05,
        margin_evi: float = 0.2,
        margin_img: float = 0.2,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_evi = lambda_evi
        self.lambda_img = lambda_img
        self.lambda_mask = lambda_mask
        self.margin_evi = margin_evi
        self.margin_img = margin_img

    def forward(self, variants: Dict[str, VariantBatch]) -> Dict[str, torch.Tensor]:
        for key in (VARIANT_POSITIVE, VARIANT_EVIDENCE_SWAP, VARIANT_IMAGE_SWAP):
            if key not in variants:
                raise KeyError(
                    f"FourWayContrastiveLoss requires variant {key}; got {list(variants)}"
                )

        v1 = variants[VARIANT_POSITIVE]
        v2 = variants[VARIANT_EVIDENCE_SWAP]
        v3 = variants[VARIANT_IMAGE_SWAP]
        v4 = variants.get(VARIANT_IMAGE_MASK)

        # Main CE covers V1/V2/V3 only. V4 is a separate diagnostic term with its
        # own weight (lambda_mask); including V4 here would double-count it
        # (caught by v2 code review).
        ce_terms = [
            F.cross_entropy(v.verdict_logits, v.label.long(), reduction="mean")
            for v in (v1, v2, v3)
        ]
        ce = torch.stack(ce_terms).sum()

        # Hinge margin on supported-prob: positive variant should score higher.
        evi_margin = F.relu(self.margin_evi - (v1.supported_prob - v2.supported_prob)).mean()
        img_margin = F.relu(self.margin_img - (v1.supported_prob - v3.supported_prob)).mean()

        if v4 is not None:
            mask_ce = F.cross_entropy(v4.verdict_logits, v4.label.long(), reduction="mean")
        else:
            mask_ce = torch.zeros((), device=ce.device)

        total = (
            self.lambda_ce * ce
            + self.lambda_evi * evi_margin
            + self.lambda_img * img_margin
            + self.lambda_mask * mask_ce
        )

        return {
            "loss": total,
            "ce": ce.detach(),
            "evi_margin": evi_margin.detach(),
            "img_margin": img_margin.detach(),
            "mask_ce": mask_ce.detach(),
        }
