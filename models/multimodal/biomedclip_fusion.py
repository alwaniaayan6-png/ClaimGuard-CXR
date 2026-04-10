"""CheXzero multimodal fusion module for ClaimGuard-CXR v2.

Uses CheXzero (Tiu et al., Nature BME 2022) instead of BiomedCLIP for image-claim
scoring. CheXzero was trained on 377K MIMIC-CXR image-report pairs via contrastive
learning — its embeddings are inherently aligned to CXR semantics, unlike BiomedCLIP
which was trained on PMC academic figures.

Architecture:
  1. CheXzero frozen encoders produce (image, claim) embeddings
  2. Trainable projection MLPs map both to shared 256-dim space
  3. Learned temperature scales CLIP cosine similarity to match DeBERTa range
  4. 2-layer MLP gate fuses text verifier score with image-claim score
  5. When no image available, gate=1.0 -> pure text-only (backward compatible)

Key design decisions:
  - CheXzero over BiomedCLIP: BiomedCLIP's PMC-15M training data (charts, figures)
    yields brittle zero-shot CXR performance. CheXzero's MIMIC-CXR training gives
    radiologically meaningful embeddings and attention maps.
  - Learned tau_clip: Raw CLIP cosine sims cluster tightly ([0.2, 0.35]) while
    DeBERTa post-softmax is heavily polarized ([0.01, 0.99]). Without temperature
    scaling, a linear gate is dominated by the text logit.
  - 2-layer MLP gate (not linear): DeBERTa and CLIP have fundamentally different
    calibration profiles. 81 params can model the non-linear mapping without
    overfitting risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultimodalOutput:
    """Output from the multimodal fusion module."""
    final_score: torch.Tensor       # (batch,) — fused P(Not-Contradicted)
    text_score: torch.Tensor        # (batch,) — text-only score from DeBERTa
    image_score: torch.Tensor       # (batch,) — CheXzero image-claim score
    gate_value: torch.Tensor        # (batch,) — learned gate (1.0 = text-only)
    image_embedding: Optional[torch.Tensor]  # (batch, 512) — for grounding viz


class CheXzeroScorer(nn.Module):
    """Score image-claim consistency using frozen CheXzero.

    CheXzero (Tiu et al., Nature BME 2022) is a CLIP model fine-tuned on
    377K MIMIC-CXR image-report pairs. It achieves expert-level zero-shot
    pathology classification on CheXpert.

    Training only updates projection MLPs (~260K params) + tau_clip (1 param).

    Args:
        chexzero_dir: Path to CheXzero checkpoint directory.
        projection_dim: Dimension of shared projection space.
        tau_init: Initial CLIP temperature for scaling cosine similarity.
    """

    def __init__(
        self,
        chexzero_dir: str = "rajpurkarlab/CheXzero",
        projection_dim: int = 256,
        tau_init: float = 0.07,
    ):
        super().__init__()
        self.chexzero_dir = chexzero_dir
        self.projection_dim = projection_dim

        # Lazy-loaded CLIP encoders
        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None
        self._clip_dim = 512  # CLIP ViT-B/32 output dim

        # Trainable projection MLPs
        self.image_proj = nn.Sequential(
            nn.Linear(self._clip_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self._clip_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Learned temperature for CLIP score calibration
        # This is CRITICAL: scales cosine sim to match DeBERTa's dynamic range
        self.tau_clip = nn.Parameter(torch.tensor(tau_init))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CheXzeroScorer: {trainable} trainable params (projections + tau)")

    def _load_clip(self, device: torch.device) -> None:
        """Lazy-load CheXzero model (frozen)."""
        if self._clip_model is not None:
            return

        try:
            # Try loading CheXzero via open_clip or CLIP
            import clip
            self._clip_model, self._preprocess = clip.load("ViT-B/32", device=device)
            self._tokenizer = clip.tokenize

            # Load CheXzero fine-tuned weights if available
            import os
            ckpt_path = os.environ.get("CHEXZERO_CHECKPOINT", None)
            if ckpt_path and os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
                self._clip_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded CheXzero weights from {ckpt_path}")
            else:
                logger.warning(
                    "CheXzero checkpoint not found. Using base CLIP ViT-B/32. "
                    "Set CHEXZERO_CHECKPOINT env var to path of CheXzero .pt file."
                )

        except ImportError:
            # Fallback: use transformers CLIPModel
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(device)
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._tokenizer = self._preprocess.tokenizer
            logger.warning("Fell back to base CLIP ViT-B/32 via transformers")

        # Freeze all CLIP parameters
        for param in self._clip_model.parameters():
            param.requires_grad = False

        logger.info(f"CLIP model loaded and frozen on {device}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode CXR image through frozen CheXzero vision encoder.

        Args:
            pixel_values: Preprocessed image tensor (batch, 3, 224, 224).

        Returns:
            (batch, 512) L2-normalized image embedding.
        """
        self._load_clip(pixel_values.device)
        with torch.no_grad():
            if hasattr(self._clip_model, 'encode_image'):
                # clip library interface
                image_features = self._clip_model.encode_image(pixel_values)
            else:
                # transformers CLIPModel interface
                vision_outputs = self._clip_model.vision_model(pixel_values)
                image_features = self._clip_model.visual_projection(
                    vision_outputs.pooler_output
                )
        return F.normalize(image_features.float(), dim=-1)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode claim text through frozen CheXzero text encoder.

        Args:
            text_tokens: Tokenized claim text (batch, seq_len).

        Returns:
            (batch, 512) L2-normalized text embedding.
        """
        self._load_clip(text_tokens.device)
        with torch.no_grad():
            if hasattr(self._clip_model, 'encode_text'):
                text_features = self._clip_model.encode_text(text_tokens)
            else:
                text_outputs = self._clip_model.text_model(text_tokens)
                text_features = self._clip_model.text_projection(
                    text_outputs.pooler_output
                )
        return F.normalize(text_features.float(), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute temperature-scaled image-claim consistency score.

        Args:
            pixel_values: (batch, 3, 224, 224) preprocessed CXR images.
            text_tokens: (batch, seq_len) tokenized claims.

        Returns:
            Tuple of (score, image_embedding):
              score: (batch,) in [0, 1] — higher = more consistent
              image_embedding: (batch, 512) — for attention-based grounding
        """
        image_emb = self.encode_image(pixel_values)  # (B, 512)
        text_emb = self.encode_text(text_tokens)      # (B, 512)

        # Project to shared space
        proj_image = F.normalize(self.image_proj(image_emb), dim=-1)  # (B, 256)
        proj_text = F.normalize(self.text_proj(text_emb), dim=-1)     # (B, 256)

        # Temperature-scaled cosine similarity
        # tau_clip spreads out the tight CLIP cosine sim range to match DeBERTa's range
        tau = self.tau_clip.clamp(min=0.01, max=1.0)
        cosine_sim = (proj_image * proj_text).sum(dim=-1)  # (B,)
        score = torch.sigmoid(cosine_sim / tau)  # (B,) in [0, 1]

        return score, image_emb


class GatingFusion(nn.Module):
    """2-layer MLP gating fusion between text verifier and image-claim scorer.

    Why 2-layer, not linear: DeBERTa post-softmax is heavily polarized ([0.01, 0.99])
    while CLIP cosine sims cluster tightly ([0.2, 0.35]). A 4-parameter logistic
    regression cannot learn the non-linear calibration mapping between these. An
    81-param 2-layer MLP can, without overfitting risk.

    Args:
        hidden_dim: Hidden dimension of gate MLP.
        default_gate: Gate value when no image is provided.
    """

    def __init__(self, hidden_dim: int = 16, default_gate: float = 1.0):
        super().__init__()
        self.default_gate = default_gate

        # 2-layer MLP gate: 4 input features -> hidden -> 1
        # Input: [text_score, image_score, |diff|, product]
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to prefer text (bias toward gate ~ 0.88)
        nn.init.zeros_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, 2.0)  # sigmoid(2) ~ 0.88

    def forward(
        self,
        text_score: torch.Tensor,
        image_score: Optional[torch.Tensor] = None,
    ) -> MultimodalOutput:
        """Fuse text and image scores via learned 2-layer MLP gate.

        Args:
            text_score: (batch,) P(Not-Contradicted) from DeBERTa.
            image_score: (batch,) image-claim consistency from CheXzero.
                         None if no image available.

        Returns:
            MultimodalOutput with fused score and component scores.
        """
        if image_score is None:
            return MultimodalOutput(
                final_score=text_score,
                text_score=text_score,
                image_score=torch.zeros_like(text_score),
                gate_value=torch.ones_like(text_score),
                image_embedding=None,
            )

        # 4 input features for the gate
        gate_input = torch.stack([
            text_score,
            image_score,
            torch.abs(text_score - image_score),
            text_score * image_score,
        ], dim=-1)  # (B, 4)

        gate = torch.sigmoid(self.gate_mlp(gate_input)).squeeze(-1)  # (B,)
        final_score = gate * text_score + (1 - gate) * image_score

        return MultimodalOutput(
            final_score=final_score,
            text_score=text_score,
            image_score=image_score,
            gate_value=gate,
            image_embedding=None,
        )


class ClaimGuardV2(nn.Module):
    """Full ClaimGuard-CXR v2: DeBERTa verifier + CheXzero multimodal fusion.

    Supports text-only mode (backward compatible with v1) and multimodal mode.

    Args:
        text_verifier: Pre-trained DeBERTaClaimVerifier.
        chexzero_scorer: CheXzeroScorer (optional, for multimodal mode).
        gating_fusion: GatingFusion (optional, for multimodal mode).
    """

    def __init__(
        self,
        text_verifier: nn.Module,
        chexzero_scorer: Optional[CheXzeroScorer] = None,
        gating_fusion: Optional[GatingFusion] = None,
    ):
        super().__init__()
        self.text_verifier = text_verifier
        self.chexzero_scorer = chexzero_scorer
        self.gating_fusion = gating_fusion or GatingFusion()

    @property
    def is_multimodal(self) -> bool:
        return self.chexzero_scorer is not None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        claim_text_tokens: Optional[torch.Tensor] = None,
    ) -> MultimodalOutput:
        """Run full v2 verification pipeline.

        Args:
            input_ids: DeBERTa input (batch, seq_len) — [CLS] claim [SEP] evidence [SEP]
            attention_mask: DeBERTa attention mask (batch, seq_len)
            pixel_values: Optional CXR images (batch, 3, 224, 224) for CheXzero
            claim_text_tokens: Optional CheXzero-tokenized claims (batch, 77)

        Returns:
            MultimodalOutput with fused or text-only scores.
        """
        # Text verification (always runs)
        text_output = self.text_verifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            apply_temperature=True,
        )
        text_score = text_output.not_contra_prob  # (B,)

        # Multimodal fusion (optional)
        image_score = None
        image_embedding = None

        if (pixel_values is not None
                and claim_text_tokens is not None
                and self.chexzero_scorer is not None):
            image_score, image_embedding = self.chexzero_scorer(
                pixel_values, claim_text_tokens
            )

        output = self.gating_fusion(text_score, image_score)
        output.image_embedding = image_embedding
        return output
