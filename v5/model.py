"""ClaimGuard-CXR v5 image-grounded claim verifier.

Architecture (see ARCHITECTURE_V5_IMAGE_GROUNDED.md §5):

    image (224x224x3)  ──► BiomedCLIP ViT-B/16 ──► 196 + 1 patch/CLS tokens, d=768
    claim + evidence   ──► RoBERTa-large           ──► up to 256 tokens,      d=1024
                                                           │
                                       each projected ──► d=768 shared space
                                                           │
                                  prepend learnable [VERDICT] token
                                                           │
                     4-layer bidirectional cross-modal transformer (d=768, heads=12)
                                                           │
                        ┌──────────┬──────────┬──────────┬──────────┐
                        │ verdict  │ score    │ grounding│ uncertainty
                        │ head     │ head     │ head     │ (MC-drop)
                        └──────────┴──────────┴──────────┴──────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class V5Config:
    """Configuration for ImageGroundedVerifier."""

    image_backbone: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    image_backbone_revision: str | None = None
    text_backbone: str = "roberta-large"
    text_backbone_revision: str | None = None
    shared_dim: int = 768
    fusion_layers: int = 4
    fusion_heads: int = 12
    fusion_ffn_dim: int = 3072
    fusion_dropout: float = 0.1
    mc_dropout_p: float = 0.2
    num_verdict_classes: int = 2
    image_patches_side: int = 14  # 224 / 16
    max_text_tokens: int = 256
    freeze_image_layers: int = 8  # of 12 in ViT-B
    freeze_text_layers: int = 16  # of 24 in RoBERTa-large
    grounding_enabled: bool = True
    uncertainty_samples: int = 5  # MC-dropout passes at eval


class _DomainAdapter(nn.Module):
    """2-layer MLP domain adapter: PMC-OA → CXR.

    Applied to BiomedCLIP patch tokens. Trained from scratch via MAE
    objective on CheXpert Plus images before downstream training.
    """

    def __init__(self, dim: int = 768, hidden: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual: never lose the pretrained signal


class _CrossModalBlock(nn.Module):
    """One transformer-encoder block over concatenated image+text tokens."""

    def __init__(self, cfg: V5Config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.shared_dim,
            num_heads=cfg.fusion_heads,
            dropout=cfg.fusion_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.shared_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.shared_dim, cfg.fusion_ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_ffn_dim, cfg.shared_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.shared_dim)
        self.dropout = nn.Dropout(cfg.fusion_dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        attn, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class ImageGroundedVerifier(nn.Module):
    """v5 verifier returning verdict / score / grounding / uncertainty."""

    def __init__(self, cfg: V5Config):
        super().__init__()
        self.cfg = cfg

        # --- image backbone ------------------------------------------------
        self.image_encoder = AutoModel.from_pretrained(
            cfg.image_backbone,
            revision=cfg.image_backbone_revision,
            trust_remote_code=True,
        )
        img_hidden = getattr(self.image_encoder.config, "hidden_size", None) or 768
        self._freeze_image(cfg.freeze_image_layers)
        self.image_domain_adapter = _DomainAdapter(img_hidden, hidden=img_hidden)
        self.image_proj = nn.Linear(img_hidden, cfg.shared_dim)

        # --- text backbone -------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(
            cfg.text_backbone,
            revision=cfg.text_backbone_revision,
        )
        txt_hidden = self.text_encoder.config.hidden_size
        self._freeze_text(cfg.freeze_text_layers)
        self.text_proj = nn.Linear(txt_hidden, cfg.shared_dim)

        # --- fusion --------------------------------------------------------
        self.verdict_token = nn.Parameter(torch.zeros(1, 1, cfg.shared_dim))
        nn.init.normal_(self.verdict_token, std=0.02)
        self.fusion_blocks = nn.ModuleList(
            [_CrossModalBlock(cfg) for _ in range(cfg.fusion_layers)]
        )
        self.fusion_norm = nn.LayerNorm(cfg.shared_dim)

        # --- heads ---------------------------------------------------------
        self.verdict_head = nn.Sequential(
            nn.Dropout(cfg.mc_dropout_p),
            nn.Linear(cfg.shared_dim, cfg.num_verdict_classes),
        )
        self.score_head = nn.Sequential(
            nn.Dropout(cfg.mc_dropout_p),
            nn.Linear(cfg.shared_dim, 1),
        )
        self.grounding_head = nn.Linear(cfg.shared_dim, 1)

    # ------------------------------------------------------------------ utils

    def _freeze_image(self, n_freeze: int) -> None:
        """Freeze the first n_freeze transformer blocks of the ViT.

        If the encoder structure is unrecognized, we RAISE rather than silently
        freezing the entire encoder (which would prevent the grounding head
        from learning). Any new backbone must add its attribute path here.
        """
        # Candidate attribute paths across HF/open_clip/timm ViT variants.
        # Ordered roughly from most-specific to most-generic.
        candidate_paths = (
            "encoder.layer",                 # HF BERT-style / some ViT
            "vision_model.encoder.layers",   # CLIPVisionModel
            "visual.trunk.blocks",           # open_clip timm-backed
            "visual.transformer.resblocks",  # open_clip classic
            "vision_model.trunk.blocks",     # BiomedCLIP HF wrapper
            "trunk.blocks",                  # bare timm ViT
            "blocks",                        # raw timm
            "encoder.layers",                # some variants
            "vision_model.vision_model.encoder.layers",  # double-wrapped
        )
        blocks = None
        matched_path: str | None = None
        for path in candidate_paths:
            obj: Any = self.image_encoder
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                # Require it to be indexable (list/Sequential/ModuleList)
                _ = len(obj)
                blocks = obj
                matched_path = path
                break
            except (AttributeError, TypeError):
                continue

        if blocks is None:
            # List top-level attributes to help future debugging.
            top = [a for a in dir(self.image_encoder) if not a.startswith("_")][:30]
            raise RuntimeError(
                "Could not locate image encoder transformer blocks; refusing to "
                "fall through to whole-encoder freeze (that would silently disable "
                "grounding). Add the correct attribute path to _freeze_image. "
                f"Top-level attrs on encoder: {top}"
            )

        total = len(blocks)
        n_freeze = min(n_freeze, total)
        for i, block in enumerate(blocks):
            frozen = i < n_freeze
            for p in block.parameters():
                p.requires_grad = not frozen
        logger.info(
            "froze first %d of %d image encoder blocks (path=%s)",
            n_freeze, total, matched_path,
        )

    def _freeze_text(self, n_freeze: int) -> None:
        for name, p in self.text_encoder.named_parameters():
            p.requires_grad = True
        layers = self.text_encoder.encoder.layer
        n_freeze = min(n_freeze, len(layers))
        for i, block in enumerate(layers):
            if i < n_freeze:
                for p in block.parameters():
                    p.requires_grad = False
        # Always freeze the embeddings (too many params for a small task; unfrozen text
        # encoders destabilize cross-modal training).
        for p in self.text_encoder.embeddings.parameters():
            p.requires_grad = False
        logger.info("Froze first %d of %d text encoder blocks", n_freeze, len(layers))

    # ------------------------------------------------------------- encoders

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return (B, 1 + P, shared_dim) with CLS followed by patch tokens."""
        outputs = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs.get("last_hidden_state", None)
            if tokens is None:
                raise RuntimeError("image encoder did not return last_hidden_state")
        tokens = self.image_domain_adapter(tokens)
        return self.image_proj(tokens)

    def encode_image_masked(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Zero-image baseline used for the image-masked consistency loss."""
        zeros = torch.zeros_like(pixel_values)
        return self.encode_image(zeros)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        tokens = self.text_proj(outputs.last_hidden_state)
        return tokens, attention_mask.bool()

    # --------------------------------------------------------------- forward

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        image_masked: bool = False,
        return_grounding: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = input_ids.size(0)

        if image_masked:
            img_tokens = self.encode_image_masked(pixel_values)
        else:
            img_tokens = self.encode_image(pixel_values)  # (B, 1+P, d)
        txt_tokens, txt_mask = self.encode_text(input_ids, attention_mask)  # (B, T, d)

        verdict_tok = self.verdict_token.expand(B, -1, -1)  # (B, 1, d)
        seq = torch.cat([verdict_tok, img_tokens, txt_tokens], dim=1)

        # Build key_padding_mask (True = pad, False = real). Image tokens are never pad;
        # verdict token is never pad; text tokens obey attention_mask.
        img_pad = torch.zeros(B, img_tokens.size(1), dtype=torch.bool, device=seq.device)
        verdict_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        # nn.MultiheadAttention uses True = pad; ours uses True = real, so invert.
        txt_pad = ~txt_mask
        pad_mask = torch.cat([verdict_pad, img_pad, txt_pad], dim=1)

        for block in self.fusion_blocks:
            seq = block(seq, key_padding_mask=pad_mask)
        seq = self.fusion_norm(seq)

        verdict_feat = seq[:, 0]  # (B, d)
        verdict_logits = self.verdict_head(verdict_feat)  # (B, 2)
        support_score = torch.sigmoid(self.score_head(verdict_feat)).squeeze(-1)  # (B,)

        out: dict[str, torch.Tensor] = {
            "verdict_logits": verdict_logits,
            "support_score": support_score,
        }

        if return_grounding and self.cfg.grounding_enabled:
            # Patch tokens only, drop CLS (which is position 0 of img_tokens ⇒ seq[:, 1]).
            n_img = img_tokens.size(1)
            img_out = seq[:, 1 : 1 + n_img]  # (B, 1+P, d)
            patch_out = img_out[:, 1:]  # drop CLS, keep P=196
            grounding_logits = self.grounding_head(patch_out).squeeze(-1)  # (B, P)
            side = self.cfg.image_patches_side
            out["grounding_logits"] = grounding_logits.view(B, side, side)

        return out

    # --------------------------------------------------------- uncertainty

    def predict_with_uncertainty(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """MC-dropout ensemble. Keeps dropout active during sampling."""
        n_samples = n_samples or self.cfg.uncertainty_samples
        was_training = self.training
        self.train()  # enable dropout
        probs = []
        try:
            with torch.no_grad():
                for _ in range(n_samples):
                    out = self.forward(pixel_values, input_ids, attention_mask)
                    probs.append(torch.softmax(out["verdict_logits"], dim=-1))
        finally:
            self.train(was_training)
        probs_stack = torch.stack(probs, dim=0)  # (S, B, C)
        mean = probs_stack.mean(dim=0)
        # predictive entropy
        entropy = -(mean.clamp_min(1e-9).log() * mean).sum(dim=-1)
        # epistemic (mutual information): entropy of mean - mean of entropy
        sample_entropy = -(probs_stack.clamp_min(1e-9).log() * probs_stack).sum(dim=-1).mean(dim=0)
        epistemic = entropy - sample_entropy
        return {
            "mean_probs": mean,
            "predictive_entropy": entropy,
            "epistemic_uncertainty": epistemic,
        }


def build_v5_tokenizer(cfg: V5Config) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        cfg.text_backbone,
        revision=cfg.text_backbone_revision,
        use_fast=True,
    )


def build_v5_model(cfg: V5Config | None = None) -> ImageGroundedVerifier:
    return ImageGroundedVerifier(cfg or V5Config())
