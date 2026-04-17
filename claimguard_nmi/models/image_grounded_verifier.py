"""Image-grounded claim verifier.

BiomedCLIP vision encoder + RoBERTa-large text encoder (shared for claim
and evidence) + cross-modal attention + per-claim ROI pooling + verdict /
score / uncertainty heads.

See ARCHITECTURE_PATH_B.md Section 4.1 for the architectural rationale
and Section 4.2 for the 4-way contrastive training objective.

Design constraints (v2):
  * The image pathway must be load-bearing. Training uses image-swap
    negatives (V3 in §4.2) so the HO baseline cannot match the full
    verifier's loss.
  * Region prior is soft, not hard. The ROI pooling module receives an
    additive attention bias from claim structure.
  * All Hugging Face loads pin ``revision`` to a specific commit hash
    (model.yaml) to prevent silent weight drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VerifierOutput:
    """Structured output of one forward pass through the image-grounded verifier."""
    verdict_logits: torch.Tensor   # (B, 2) — [not_contradicted, contradicted]
    supported_prob: torch.Tensor   # (B,)   — sigmoid of score head; for cfBH
    epistemic_var: Optional[torch.Tensor] = None  # (B,) — MC-dropout variance
    fused_features: Optional[torch.Tensor] = None  # (B, D) — fused representation, diagnostic

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.verdict_logits, dim=-1)

    @property
    def contradicted_prob(self) -> torch.Tensor:
        return self.probs[:, 1]


class _SharedRobertaEncoder(nn.Module):
    """RoBERTa-large used with shared weights for claim and evidence text."""

    def __init__(self, hf_name: str, hf_revision: str):
        super().__init__()
        from transformers import AutoModel  # local import avoids top-level cost

        self.backbone = AutoModel.from_pretrained(hf_name, revision=hf_revision)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over valid tokens.
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class _BiomedCLIPImageEncoder(nn.Module):
    """BiomedCLIP ViT-B/16 vision tower with last-N layers unfrozen."""

    def __init__(
        self,
        hf_name: str,
        hf_revision: str,
        unfrozen_last_n: int,
    ):
        super().__init__()
        from transformers import AutoModel

        full = AutoModel.from_pretrained(hf_name, revision=hf_revision, trust_remote_code=True)
        # BiomedCLIP's HF wrapper exposes `.vision_model`; fall back to `.visual` if needed.
        self.vision = getattr(full, "vision_model", None) or getattr(full, "visual", None)
        if self.vision is None:
            raise RuntimeError(
                "BiomedCLIP image encoder not found under .vision_model or .visual"
            )
        self._freeze_except_last_n(unfrozen_last_n)
        # Output feature dim (ViT-B/16 hidden).
        self.hidden_size = getattr(self.vision.config, "hidden_size", 768)

    def _freeze_except_last_n(self, n: int) -> None:
        # Freeze everything first.
        for p in self.vision.parameters():
            p.requires_grad = False
        # Unfreeze the last n transformer layers if we can locate them.
        attr_paths = ("encoder.layers", "transformer.resblocks", "encoder.layer")
        layers = None
        for attr in attr_paths:
            try:
                layers = self._rgetattr(self.vision, attr)
                break
            except AttributeError:
                continue
        if layers is None:
            # v2 review caught this as silent whole-freeze. Must not
            # silently leave every param frozen — training would appear
            # to work but all image grad would be zero.
            raise RuntimeError(
                f"BiomedCLIP image encoder transformer layers not located under "
                f"any of {attr_paths}. Cannot selectively unfreeze last {n} layers."
            )
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    @staticmethod
    def _rgetattr(obj, dotted):
        for key in dotted.split("."):
            obj = getattr(obj, key)
        return obj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch features, shape (B, N_patches, D)."""
        out = self.vision(pixel_values=pixel_values, return_dict=True)
        # Most ViTs expose .last_hidden_state with [CLS] + patches. Return patches only.
        hidden = out.last_hidden_state
        return hidden[:, 1:, :]  # drop [CLS]


class _CrossModalBlock(nn.Module):
    """One layer of bidirectional cross-attention between text and image."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.text_to_img = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.img_to_text = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_t = nn.LayerNorm(hidden_size)
        self.norm_i = nn.LayerNorm(hidden_size)
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm_tf = nn.LayerNorm(hidden_size)
        self.norm_if = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        image_attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # text: (B, Tt, D); image: (B, Ti, D); bias: (B, Ti) added to text->img attn logits.
        t_attn, _ = self.text_to_img(
            query=text, key=image, value=image,
            attn_mask=self._expand_bias(
                image_attn_bias, text.size(1), self.text_to_img.num_heads,
            ),
        )
        text = self.norm_t(text + t_attn)
        text = self.norm_tf(text + self.ffn_t(text))

        i_attn, _ = self.img_to_text(query=image, key=text, value=text)
        image = self.norm_i(image + i_attn)
        image = self.norm_if(image + self.ffn_i(image))

        return text, image

    @staticmethod
    def _expand_bias(
        bias: Optional[torch.Tensor], n_query: int, n_heads: int,
    ) -> Optional[torch.Tensor]:
        """Expand a per-key bias (B, Ti) to (B * n_heads, Tt, Ti).

        PyTorch ``nn.MultiheadAttention`` with ``batch_first=True`` expects
        ``attn_mask`` of shape ``(L, S)`` or ``(N * num_heads, L, S)``.
        We broadcast the per-key bias across query tokens and replicate
        across heads.
        """
        if bias is None:
            return None
        B, Ti = bias.shape
        # (B, 1, Ti) -> (B, n_query, Ti) -> repeat for each head -> (B*n_heads, n_query, Ti)
        expanded = bias.unsqueeze(1).expand(B, n_query, Ti).contiguous()
        return expanded.repeat_interleave(n_heads, dim=0)


class ImageGroundedVerifier(nn.Module):
    """End-to-end image-grounded claim verifier.

    Forward signature:
        forward(
            claim_ids, claim_mask,
            evidence_ids, evidence_mask,
            pixel_values,
            region_attn_bias=None,           # (B, N_patches); soft prior over image patches
        ) -> VerifierOutput
    """

    def __init__(
        self,
        text_encoder: _SharedRobertaEncoder,
        image_encoder: _BiomedCLIPImageEncoder,
        cross_attn_hidden: int = 1024,
        cross_attn_layers: int = 4,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        mc_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Project image patches to cross-attention hidden size.
        self.image_proj = nn.Linear(image_encoder.hidden_size, cross_attn_hidden)
        # Project text features to cross-attention hidden size if needed.
        # Built unconditionally in __init__ — lazy construction inside forward
        # would miss optimizer registration and break state_dict round-trips
        # (flagged by v2 review).
        if text_encoder.hidden_size == cross_attn_hidden:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(text_encoder.hidden_size, cross_attn_hidden)

        self.cross_blocks = nn.ModuleList(
            [
                _CrossModalBlock(cross_attn_hidden, cross_attn_heads, dropout)
                for _ in range(cross_attn_layers)
            ]
        )

        # Fusion: concat mean-pooled text (claim + evidence) with mean-pooled image.
        fused_dim = cross_attn_hidden * 3  # claim, evidence, image
        self.verdict_head = nn.Sequential(
            nn.Linear(fused_dim, cross_attn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_attn_hidden, 2),
        )
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, cross_attn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_attn_hidden, 1),
        )
        self.mc_dropout = nn.Dropout(mc_dropout_rate)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _encode_text(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, mask)

    def forward(
        self,
        claim_ids: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_ids: torch.Tensor,
        evidence_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        region_attn_bias: Optional[torch.Tensor] = None,
        enable_mc_dropout: bool = False,
        n_mc_samples: int = 1,
    ) -> VerifierOutput:
        claim_vec = self._encode_text(claim_ids, claim_mask)        # (B, D_text)
        evi_vec = self._encode_text(evidence_ids, evidence_mask)    # (B, D_text)

        img_patches = self.image_encoder(pixel_values)              # (B, N, D_img)
        img_patches = self.image_proj(img_patches)                  # (B, N, D_fused)

        # Build a short text sequence from (claim, evidence) pooled vectors, to drive
        # cross-attention. We use 2 "virtual tokens" per sample.
        text_seq = torch.stack([claim_vec, evi_vec], dim=1)         # (B, 2, D_text)
        text_seq = self.text_proj(text_seq)                          # (B, 2, D_fused)

        # Cross-modal attention stack.
        t, i = text_seq, img_patches
        for block in self.cross_blocks:
            t, i = block(t, i, image_attn_bias=region_attn_bias)

        claim_out, evi_out = t[:, 0, :], t[:, 1, :]
        img_out = i.mean(dim=1)
        fused = torch.cat([claim_out, evi_out, img_out], dim=-1)     # (B, 3*D)

        if enable_mc_dropout and n_mc_samples > 1:
            verdict_logits_stack = []
            score_stack = []
            for _ in range(n_mc_samples):
                f = self.mc_dropout(fused)
                verdict_logits_stack.append(self.verdict_head(f))
                score_stack.append(torch.sigmoid(self.score_head(f).squeeze(-1)))
            verdict_logits = torch.stack(verdict_logits_stack, dim=0).mean(dim=0)
            score_stack_t = torch.stack(score_stack, dim=0)
            supported_prob = score_stack_t.mean(dim=0)
            epistemic_var = score_stack_t.var(dim=0)
        else:
            verdict_logits = self.verdict_head(fused)
            supported_prob = torch.sigmoid(self.score_head(fused).squeeze(-1))
            epistemic_var = None

        return VerifierOutput(
            verdict_logits=verdict_logits,
            supported_prob=supported_prob,
            epistemic_var=epistemic_var,
            fused_features=fused.detach(),
        )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def build_image_grounded_verifier(cfg: dict) -> ImageGroundedVerifier:
    """Build the verifier from a parsed model.yaml dict.

    Does NOT load any checkpoint — for that, use a dedicated loader that
    verifies state-dict keys exactly (do not use strict=False; see the D21
    incident in the v1 codebase).
    """
    te_cfg = cfg["model"]["text_encoder"]
    ie_cfg = cfg["model"]["image_encoder"]
    ca_cfg = cfg["model"]["cross_attn"]
    heads_cfg = cfg["model"]["heads"]

    text_enc = _SharedRobertaEncoder(
        hf_name=te_cfg["hf_name"],
        hf_revision=te_cfg["hf_revision"],
    )
    image_enc = _BiomedCLIPImageEncoder(
        hf_name=ie_cfg["hf_name"],
        hf_revision=ie_cfg["hf_revision"],
        unfrozen_last_n=ie_cfg["unfrozen_last_n_layers"],
    )
    return ImageGroundedVerifier(
        text_encoder=text_enc,
        image_encoder=image_enc,
        cross_attn_hidden=ca_cfg["hidden_size"],
        cross_attn_layers=ca_cfg["n_layers"],
        cross_attn_heads=ca_cfg["n_heads"],
        dropout=ca_cfg["dropout"],
        mc_dropout_rate=heads_cfg["uncertainty"]["dropout_rate"],
    )
