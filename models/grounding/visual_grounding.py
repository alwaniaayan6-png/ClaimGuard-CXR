"""
Visual evidence grounding module for ClaimGuard-CXR.

Cross-attention between PubMedBERT claim embeddings and RadJEPA spatial
features produces a 27x27 heatmap per claim that localizes the image region
supporting (or refuting) each claim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualGroundingModule(nn.Module):
    """Grounds natural-language claims onto CXR spatial feature maps.

    Takes a claim embedding produced by PubMedBERT and a spatial feature
    tensor from RadJEPA (27x27 patch grid at 384 px input) and returns a
    per-patch attention heatmap.

    Args:
        embed_dim: Dimensionality shared by claim and spatial features (768).
        num_heads: Number of cross-attention heads (4).
        dropout: Dropout probability applied inside attention (0.0).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 192

        # Project claim embedding into the shared dim (768 -> 768)
        self.claim_proj = nn.Linear(embed_dim, embed_dim)

        # Q / K / V projections for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Map attended features down to a scalar logit per spatial token
        self.heatmap_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, L, D) -> (B, H, L, head_dim)."""
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, L, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, L, head_dim) -> (B, L, D)."""
        B, H, L, hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, H * hd)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        claim_embeddings: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run cross-attention and produce 27x27 heatmaps.

        Args:
            claim_embeddings: PubMedBERT [CLS] embeddings, shape (B, 768).
            spatial_features: RadJEPA patch tokens, shape (B, 729, 768).
                729 = 27 * 27 patches at 384 px input with patch size 14.

        Returns:
            heatmaps: Sigmoid-activated grounding maps, shape (B, 27, 27).
        """
        B, num_patches, _ = spatial_features.shape  # (B, 729, 768)

        # Project claim embedding and add sequence dimension -> (B, 1, 768)
        claim_proj = self.claim_proj(claim_embeddings).unsqueeze(1)

        # Compute Q from claim (1 query token), K/V from spatial tokens
        Q = self._split_heads(self.q_proj(claim_proj))           # (B, H, 1, hd)
        K = self._split_heads(self.k_proj(spatial_features))     # (B, H, 729, hd)
        V = self._split_heads(self.v_proj(spatial_features))     # (B, H, 729, hd)

        # Scaled dot-product attention: query over all spatial tokens
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, 1, 729)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attended context: (B, H, 1, hd) -> (B, 1, 768)
        context = torch.matmul(attn_weights, V)                  # (B, H, 1, hd)
        context = self._merge_heads(context)                     # (B, 1, 768)
        context = self.out_proj(context)                         # (B, 1, 768)

        # Broadcast attended context to each spatial position and combine
        # with spatial features to produce per-token heatmap logits
        context_expanded = context.expand(B, num_patches, self.embed_dim)  # (B, 729, 768)
        combined = spatial_features + context_expanded            # residual blend

        # Collapse to scalar per patch then sigmoid
        logits = self.heatmap_head(combined).squeeze(-1)          # (B, 729)
        heatmap_flat = torch.sigmoid(logits)                      # (B, 729)

        # Reshape to 2-D spatial grid
        heatmaps = heatmap_flat.view(B, 27, 27)                   # (B, 27, 27)
        return heatmaps

    def get_heatmap_384(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Upsample 27x27 heatmaps to 384x384 for visualization.

        Uses bilinear interpolation with ``align_corners=False`` to match
        the coordinate conventions of torchvision transforms.

        Args:
            heatmaps: Grounding maps from :meth:`forward`, shape (B, 27, 27).

        Returns:
            Upsampled heatmaps of shape (B, 384, 384), values in [0, 1].
        """
        # F.interpolate expects (B, C, H, W)
        x = heatmaps.unsqueeze(1)                                 # (B, 1, 27, 27)
        x = F.interpolate(
            x,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(1)                                        # (B, 384, 384)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def grounding_loss(
    predicted_heatmaps: torch.Tensor,
    target_masks: torch.Tensor,
    iou_weight: float = 0.1,
) -> torch.Tensor:
    """Combined grounding supervision loss: BCE + IoU regularization.

    The BCE term provides dense pixel-level supervision.  The IoU term
    penalizes poor spatial overlap with the ground-truth mask, which helps
    when the positive region is small relative to the full image.

    Args:
        predicted_heatmaps: Model output *before* sigmoid if used standalone,
            but here expected as already-sigmoidified values from
            :meth:`VisualGroundingModule.forward`, shape (B, 27, 27).
            Values must be in [0, 1].
        target_masks: Binary ground-truth masks downsampled to 27x27,
            shape (B, 27, 27).  Values should be 0 or 1.
        iou_weight: Scalar weight for the IoU regularization term (default 0.1).

    Returns:
        Scalar loss tensor suitable for ``loss.backward()``.
    """
    # Flatten spatial dims for easier computation
    pred_flat = predicted_heatmaps.view(predicted_heatmaps.size(0), -1)   # (B, 729)
    tgt_flat = target_masks.view(target_masks.size(0), -1).float()        # (B, 729)

    # Binary cross-entropy (predicted values already in [0,1])
    bce = F.binary_cross_entropy(pred_flat, tgt_flat, reduction="mean")

    # Soft IoU regularization
    # Numerator:  sum of element-wise product (soft intersection)
    # Denominator: sum of union (pred + target - intersection)
    intersection = (pred_flat * tgt_flat).sum(dim=1)                      # (B,)
    union = (pred_flat + tgt_flat - pred_flat * tgt_flat).sum(dim=1)      # (B,)
    iou = intersection / (union + 1e-6)                                   # (B,)
    iou_reg = 1.0 - iou.mean()                                            # scalar

    return bce + iou_weight * iou_reg
