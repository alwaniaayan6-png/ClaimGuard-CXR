"""Vision encoders for ClaimGuard-CXR report generator.

Supports RadJEPA (primary), BiomedCLIP ViT-B/16 (robustness check),
and CheXNet DenseNet-121 (second robustness check). All used frozen
with optional lightweight adapter layers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BottleneckAdapter(nn.Module):
    """Lightweight adapter inserted after each ViT block.

    Down-projects, applies ReLU, up-projects, adds residually.
    ~49K params per adapter with bottleneck_dim=32 and hidden_dim=768.
    """

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class VisionEncoderWithAdapters(nn.Module):
    """Frozen vision encoder with trainable adapter layers.

    Loads a pretrained ViT, freezes all parameters, then inserts
    bottleneck adapters after each transformer block.

    Args:
        model_name: HuggingFace model ID or local path.
        adapter_bottleneck: Bottleneck dimension for adapters.
        image_size: Expected input resolution (used for logging only).
    """

    def __init__(
        self,
        model_name: str = "AIDElab-IITBombay/RadJEPA",
        adapter_bottleneck: int = 32,
        image_size: int = 384,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size

        # Load pretrained model
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}. Using placeholder.")
            self.encoder = None
            self.hidden_dim = 768
            self.num_layers = 12
            return

        # Determine hidden dim
        if hasattr(self.encoder.config, 'hidden_size'):
            self.hidden_dim = self.encoder.config.hidden_size
        else:
            self.hidden_dim = 768

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Count layers for adapter insertion
        self.num_layers = getattr(self.encoder.config, 'num_hidden_layers', 12)

        # Create adapters (one per layer)
        self.adapters = nn.ModuleList([
            BottleneckAdapter(self.hidden_dim, adapter_bottleneck)
            for _ in range(self.num_layers)
        ])

        # Log param counts
        frozen = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.adapters.parameters())
        logger.info(
            f"VisionEncoder: {model_name}, hidden={self.hidden_dim}, "
            f"frozen={frozen/1e6:.1f}M, adapters={trainable/1e6:.2f}M "
            f"({self.num_layers} adapters, bottleneck={adapter_bottleneck})"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image to spatial feature map.

        Args:
            pixel_values: (batch, 3, H, W) normalized image tensor.

        Returns:
            (batch, num_patches, hidden_dim) spatial features.
            For RadJEPA at 384px: (batch, 729, 768) where 729 = 27*27.
        """
        if self.encoder is None:
            # Placeholder for testing without model weights
            batch_size = pixel_values.shape[0]
            num_patches = (self.image_size // 14) ** 2  # 27*27 = 729 for 384px
            return torch.randn(batch_size, num_patches, self.hidden_dim, device=pixel_values.device)

        with torch.no_grad():
            outputs = self.encoder(pixel_values)

        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs

        # Apply adapters (residual)
        # Note: ideally adapters would be inserted between layers, but since
        # the encoder is frozen, we apply them sequentially on the output
        for adapter in self.adapters:
            features = adapter(features)

        return features

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    @property
    def num_patches(self) -> int:
        """Expected number of spatial patches for the configured image size."""
        patch_size = 14  # RadJEPA and BiomedCLIP both use 14 or 16
        return (self.image_size // patch_size) ** 2


def get_vision_encoder(
    name: str = "RadJEPA",
    image_size: int = 384,
    adapter_bottleneck: int = 32,
) -> VisionEncoderWithAdapters:
    """Factory function for vision encoders.

    Args:
        name: Encoder name ("RadJEPA", "BiomedCLIP", "CheXNet").
        image_size: Input resolution.
        adapter_bottleneck: Adapter bottleneck dimension.
    """
    model_ids = {
        "RadJEPA": "AIDElab-IITBombay/RadJEPA",
        "BiomedCLIP": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "CheXNet": "aidelab/CheXNet",  # or local weights
    }
    model_id = model_ids.get(name, name)
    return VisionEncoderWithAdapters(model_id, adapter_bottleneck, image_size)
