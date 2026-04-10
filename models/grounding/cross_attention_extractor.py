"""Extract grounding heatmaps from the generator's cross-attention layers.

Instead of training a separate grounding module (which requires PhysioNet
data like ChestImagenome), we extract attention maps directly from the
generator's cross-attention between text tokens and image patches.

This is FREE — no extra data, no extra training. The generator already
learns to attend to relevant image regions when generating report text.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def extract_cross_attention_heatmap(
    model,
    vision_features: torch.Tensor,
    claim_text: str,
    tokenizer,
    spatial_size: int = 27,
    layer_indices: Optional[list[int]] = None,
    aggregate: str = "mean",
) -> np.ndarray:
    """Extract a grounding heatmap from generator cross-attention.

    Runs the claim text through the decoder with vision features,
    captures cross-attention weights, and aggregates them into a
    spatial heatmap showing which image regions the model attends to.

    Args:
        model: The generator decoder (Phi-3 with cross-attention).
        vision_features: (1, n_patches, hidden_dim) from vision encoder.
        claim_text: The claim to ground.
        tokenizer: Decoder tokenizer.
        spatial_size: Spatial dimension of the heatmap (27 for 384px/14px patches).
        layer_indices: Which cross-attention layers to use. None = all.
        aggregate: How to combine layers/heads — "mean", "max", or "last".

    Returns:
        (spatial_size, spatial_size) numpy array with attention values in [0, 1].
    """
    model.eval()
    device = vision_features.device

    # Tokenize the claim
    inputs = tokenizer(claim_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)

    # Get text embeddings
    text_embeds = model.get_input_embeddings()(input_ids)

    # If there's a vision projection, project vision features
    if hasattr(model, 'vision_proj'):
        vis_projected = model.vision_proj(vision_features)
    else:
        vis_projected = vision_features

    # Concatenate vision + text embeddings
    combined = torch.cat([vis_projected, text_embeds], dim=1)
    n_vision_tokens = vis_projected.shape[1]

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            inputs_embeds=combined,
            output_attentions=True,
        )

    # Extract attention weights
    # attentions is a tuple of (n_layers,) each shape (batch, n_heads, seq_len, seq_len)
    attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

    if attentions is None:
        logger.warning("Model did not return attention weights. Using uniform heatmap.")
        return np.ones((spatial_size, spatial_size)) / (spatial_size * spatial_size)

    # Select layers
    if layer_indices is not None:
        selected = [attentions[i] for i in layer_indices if i < len(attentions)]
    else:
        selected = list(attentions)

    if not selected:
        return np.ones((spatial_size, spatial_size)) / (spatial_size * spatial_size)

    # For each attention layer: extract text-to-vision attention
    # Shape of each: (1, n_heads, seq_len, seq_len)
    # We want: text tokens attending to vision tokens
    # text tokens are at positions [n_vision_tokens:], vision at [0:n_vision_tokens]
    heatmaps = []
    for attn in selected:
        # (1, n_heads, seq_len, seq_len) -> text-to-vision slice
        text_to_vision = attn[0, :, n_vision_tokens:, :n_vision_tokens]  # (n_heads, n_text, n_vision)

        # Average over text tokens and heads
        avg_attn = text_to_vision.mean(dim=(0, 1))  # (n_vision,)
        heatmaps.append(avg_attn.cpu().numpy())

    # Aggregate across layers
    heatmaps = np.stack(heatmaps)  # (n_layers, n_vision)
    if aggregate == "mean":
        combined_attn = heatmaps.mean(axis=0)
    elif aggregate == "max":
        combined_attn = heatmaps.max(axis=0)
    elif aggregate == "last":
        combined_attn = heatmaps[-1]
    else:
        combined_attn = heatmaps.mean(axis=0)

    # Reshape to spatial grid
    n_patches = combined_attn.shape[0]
    grid_size = int(np.sqrt(n_patches))
    if grid_size * grid_size != n_patches:
        # Handle non-square (e.g., CLS token included)
        # Try removing first token (CLS)
        if (grid_size + 1) * (grid_size + 1) == n_patches + 1:
            combined_attn = combined_attn  # already correct
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        # Pad to square
        padded = np.zeros(grid_size * grid_size)
        padded[:n_patches] = combined_attn
        combined_attn = padded

    heatmap = combined_attn.reshape(grid_size, grid_size)

    # Resize to target spatial size if needed
    if grid_size != spatial_size:
        from PIL import Image
        img = Image.fromarray((heatmap * 255).astype(np.uint8))
        img = img.resize((spatial_size, spatial_size), Image.BILINEAR)
        heatmap = np.array(img).astype(np.float32) / 255.0

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def batch_extract_heatmaps(
    model,
    vision_features: torch.Tensor,
    claims: list[str],
    tokenizer,
    spatial_size: int = 27,
) -> np.ndarray:
    """Extract heatmaps for a batch of claims.

    Args:
        model: Generator decoder.
        vision_features: (1, n_patches, hidden_dim) — same image for all claims.
        claims: List of claim texts.
        tokenizer: Decoder tokenizer.
        spatial_size: Target heatmap size.

    Returns:
        (n_claims, spatial_size, spatial_size) numpy array.
    """
    heatmaps = []
    for claim in claims:
        hm = extract_cross_attention_heatmap(
            model, vision_features, claim, tokenizer, spatial_size
        )
        heatmaps.append(hm)
    return np.stack(heatmaps)
