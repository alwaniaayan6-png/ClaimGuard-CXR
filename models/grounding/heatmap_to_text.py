"""Convert grounding heatmaps to textual region descriptions.

Bridges the visual grounding module to the text-only verifier by
translating spatial attention patterns into natural language descriptions
of which anatomical regions the model attends to for each claim.
This gives the verifier indirect visual evidence without requiring
a multimodal architecture.
"""

from __future__ import annotations

import numpy as np

# Anatomical region grid for 27x27 spatial features (384px input / 14px patches)
# Approximate CXR anatomy layout (frontal PA/AP view)
_REGION_MAP = {
    "right upper lung": (0, 0, 9, 9),      # top-left (patient's right)
    "right mid lung": (0, 9, 9, 18),        # mid-left
    "right lower lung": (0, 18, 9, 27),     # bottom-left
    "left upper lung": (18, 0, 27, 9),      # top-right (patient's left)
    "left mid lung": (18, 9, 27, 18),       # mid-right
    "left lower lung": (18, 18, 27, 27),    # bottom-right
    "mediastinum": (9, 0, 18, 14),          # center-upper
    "cardiac silhouette": (8, 10, 19, 22),  # center
    "right costophrenic angle": (3, 22, 9, 27),   # lower-left corner
    "left costophrenic angle": (18, 22, 24, 27),  # lower-right corner
    "upper mediastinum": (10, 0, 17, 7),    # top-center (trachea area)
    "right hilum": (8, 8, 12, 14),          # right hilum
    "left hilum": (15, 8, 19, 14),          # left hilum
}

_INTENSITY_THRESHOLDS = {
    "strong": 0.7,
    "moderate": 0.4,
    "weak": 0.15,
}


def heatmap_to_description(
    heatmap: np.ndarray,
    top_k: int = 3,
    threshold: float = 0.15,
) -> str:
    """Convert a 27x27 grounding heatmap to a text description.

    Args:
        heatmap: Attention heatmap of shape (27, 27) with values in [0, 1].
        top_k: Maximum number of regions to describe.
        threshold: Minimum mean attention to include a region.

    Returns:
        Natural language description, e.g.
        "strong attention on left lower lung, moderate attention on cardiac silhouette"
    """
    if heatmap.ndim != 2:
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        else:
            return ""

    h, w = heatmap.shape
    if h != 27 or w != 27:
        # Resize to 27x27 if different
        from PIL import Image
        img = Image.fromarray((heatmap * 255).astype(np.uint8))
        img = img.resize((27, 27), Image.BILINEAR)
        heatmap = np.array(img).astype(np.float32) / 255.0

    # Score each anatomical region
    region_scores = {}
    for region_name, (x1, y1, x2, y2) in _REGION_MAP.items():
        region_patch = heatmap[y1:y2, x1:x2]
        if region_patch.size == 0:
            continue
        mean_attention = float(region_patch.mean())
        max_attention = float(region_patch.max())
        region_scores[region_name] = {
            "mean": mean_attention,
            "max": max_attention,
        }

    # Sort by mean attention, take top-k above threshold
    sorted_regions = sorted(
        region_scores.items(),
        key=lambda x: x[1]["mean"],
        reverse=True,
    )

    descriptions = []
    for region_name, scores in sorted_regions[:top_k]:
        mean_att = scores["mean"]
        if mean_att < threshold:
            continue

        # Determine intensity label
        if mean_att >= _INTENSITY_THRESHOLDS["strong"]:
            intensity = "strong"
        elif mean_att >= _INTENSITY_THRESHOLDS["moderate"]:
            intensity = "moderate"
        else:
            intensity = "weak"

        descriptions.append(f"{intensity} attention on {region_name}")

    if not descriptions:
        return "no significant regional attention detected"

    return ", ".join(descriptions)


def batch_heatmaps_to_descriptions(
    heatmaps: np.ndarray,
    top_k: int = 3,
    threshold: float = 0.15,
) -> list[str]:
    """Convert a batch of heatmaps to text descriptions.

    Args:
        heatmaps: Array of shape (batch, 27, 27).
        top_k: Max regions per description.
        threshold: Min attention threshold.

    Returns:
        List of description strings.
    """
    if heatmaps.ndim == 2:
        return [heatmap_to_description(heatmaps, top_k, threshold)]

    return [
        heatmap_to_description(heatmaps[i], top_k, threshold)
        for i in range(heatmaps.shape[0])
    ]
