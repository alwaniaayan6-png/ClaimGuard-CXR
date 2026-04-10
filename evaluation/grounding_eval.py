"""Grounding evaluation for ClaimGuard-CXR.

Measures spatial alignment between model-generated attention heatmaps and
radiologist-annotated bounding boxes / point annotations.
"""

from __future__ import annotations

import numpy as np


def _heatmap_to_mask(
    heatmap: np.ndarray,
    threshold: float,
    image_size: int,
) -> np.ndarray:
    """Binarise a single heatmap at ``threshold`` and resize to ``image_size``.

    Args:
        heatmap: 2-D float array (H, W), values in [0, 1].
        threshold: Pixels above this value are foreground.
        image_size: Target spatial size; heatmaps are assumed to already be
            this size.  If (H, W) != (image_size, image_size) the heatmap is
            rescaled via nearest-neighbour interpolation.

    Returns:
        Boolean mask of shape (image_size, image_size).
    """
    h, w = heatmap.shape
    if h != image_size or w != image_size:
        # Simple nearest-neighbour resize using index arithmetic
        row_idx = (np.arange(image_size) * h / image_size).astype(int)
        col_idx = (np.arange(image_size) * w / image_size).astype(int)
        heatmap = heatmap[np.ix_(row_idx, col_idx)]
    return heatmap >= threshold


def _bbox_to_mask(bbox: np.ndarray, image_size: int) -> np.ndarray:
    """Convert a bounding box to a binary mask.

    Args:
        bbox: Array of shape (4,) with values [x_min, y_min, x_max, y_max]
            in pixel coordinates (already at ``image_size`` scale).
        image_size: Side length of the square image.

    Returns:
        Boolean mask of shape (image_size, image_size).
    """
    mask = np.zeros((image_size, image_size), dtype=bool)
    x_min, y_min, x_max, y_max = bbox.astype(int)
    x_min = np.clip(x_min, 0, image_size - 1)
    y_min = np.clip(y_min, 0, image_size - 1)
    x_max = np.clip(x_max, 0, image_size)
    y_max = np.clip(y_max, 0, image_size)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iou(
    predicted_heatmaps: np.ndarray,
    target_bboxes: np.ndarray,
    image_size: int = 384,
    threshold: float = 0.5,
) -> float:
    """Compute mean Intersection-over-Union between heatmaps and bounding boxes.

    Each heatmap is thresholded to a binary foreground mask; the bbox is
    rasterised to a mask; IoU is computed per sample and averaged.

    Args:
        predicted_heatmaps: Float array of shape (B, H, W) with values in
            [0, 1].  H and W need not equal ``image_size``; the function
            rescales if needed.
        target_bboxes: Float/int array of shape (B, 4) with columns
            [x_min, y_min, x_max, y_max] in pixels at ``image_size`` scale.
        image_size: Spatial resolution used when rasterising bboxes.
            Default 384 (matches CheXpert/MIMIC 384-px crops).
        threshold: Binarisation threshold for heatmaps.  Default 0.5.

    Returns:
        Mean IoU across the batch (scalar float).  Returns 0.0 for an
        empty batch.
    """
    if len(predicted_heatmaps) == 0:
        return 0.0

    b = len(predicted_heatmaps)
    if len(target_bboxes) != b:
        raise ValueError(
            f"predicted_heatmaps has {b} samples but target_bboxes has "
            f"{len(target_bboxes)}"
        )

    ious: list[float] = []
    for i in range(b):
        pred_mask = _heatmap_to_mask(predicted_heatmaps[i], threshold, image_size)
        gt_mask = _bbox_to_mask(target_bboxes[i], image_size)

        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()

        if union == 0:
            # Both masks are empty — treat as perfect agreement
            ious.append(1.0)
        else:
            ious.append(float(intersection) / float(union))

    return float(np.mean(ious))


def compute_pointing_game(
    predicted_heatmaps: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """Compute the pointing-game accuracy for a batch of heatmaps.

    A hit is scored when the pixel of maximum activation in the heatmap falls
    inside the annotated target region (given as a point here, so within 1
    pixel is exact equality).  Per Selvaraju et al. (Grad-CAM, ICCV 2017),
    the "point" can be any annotated pixel; the argmax must coincide.

    Args:
        predicted_heatmaps: Float array of shape (B, H, W).
        target_points: Int array of shape (B, 2) with columns [x, y] in pixel
            coordinates (column-first, consistent with image (x, y) convention).

    Returns:
        Fraction of samples where argmax coincides with the target point.
        Returns 0.0 for an empty batch.
    """
    if len(predicted_heatmaps) == 0:
        return 0.0

    b = len(predicted_heatmaps)
    if len(target_points) != b:
        raise ValueError(
            f"predicted_heatmaps has {b} samples but target_points has "
            f"{len(target_points)}"
        )

    hits = 0
    for i in range(b):
        heatmap = predicted_heatmaps[i]  # (H, W)
        # argmax returns flat index; convert to (row, col)
        flat_idx = int(np.argmax(heatmap))
        w = heatmap.shape[1]
        pred_row = flat_idx // w
        pred_col = flat_idx % w

        # target_points are [x (col), y (row)]
        tgt_col = int(target_points[i, 0])
        tgt_row = int(target_points[i, 1])

        if pred_row == tgt_row and pred_col == tgt_col:
            hits += 1

    return hits / b
