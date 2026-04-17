"""Anatomy segmentation masks: CheXmask-open substitute.

The canonical CheXmask release is PhysioNet-credentialed. v5 rebuilds anatomy
segmentation on public CXR using `torchxrayvision`'s lung+heart segmentation
head (PSPNet trained on publicly reproducible anatomy labels, no MIMIC-CXR
dependency for inference).

We extend the 2-class output (lungs, heart) into a 5-region scheme by splitting
each lung into upper/mid/lower zones based on the lung mask's vertical extent.
This is a coarse but deterministic operation sufficient for the claim matcher
(TAU_IOU=0.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .claim_matcher import AnatomyMask

logger = logging.getLogger(__name__)

REGIONS = [
    "left_upper_lung",
    "left_mid_lung",
    "left_lower_lung",
    "right_upper_lung",
    "right_mid_lung",
    "right_lower_lung",
    "heart",
    "mediastinum",
]


def _split_lung_vertical(lung_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a binary lung mask into upper/mid/lower zones along vertical axis."""
    ys, xs = np.nonzero(lung_mask > 0)
    if len(ys) == 0:
        return (
            np.zeros_like(lung_mask),
            np.zeros_like(lung_mask),
            np.zeros_like(lung_mask),
        )
    y_min, y_max = ys.min(), ys.max()
    upper_cut = y_min + (y_max - y_min) // 3
    mid_cut = y_min + 2 * (y_max - y_min) // 3
    upper = np.zeros_like(lung_mask)
    mid = np.zeros_like(lung_mask)
    lower = np.zeros_like(lung_mask)
    upper[y_min:upper_cut] = lung_mask[y_min:upper_cut]
    mid[upper_cut:mid_cut] = lung_mask[upper_cut:mid_cut]
    lower[mid_cut : y_max + 1] = lung_mask[mid_cut : y_max + 1]
    return upper, mid, lower


def _split_lung_left_right(full_lung: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split lung mask into left and right components based on component centroids."""
    from scipy import ndimage  # type: ignore[import-untyped]

    labeled, n = ndimage.label(full_lung > 0)
    if n == 0:
        return np.zeros_like(full_lung), np.zeros_like(full_lung)
    comps = []
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() < 100:
            continue
        ys, xs = np.nonzero(comp)
        cx = xs.mean() / full_lung.shape[1]
        comps.append((cx, comp))
    comps.sort()
    if len(comps) < 2:
        # one component only — split by image midline
        W = full_lung.shape[1]
        left = np.zeros_like(full_lung)
        right = np.zeros_like(full_lung)
        right[:, : W // 2] = full_lung[:, : W // 2]  # observer-left = patient-right
        left[:, W // 2 :] = full_lung[:, W // 2 :]
        return left, right
    # Lowest-centroid x is observer-left = patient-RIGHT. Highest is patient-LEFT.
    right_comp = comps[0][1]
    left_comp = comps[-1][1]
    return left_comp.astype(np.uint8), right_comp.astype(np.uint8)


def compute_anatomy_masks(image_path: Path) -> AnatomyMask:
    """Run a pretrained segmenter and produce 5-region masks.

    Requires torchxrayvision. If the model is unavailable we fall back to
    image-midline heuristic: the matcher will still mostly function but with
    softer overlap scores.
    """
    from PIL import Image

    img = Image.open(image_path).convert("L")
    W, H = img.size
    regions_out: dict[str, np.ndarray] = {}
    try:
        import torch
        import torchxrayvision as xrv  # type: ignore[import-untyped]

        model = xrv.baseline_models.chestx_det.PSPNet()
        model.eval()
        arr = np.asarray(img.resize((512, 512))).astype(np.float32)
        arr = (arr / 255.0 * 2) - 1.0
        t = torch.from_numpy(arr)[None, None]
        with torch.no_grad():
            pred = model(t).sigmoid().cpu().numpy()[0]
        # torchxrayvision pspnet outputs 14 anatomy classes; the ones we care about:
        target_names = list(getattr(model, "targets", [
            "Left Clavicle",
            "Right Clavicle",
            "Left Scapula",
            "Right Scapula",
            "Left Lung",
            "Right Lung",
            "Left Hilus Pulmonis",
            "Right Hilus Pulmonis",
            "Heart",
            "Aorta",
            "Facies Diaphragmatica",
            "Mediastinum",
            "Weasand",
            "Spine",
        ]))

        def _resize_bin(m: np.ndarray) -> np.ndarray:
            return np.asarray(
                Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
            ) > 127

        def _find(name: str) -> np.ndarray | None:
            try:
                idx = target_names.index(name)
                return _resize_bin(pred[idx]).astype(np.uint8)
            except ValueError:
                return None

        left_lung = _find("Left Lung")
        right_lung = _find("Right Lung")
        heart = _find("Heart")
        mediastinum = _find("Mediastinum")

        if left_lung is None or right_lung is None:
            # fallback: treat single lung mask and split
            full = np.zeros((H, W), dtype=np.uint8)
            if left_lung is not None:
                full |= left_lung
            if right_lung is not None:
                full |= right_lung
            left_lung, right_lung = _split_lung_left_right(full)

        lu, lm, ll = _split_lung_vertical(left_lung)
        ru, rm, rl = _split_lung_vertical(right_lung)
        regions_out = {
            "left_upper_lung": lu,
            "left_mid_lung": lm,
            "left_lower_lung": ll,
            "right_upper_lung": ru,
            "right_mid_lung": rm,
            "right_lower_lung": rl,
            "heart": heart if heart is not None else np.zeros((H, W), dtype=np.uint8),
            "mediastinum": mediastinum if mediastinum is not None else np.zeros((H, W), dtype=np.uint8),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("torchxrayvision unavailable (%s); using midline fallback", exc)
        # midline-split fallback: treat full image as lungs
        full = np.ones((H, W), dtype=np.uint8)
        left_lung, right_lung = _split_lung_left_right(full)
        lu, lm, ll = _split_lung_vertical(left_lung)
        ru, rm, rl = _split_lung_vertical(right_lung)
        regions_out = {
            "left_upper_lung": lu.astype(np.uint8),
            "left_mid_lung": lm.astype(np.uint8),
            "left_lower_lung": ll.astype(np.uint8),
            "right_upper_lung": ru.astype(np.uint8),
            "right_mid_lung": rm.astype(np.uint8),
            "right_lower_lung": rl.astype(np.uint8),
            "heart": np.zeros((H, W), dtype=np.uint8),
            "mediastinum": np.zeros((H, W), dtype=np.uint8),
        }

    return AnatomyMask(
        image_id=image_path.stem,
        masks=regions_out,
        height=H,
        width=W,
    )


def save_anatomy_masks(masks: AnatomyMask, out_dir: Path) -> None:
    """Save each region as a PNG inside out_dir/<image_id>/<region>.png."""
    from PIL import Image

    d = out_dir / masks.image_id
    d.mkdir(parents=True, exist_ok=True)
    for region, m in masks.masks.items():
        Image.fromarray((m * 255).astype(np.uint8)).save(d / f"{region}.png")


def load_anatomy_masks(image_id: str, root: Path) -> AnatomyMask | None:
    from PIL import Image

    d = root / image_id
    if not d.exists():
        return None
    masks = {}
    H = W = 0
    for region in REGIONS:
        p = d / f"{region}.png"
        if not p.exists():
            continue
        arr = np.array(Image.open(p))
        masks[region] = (arr > 127).astype(np.uint8)
        H, W = arr.shape[:2]
    return AnatomyMask(image_id=image_id, masks=masks, height=H, width=W)
