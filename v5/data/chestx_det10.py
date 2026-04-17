"""ChestX-Det10 loader.

3,543 CXRs with radiologist-drawn bounding boxes for 10 findings.
Access: Github (Chen et al. 2020) — public under CC-BY-NC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation

CHESTX_DET10_TO_UNIFIED = {
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Consolidation": "consolidation",
    "Effusion": "pleural_effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Fracture": "fracture",
    "Mass": "mass",
    "Nodule": "lung_lesion",
    "Pneumothorax": "pneumothorax",
}


@dataclass
class ChestXDet10Record:
    image_id: str
    image_path: Path
    boxes: list[tuple[str, tuple[float, float, float, float]]]  # (native_label, bbox_norm)


DEFAULT_ANNOT_ROOT = Path.home() / "data" / "chestx_det10"
DEFAULT_IMAGE_ROOT = Path.home() / "data" / "chestx_det10_images"

# Inner dirs after unzipping deepwise.com archives:
#   train_data.zip → train/train-old/*.png
#   test_data.zip  → test/test_data/*.png
_SPLIT_SUBDIRS = {"train": "train/train-old", "test": "test/test_data"}


def iter_chestx_det10(
    annot_root: Path = DEFAULT_ANNOT_ROOT,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    splits: tuple[str, ...] = ("train", "test"),
) -> Iterator[ChestXDet10Record]:
    """Iterate ChestX-Det10 records.

    Annotation format (actual, from Deepwise AI Lab repo):
        [{"file_name": "36204.png", "syms": ["Nodule"], "boxes": [[x1,y1,x2,y2], ...]}, ...]
    where x1,y1,x2,y2 are absolute pixel coordinates (top-left, bottom-right).

    Images live in image_root/{train,test}/ after unzipping train_data.zip / test_data.zip.
    """
    import json

    from PIL import Image

    for split in splits:
        manifest = annot_root / f"{split}.json"
        if not manifest.exists():
            continue
        with manifest.open() as f:
            records = json.load(f)

        subdir = _SPLIT_SUBDIRS.get(split, split)
        img_dir = image_root / subdir if (image_root / subdir).exists() else image_root

        for rec in records:
            fname = rec.get("file_name", "")
            image_path = img_dir / fname
            if not image_path.exists():
                continue
            syms: list[str] = rec.get("syms", [])
            raw_boxes: list[list[int]] = rec.get("boxes", [])

            try:
                W, H = Image.open(image_path).size
            except Exception:
                continue

            boxes: list[tuple[str, tuple[float, float, float, float]]] = []
            for sym, b in zip(syms, raw_boxes):
                if len(b) < 4:
                    continue
                x1, y1, x2, y2 = b[:4]
                boxes.append((sym, (x1 / W, y1 / H, x2 / W, y2 / H)))

            yield ChestXDet10Record(
                image_id=f"chestxdet10_{split}_{fname}",
                image_path=image_path,
                boxes=boxes,
            )


def annotations_for_record(rec: ChestXDet10Record) -> list[Annotation]:
    out: list[Annotation] = []
    for label, bbox in rec.boxes:
        unified = CHESTX_DET10_TO_UNIFIED.get(label, "unknown")
        if unified == "unknown":
            continue
        out.append(
            Annotation(
                image_id=rec.image_id,
                finding=unified,
                bbox=bbox,
                source="chestx_det10",
            )
        )
    return out
