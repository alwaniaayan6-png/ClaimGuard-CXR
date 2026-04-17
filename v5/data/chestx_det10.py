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


def iter_chestx_det10(root: Path) -> Iterator[ChestXDet10Record]:
    import json

    from PIL import Image

    manifest = root / "ChestX_Det_train.json"
    if not manifest.exists():
        manifest = root / "ChestX_Det_test.json"
    with manifest.open() as f:
        records = json.load(f)
    for rec in records:
        image_id = rec.get("file_name", rec.get("image_id"))
        image_path = root / "images" / image_id
        if not image_path.exists():
            continue
        W, H = Image.open(image_path).size
        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        for b in rec.get("boxes", []):
            x1, y1, x2, y2 = b[:4]
            label = b[4] if len(b) >= 5 else rec.get("syms", [None])[0]
            if label is None:
                continue
            boxes.append(
                (label, (x1 / W, y1 / H, x2 / W, y2 / H))
            )
        yield ChestXDet10Record(image_id=image_id, image_path=image_path, boxes=boxes)


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
