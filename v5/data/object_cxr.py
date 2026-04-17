"""Object-CXR foreign-object detection loader.

~9,000 CXRs with radiologist-drawn bounding boxes for foreign objects (jewelry,
buttons, implants, pacemakers). MICCAI 2020 dataset.
Access: Github (JF Healthcare) — public.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation


@dataclass
class ObjectCXRRecord:
    image_id: str
    image_path: Path
    bboxes_norm: list[tuple[float, float, float, float]]
    image_width: int
    image_height: int


def iter_object_cxr(root: Path) -> Iterator[ObjectCXRRecord]:
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(root / "train.csv")  # columns: image_name, annotation
    for _, row in df.iterrows():
        image_path = root / "train" / row["image_name"]
        if not image_path.exists():
            continue
        W, H = Image.open(image_path).size
        raw = str(row.get("annotation", "") or "")
        bboxes: list[tuple[float, float, float, float]] = []
        if raw.strip():
            # Object-CXR annotation format: "type x1 y1 x2 y2 [x3 y3 ...]; type ..."
            for obj in raw.split(";"):
                parts = obj.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                xs = coords[0::2]
                ys = coords[1::2]
                if not xs or not ys:
                    continue
                bboxes.append(
                    (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H)
                )
        yield ObjectCXRRecord(
            image_id=row["image_name"],
            image_path=image_path,
            bboxes_norm=bboxes,
            image_width=W,
            image_height=H,
        )


def annotations_for_record(rec: ObjectCXRRecord) -> list[Annotation]:
    if not rec.bboxes_norm:
        return [
            Annotation(
                image_id=rec.image_id,
                finding="foreign_object",
                source="object_cxr",
                is_structured_negative=True,
            )
        ]
    return [
        Annotation(
            image_id=rec.image_id,
            finding="foreign_object",
            bbox=bbox,
            source="object_cxr",
        )
        for bbox in rec.bboxes_norm
    ]
