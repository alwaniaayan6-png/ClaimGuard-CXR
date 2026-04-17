"""RSNA Pneumonia Detection Challenge loader.

30,227 frontal CXRs with radiologist-drawn bounding boxes for pneumonia /
opacity. Labels: 'Normal', 'Lung Opacity', 'No Lung Opacity / Not Normal'.

Access: Kaggle (RSNA Pneumonia Detection Challenge) — fully public.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation


@dataclass
class RSNARecord:
    image_id: str
    image_path: Path
    label: str  # native label
    bbox_norm: tuple[float, float, float, float] | None
    image_width: int = 1024
    image_height: int = 1024


def iter_rsna(root: Path) -> Iterator[RSNARecord]:
    import pandas as pd

    labels = pd.read_csv(root / "stage_2_train_labels.csv")
    class_info = pd.read_csv(root / "stage_2_detailed_class_info.csv")
    merged = labels.merge(class_info, on="patientId", how="left")
    for _, row in merged.iterrows():
        has_bbox = not (
            row.get("x") is None or (isinstance(row.get("x"), float) and row.get("x") != row.get("x"))
        )
        if has_bbox:
            W, H = 1024.0, 1024.0  # native size of stage2 images
            bbox_norm = (
                float(row["x"]) / W,
                float(row["y"]) / H,
                (float(row["x"]) + float(row["width"])) / W,
                (float(row["y"]) + float(row["height"])) / H,
            )
        else:
            bbox_norm = None
        yield RSNARecord(
            image_id=str(row["patientId"]),
            image_path=root / "stage_2_train_images" / f"{row['patientId']}.dcm",
            label=row.get("class", "Unknown"),
            bbox_norm=bbox_norm,
        )


def annotations_for_record(rec: RSNARecord) -> list[Annotation]:
    if rec.label == "Normal":
        return [
            Annotation(
                image_id=rec.image_id,
                finding="no_finding",
                source="rsna_pneumonia",
                is_structured_negative=False,
            )
        ]
    if rec.label == "Lung Opacity" and rec.bbox_norm is not None:
        return [
            Annotation(
                image_id=rec.image_id,
                finding="lung_opacity",
                bbox=rec.bbox_norm,
                source="rsna_pneumonia",
            )
        ]
    # "No Lung Opacity / Not Normal" → not useful; return empty
    return []
