"""PadChest loader (structured labels + localized subset).

160,868 frontal/lateral CXRs from BIMCV (Valencia, Spain). Labels cover 174
findings; 27% of labels were drawn by trained radiologists, the rest NLP.
A subset has radiologist bounding boxes (PadChest-localized); we use that
subset for image-grounded evaluation.

Access: BIMCV registration — not PhysioNet credentialing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation
from .claim_parser import load_ontology


@dataclass
class PadChestRecord:
    image_id: str
    image_path: Path
    report_raw: str             # Spanish
    report_en: str | None       # Translated, if available
    structured_labels: list[str]  # native PadChest 174-class labels
    bboxes: list[tuple[str, tuple[float, float, float, float]]]  # (native_label, bbox_norm)
    is_radiologist_labeled: bool


def _safe_parse_list(s: str) -> list:
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (SyntaxError, ValueError):
        return []


def iter_padchest(root: Path, english_reports: bool = True) -> Iterator[PadChestRecord]:
    import pandas as pd

    csv_path = root / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    bbox_path = root / "padchest_localized.csv"  # user-supplied localized subset
    bbox_df = None
    if bbox_path.exists():
        bbox_df = pd.read_csv(bbox_path)

    for _, row in df.iterrows():
        image_id = row["ImageID"]
        image_path = root / "images" / image_id
        if not image_path.exists():
            continue
        labels = _safe_parse_list(row.get("Labels", "[]"))
        bboxes: list[tuple[str, tuple[float, float, float, float]]] = []
        if bbox_df is not None:
            sub = bbox_df[bbox_df["ImageID"] == image_id]
            for _, b in sub.iterrows():
                bboxes.append(
                    (b["label"], (b["x1"], b["y1"], b["x2"], b["y2"]))
                )
        yield PadChestRecord(
            image_id=image_id,
            image_path=image_path,
            report_raw=str(row.get("Report", "")),
            report_en=str(row.get("Report_EN")) if english_reports and "Report_EN" in row else None,
            structured_labels=labels,
            bboxes=bboxes,
            is_radiologist_labeled=row.get("MethodLabel", "") == "physician",
        )


def annotations_for_record(rec: PadChestRecord) -> list[Annotation]:
    ont = load_ontology()
    map_ = ont.get("padchest_seed_to_unified", {})
    out: list[Annotation] = []
    labels_unified = {map_.get(lbl.lower(), None) for lbl in rec.structured_labels}
    labels_unified.discard(None)
    for label, bbox in rec.bboxes:
        unified = map_.get(label.lower(), "unknown")
        if unified == "unknown":
            continue
        out.append(
            Annotation(
                image_id=rec.image_id,
                finding=unified,
                bbox=bbox,
                source="padchest_localized",
                confidence=1.0 if rec.is_radiologist_labeled else 0.7,
            )
        )
    # include structured negatives: unified classes NOT mentioned → explicit absence
    mentioned = {a.finding for a in out}
    for unified in (ont.get("unified_classes") or []):
        if unified == "no_finding":
            continue
        if unified not in mentioned and unified not in labels_unified:
            out.append(
                Annotation(
                    image_id=rec.image_id,
                    finding=unified,
                    source="padchest_structured_negative",
                    is_structured_negative=True,
                    confidence=0.6,
                )
            )
    if "normal" in [lbl.lower() for lbl in rec.structured_labels]:
        out.append(
            Annotation(
                image_id=rec.image_id,
                finding="no_finding",
                source="padchest",
            )
        )
    return out
