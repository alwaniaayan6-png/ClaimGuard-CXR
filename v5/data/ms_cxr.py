"""MS-CXR loader.

MS-CXR: 1,162 radiologist-verified (image, phrase, bounding-box) triples across
eight pathologies. Publicly hosted on HuggingFace as a dataset.

Access: `datasets.load_dataset("StanfordAIMI/ms-cxr")` — no PhysioNet
credentialing required. The underlying images are sourced from public CXR
collections (not MIMIC; the HF release is re-hosted).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from .claim_matcher import Annotation
from .claim_parser import load_ontology

logger = logging.getLogger(__name__)

MS_CXR_LABEL_TO_UNIFIED = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Lung Opacity": "lung_opacity",
    "Pleural Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}


@dataclass
class MSCXRRecord:
    image_id: str
    image_path: Path
    phrase: str
    bbox_norm: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0-1
    finding_native: str
    finding_unified: str
    image_width: int
    image_height: int


def iter_ms_cxr(root: Path) -> Iterator[MSCXRRecord]:
    """Iterate the MS-CXR HF download layout (parquet + images/)."""
    import pandas as pd  # local import to keep module light

    parquet = root / "ms_cxr.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Expected MS-CXR parquet at {parquet}. Run `scripts/download_ms_cxr.py` first."
        )
    df = pd.read_parquet(parquet)
    for _, row in df.iterrows():
        bbox = (
            float(row["x"]) / float(row["image_width"]),
            float(row["y"]) / float(row["image_height"]),
            float(row["x"] + row["w"]) / float(row["image_width"]),
            float(row["y"] + row["h"]) / float(row["image_height"]),
        )
        finding_native = row["label_text"]
        yield MSCXRRecord(
            image_id=str(row["dicom_id"]),
            image_path=root / "images" / f"{row['dicom_id']}.png",
            phrase=row["label_phrase"],
            bbox_norm=bbox,
            finding_native=finding_native,
            finding_unified=MS_CXR_LABEL_TO_UNIFIED.get(finding_native, "unknown"),
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
        )


def annotations_for_image(records: list[MSCXRRecord]) -> list[Annotation]:
    out: list[Annotation] = []
    for r in records:
        out.append(
            Annotation(
                image_id=r.image_id,
                finding=r.finding_unified,
                laterality="unknown",
                bbox=r.bbox_norm,
                source="ms_cxr",
                confidence=1.0,
            )
        )
    return out
