"""CheXpert Plus loader.

Stanford's extended CheXpert (223,462 studies, 64,725 patients) with reports
and 14-class structured labels via the CheXpert labeler. Primary training set
for v5 because (a) registration-only access, (b) images align with reports.

Access: Stanford AIMI registration — not PhysioNet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation
from .claim_parser import load_ontology


CHEXPERT_14 = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class CheXpertPlusRecord:
    image_id: str          # <patient>_<study>_<view>
    patient_id: str
    study_id: str
    image_path: Path
    report_findings: str
    report_impression: str
    chexpert_labels: dict[str, float]  # class -> label (-1 uncertain, 0 neg, 1 pos, NaN empty)
    view: str


def iter_chexpert_plus(root: Path) -> Iterator[CheXpertPlusRecord]:
    import math

    import pandas as pd

    csv_path = root / "df_chexpert_plus_240401.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    for _, row in df.iterrows():
        image_rel = row.get("path_to_image", row.get("Path"))
        if image_rel is None:
            continue
        image_path = root / image_rel
        if not image_path.exists():
            continue
        pid = str(row.get("patient_id", image_rel.split("/")[1]))
        sid = str(row.get("study", image_rel.split("/")[2]))
        labels = {}
        for cls in CHEXPERT_14:
            v = row.get(cls)
            if isinstance(v, float) and math.isnan(v):
                labels[cls] = float("nan")
            else:
                try:
                    labels[cls] = float(v)
                except (TypeError, ValueError):
                    labels[cls] = float("nan")
        yield CheXpertPlusRecord(
            image_id=f"{pid}_{sid}_{row.get('view', 'unknown')}",
            patient_id=pid,
            study_id=sid,
            image_path=image_path,
            report_findings=str(row.get("section_findings", "") or ""),
            report_impression=str(row.get("section_impression", "") or ""),
            chexpert_labels=labels,
            view=str(row.get("view", "unknown")),
        )


def annotations_for_record(rec: CheXpertPlusRecord) -> list[Annotation]:
    """CheXpert Plus has no bounding boxes; we emit structured-negative / positive labels."""
    ontology = load_ontology()
    map_ = ontology["chexpert_14_to_unified"]
    out: list[Annotation] = []
    for cls, v in rec.chexpert_labels.items():
        unified = map_.get(cls)
        if unified is None:
            continue
        if v == 1.0:
            out.append(
                Annotation(
                    image_id=rec.image_id,
                    finding=unified,
                    source="chexpert_labeler",
                    confidence=0.7,  # labeler is not radiologist-grade
                )
            )
        elif v == 0.0:
            out.append(
                Annotation(
                    image_id=rec.image_id,
                    finding=unified,
                    source="chexpert_labeler",
                    is_structured_negative=True,
                    confidence=0.7,
                )
            )
        # v == -1 (uncertain) or NaN → skip
    return out
