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


DEFAULT_METADATA_CSV = Path.home() / "data" / "claimguard" / "chexpert-plus" / "df_chexpert_plus_240401.csv"
# Images live at HPC; locally the CSV has path_to_image relative to the CheXpert+ root.
# Pass image_root=None to accept all rows regardless of image presence (report-only mode).


def iter_chexpert_plus(
    root: Path | None = None,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    require_image: bool = True,
) -> Iterator[CheXpertPlusRecord]:
    """Iterate CheXpert Plus records.

    Args:
        root: Directory containing the images (i.e. ``root/train/patientXXX/...``).
              If None or image not found, rows are skipped when *require_image* is True.
        metadata_csv: Path to ``df_chexpert_plus_240401.csv``.
        require_image: If False, emit records even when the image file is absent
            (useful for report-only claim extraction on the HPC).
    """
    import math

    import pandas as pd

    df = pd.read_csv(metadata_csv, low_memory=False)
    for _, row in df.iterrows():
        image_rel = row.get("path_to_image", row.get("Path"))
        if image_rel is None:
            continue
        image_path = (root / str(image_rel)) if root is not None else Path(str(image_rel))
        if require_image and not image_path.exists():
            continue

        # Patient / study IDs from path: train/patient42142/study5/view1_frontal.jpg
        parts = str(image_rel).replace("\\", "/").split("/")
        pid = str(row.get("deid_patient_id", parts[1] if len(parts) > 1 else "unknown"))
        sid = str(row.get("patient_report_date_order", parts[2] if len(parts) > 2 else "unknown"))
        view = parts[-1].split(".")[0] if parts else "unknown"

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

        # Report: prefer section columns, fall back to monolithic 'report' column.
        findings = str(row.get("section_findings", row.get("report", "")) or "").strip()
        impression = str(row.get("section_impression", "") or "").strip()

        yield CheXpertPlusRecord(
            image_id=f"{pid}_{sid}_{view}",
            patient_id=pid,
            study_id=sid,
            image_path=image_path,
            report_findings=findings,
            report_impression=impression,
            chexpert_labels=labels,
            view=view,
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
