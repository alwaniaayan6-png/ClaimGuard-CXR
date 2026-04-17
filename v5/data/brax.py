"""BRAX (Brazilian Radiology Assistant X-ray) loader.

40,967 CXRs from a major Brazilian hospital with Portuguese reports and
CheXpert-labeler derived labels. Radiologist-authored reports.

Access: IEEE DataPort registration — not PhysioNet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation
from .chexpert_plus import CHEXPERT_14
from .claim_parser import load_ontology


@dataclass
class BRAXRecord:
    image_id: str
    image_path: Path
    report_pt: str
    report_en: str | None
    chexpert_labels: dict[str, float]


def iter_brax(root: Path) -> Iterator[BRAXRecord]:
    import math

    import pandas as pd

    csv_path = root / "master_spreadsheet_update.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    for _, row in df.iterrows():
        image_rel = row.get("PngPath")
        if image_rel is None:
            continue
        image_path = root / image_rel
        if not image_path.exists():
            continue
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
        yield BRAXRecord(
            image_id=str(image_rel).replace("/", "_"),
            image_path=image_path,
            report_pt=str(row.get("ReportPT", "") or ""),
            report_en=str(row.get("ReportEN")) if "ReportEN" in row else None,
            chexpert_labels=labels,
        )


def annotations_for_record(rec: BRAXRecord) -> list[Annotation]:
    ontology = load_ontology()
    map_ = ontology["chexpert_14_to_unified"]
    out: list[Annotation] = []
    for cls, v in rec.chexpert_labels.items():
        unified = map_.get(cls)
        if unified is None:
            continue
        if v == 1.0:
            out.append(Annotation(image_id=rec.image_id, finding=unified, source="brax_labeler", confidence=0.7))
        elif v == 0.0:
            out.append(
                Annotation(
                    image_id=rec.image_id,
                    finding=unified,
                    source="brax_labeler",
                    is_structured_negative=True,
                    confidence=0.7,
                )
            )
    return out
