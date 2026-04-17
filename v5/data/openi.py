"""OpenI (Indiana University) loader.

3,996 frontal/lateral CXRs with full radiologist-written reports. No bounding
boxes. Used for silver evaluation and provenance dual-run experiments.

Access: NLM — fully public, no credentialing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation


@dataclass
class OpenIRecord:
    image_id: str
    image_path: Path
    report_findings: str
    report_impression: str
    mesh_terms: list[str]


def iter_openi(root: Path) -> Iterator[OpenIRecord]:
    import pandas as pd

    csv_path = root / "openi_cxr_chexpert_schema.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    for _, row in df.iterrows():
        pid = row.get("deid_patient_id")
        image_id = f"openi_{pid}"
        img_path = root / "images" / f"{pid}.png"
        if not img_path.exists():
            continue
        yield OpenIRecord(
            image_id=image_id,
            image_path=img_path,
            report_findings=str(row.get("section_findings", "") or ""),
            report_impression=str(row.get("section_impression", "") or ""),
            mesh_terms=str(row.get("mesh_terms", "") or "").split(";"),
        )


def annotations_for_record(rec: OpenIRecord) -> list[Annotation]:
    """OpenI has no pixel annotations. We emit no annotations (NO_GT)."""
    return []
