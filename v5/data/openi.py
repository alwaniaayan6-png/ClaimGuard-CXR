"""OpenI (Indiana University) loader.

3,996 frontal/lateral CXRs with full radiologist-written reports. No bounding
boxes. Used for silver evaluation and provenance dual-run experiments.

Access: NLM — fully public, no credentialing.

On-disk layout (Laughney lab):
  image_root/          e.g. ~/data/openi/
    CXR{uid}_IM-*.png  one or more views per study
  report_csv           e.g. ~/data/claimguard/iu-xray/iu_xray_reports.csv
    columns: uid, findings, impression, indication, problems, mesh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .claim_matcher import Annotation

# Default paths matching the Laughney-lab layout.
DEFAULT_IMAGE_ROOT = Path.home() / "data" / "openi"
DEFAULT_REPORT_CSV = Path.home() / "data" / "claimguard" / "iu-xray" / "iu_xray_reports.csv"


@dataclass
class OpenIRecord:
    image_id: str
    image_path: Path
    report_findings: str
    report_impression: str
    mesh_terms: list[str] = field(default_factory=list)


def iter_openi(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    report_csv: Path = DEFAULT_REPORT_CSV,
) -> Iterator[OpenIRecord]:
    """Yield one OpenIRecord per (study × frontal view) pair that has an image on disk.

    The function resolves images by globbing for ``CXR{uid}_IM-*.png`` in *image_root*.
    Lateral views are skipped (filename pattern ends with ``-2001`` / ``-3001``).
    """
    import pandas as pd

    df = pd.read_csv(report_csv, low_memory=False)
    for _, row in df.iterrows():
        uid = str(row.get("uid", "")).strip()
        if not uid:
            continue

        # Each study can have multiple views; prefer frontal (IM-xxxx-1001 pattern).
        candidates = sorted(image_root.glob(f"CXR{uid}_IM-*.png"))
        # Frontal heuristic: last segment before .png ends in 1001
        frontal = [p for p in candidates if p.stem.endswith("1001")]
        img_path = frontal[0] if frontal else (candidates[0] if candidates else None)
        if img_path is None:
            continue

        findings = str(row.get("findings", "") or "").strip()
        impression = str(row.get("impression", "") or "").strip()
        mesh_raw = str(row.get("mesh", row.get("problems", "")) or "")
        mesh_terms = [t.strip() for t in mesh_raw.split(";") if t.strip()]

        yield OpenIRecord(
            image_id=f"openi_{uid}",
            image_path=img_path,
            report_findings=findings,
            report_impression=impression,
            mesh_terms=mesh_terms,
        )


def annotations_for_record(rec: OpenIRecord) -> list[Annotation]:
    """OpenI has no pixel annotations. We emit no annotations (NO_GT)."""
    return []
