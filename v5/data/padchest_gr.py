"""PadChest-GR dataset loader.

PadChest-GR (arXiv:2411.05085, NEJM AI 2025) is a BIMCV (Valencia) release of
4,555 CXR studies with per-finding-sentence bounding boxes placed by
radiologists, bilingual Spanish/English report text, and CheXpert-style
finding labels.

Access: BIMCV Research Use Agreement at
https://bimcv.cipf.es/bimcv-projects/padchest-gr/ (46 GB download via
EUDAT B2DROP). Registration-only — not PhysioNet-credentialed.

This module expects the dataset laid out on disk as::

    <root>/
      padchest_gr_images/
        <study_id>.png  (or .jpg / .dcm)
      padchest_gr_labels.csv
      padchest_gr_boxes.jsonl   (per-sentence grounding JSONL)

``padchest_gr_labels.csv`` is expected to carry columns at minimum:
``study_id``, ``patient_id``, ``report_es``, ``report_en`` (pre-translated
or translation added by ``translate_reports()``), ``findings_labels``.

``padchest_gr_boxes.jsonl`` is expected to carry one row per
finding-sentence with keys: ``study_id``, ``sentence_es``, ``sentence_en``,
``finding``, ``bbox`` (``[x1, y1, x2, y2]`` normalized 0-1), ``is_positive``
(true if the sentence asserts presence, false if absence).

The loader is resilient to missing ``report_en``: callers can run
``translate_reports()`` (Claude Haiku, cheap) to backfill before assembly.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class PadChestGRBox:
    sentence_es: str
    sentence_en: str
    finding: str
    bbox: tuple[float, float, float, float]
    is_positive: bool


@dataclass
class PadChestGRRecord:
    study_id: str
    patient_id: str
    image_path: Path
    report_es: str
    report_en: str
    findings_labels: list[str]
    boxes: list[PadChestGRBox] = field(default_factory=list)
    site: str = "padchest-gr"
    language: str = "es"


def _read_labels_csv(csv_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            sid = r.get("study_id") or r.get("StudyID") or r.get("id")
            if not sid:
                continue
            findings = r.get("findings_labels", "")
            parsed_findings = [f.strip() for f in findings.split(";") if f.strip()]
            rows[str(sid)] = {
                "patient_id": r.get("patient_id", ""),
                "report_es": r.get("report_es", "") or r.get("report", ""),
                "report_en": r.get("report_en", ""),
                "findings_labels": parsed_findings,
            }
    return rows


def _read_boxes_jsonl(jsonl_path: Path) -> dict[str, list[PadChestGRBox]]:
    by_study: dict[str, list[PadChestGRBox]] = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("study_id", ""))
            if not sid:
                continue
            bbox = r.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            by_study.setdefault(sid, []).append(PadChestGRBox(
                sentence_es=str(r.get("sentence_es", "")),
                sentence_en=str(r.get("sentence_en", "")),
                finding=str(r.get("finding", "")),
                bbox=tuple(bbox) if len(bbox) == 4 else (0.0, 0.0, 0.0, 0.0),
                is_positive=bool(r.get("is_positive", True)),
            ))
    return by_study


def _resolve_image_path(root: Path, study_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".dcm"):
        p = root / "padchest_gr_images" / f"{study_id}{ext}"
        if p.exists():
            return p
    return root / "padchest_gr_images" / f"{study_id}.png"


def load_records(
    root: Path,
    *,
    labels_csv_name: str = "padchest_gr_labels.csv",
    boxes_jsonl_name: str = "padchest_gr_boxes.jsonl",
    max_records: int | None = None,
) -> list[PadChestGRRecord]:
    """Load records, skipping studies with no labels and emitting warnings."""
    root = Path(root)
    labels_csv = root / labels_csv_name
    boxes_jsonl = root / boxes_jsonl_name
    if not labels_csv.exists():
        raise FileNotFoundError(f"{labels_csv} not found")
    labels = _read_labels_csv(labels_csv)
    boxes = _read_boxes_jsonl(boxes_jsonl) if boxes_jsonl.exists() else {}
    records: list[PadChestGRRecord] = []
    for sid, row in labels.items():
        rec = PadChestGRRecord(
            study_id=sid,
            patient_id=str(row.get("patient_id", "")),
            image_path=_resolve_image_path(root, sid),
            report_es=str(row.get("report_es", "")),
            report_en=str(row.get("report_en", "")),
            findings_labels=list(row.get("findings_labels", [])),
            boxes=boxes.get(sid, []),
        )
        records.append(rec)
        if max_records is not None and len(records) >= max_records:
            break
    logger.info("loaded %d PadChest-GR records (%d with boxes, %d with EN reports)",
                len(records),
                sum(1 for r in records if r.boxes),
                sum(1 for r in records if r.report_en))
    return records


_TRANSLATE_SYSTEM = (
    "You are a professional medical translator. Translate the given Spanish "
    "radiology report findings into clinical English. Preserve all anatomical "
    "terms, severity qualifiers, and laterality. Output only the translated "
    "text, no preamble."
)


def translate_reports(
    records: list[PadChestGRRecord],
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 700,
    force: bool = False,
) -> int:
    """Populate ``record.report_en`` via Claude Haiku where missing (in place).

    Args:
        records: list of records to translate. Modified in place.
        model: Anthropic model ID for translation.
        max_tokens: max tokens per translation response.
        force: if True, re-translate even if report_en is non-empty.

    Returns:
        Number of translations performed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package required for translate_reports") from exc
    client = anthropic.Anthropic()
    translated = 0
    for rec in records:
        if not force and rec.report_en.strip():
            continue
        if not rec.report_es.strip():
            continue
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=_TRANSLATE_SYSTEM,
                    messages=[{"role": "user", "content": rec.report_es[:3000]}],
                )
                rec.report_en = msg.content[0].text if msg.content else ""
                translated += 1
                break
            except Exception as exc:
                if attempt == 2:
                    logger.warning("translate failed for %s: %s", rec.study_id, exc)
                    continue
                time.sleep(2 ** attempt)
    return translated


def records_to_jsonl(records: Iterable[PadChestGRRecord], out_path: Path) -> int:
    """Serialize records to a JSONL for downstream groundbench assembly.

    The JSONL schema matches the convention used by ``v5/data/_common.py``
    patient-level rows — ``site``, ``patient_id``, ``image_path``, ``report``,
    ``labels``, ``boxes``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        for rec in records:
            row = {
                "site": rec.site,
                "study_id": rec.study_id,
                "patient_id": rec.patient_id,
                "image_path": str(rec.image_path),
                "report_es": rec.report_es,
                "report_en": rec.report_en,
                "findings_labels": rec.findings_labels,
                "boxes": [
                    {
                        "sentence_es": b.sentence_es,
                        "sentence_en": b.sentence_en,
                        "finding": b.finding,
                        "bbox": list(b.bbox),
                        "is_positive": b.is_positive,
                    }
                    for b in rec.boxes
                ],
            }
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n
