"""PadChest-GR dataset loader.

PadChest-GR (arXiv:2411.05085, NEJM AI 2025) is a BIMCV (Valencia) release of
4,555 CXR studies with per-finding-sentence bounding boxes placed by
radiologists, bilingual Spanish/English report text, and CheXpert-style
finding labels.

Access: BIMCV Research Use Agreement at
https://bimcv.cipf.es/bimcv-projects/padchest-gr/ (~46 GB download via
EUDAT B2DROP, or ~7 MB if only text+boxes are needed for silver-label
validation). Registration-only — not PhysioNet-credentialed.

Canonical layout on disk (matches the actual 2024-08-19 release):

    <root>/
      grounded_reports_20240819.json     # 6.3 MB, list of per-study records
      master_table.csv                    # 2.9 MB, 8,787 (study, sentence) rows
      Padchest_GR_files/                  # 38.5 GB, PNG images (optional)
      PadChest_GR_progression_prior_studies/  # 9 GB, prior PNGs (optional)

``grounded_reports_20240819.json`` schema (per-study record):

    {
      "StudyID":           "251488034557732338959601580328898734705",
      "ImageID":           "...wigcpj.png",
      "PreviousStudyID":   null | str,
      "PreviousImageID":   null | str,
      "findings": [
        {
          "sentence_en":   "Minimal biapical pleural thickening.",
          "sentence_es":   "Mínimo engrosamiento pleural biapical.",
          "abnormal":      true,        # true = asserts a positive finding
          "boxes":         [[x1,y1,x2,y2], ...],   # normalized 0-1
          "extra_boxes":   [[...], ...],
          "labels":        ["apical pleural thickening"],
          "locations":     ["pleural"],
          "progression":   null | "stable" | "improved" | "worsened"
        },
        ...
      ]
    }

``master_table.csv`` columns (one row per (study, sentence) pair):

    StudyID, ImageID, label, boxes_count, extra_boxes_count, locations,
    prior_study, progression_status, prior_imageID, sentence_en, sentence_es,
    study_is_benchmark, study_is_validation, split, PatientID,
    patient_is_benchmark, PatientBirth, PatientSex_DICOM,
    StudyDate_DICOM, StudyDate, PatientAge, label_group, Year
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
    labels: list[str]
    locations: list[str]
    bboxes: list[tuple[float, float, float, float]]
    extra_bboxes: list[tuple[float, float, float, float]]
    is_positive: bool                       # maps from ``abnormal``
    progression: str | None = None


@dataclass
class PadChestGRRecord:
    study_id: str
    image_id: str
    patient_id: str
    image_path: Path
    split: str                              # train | val | test | "" if no master table
    report_es: str
    report_en: str
    findings: list[PadChestGRBox] = field(default_factory=list)
    patient_age: int | None = None
    patient_sex: str | None = None
    previous_study_id: str | None = None
    previous_image_id: str | None = None
    site: str = "padchest-gr"
    language: str = "es"


def _load_grounded_reports_json(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"grounded_reports JSON root is {type(data)}, expected list")
    return data


def _load_master_table(csv_path: Path) -> dict[str, dict]:
    """Index master_table.csv by ``StudyID``.

    master_table has one row per (study, sentence), so we consolidate
    study-level fields (patient_id, split, demographics) by taking the first
    row per study. Per-sentence fields are loaded separately from the JSON.
    """
    by_study: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("StudyID")
            if not sid or sid in by_study:
                continue
            try:
                age = int(row["PatientAge"]) if row.get("PatientAge") else None
            except ValueError:
                age = None
            by_study[sid] = {
                "patient_id": row.get("PatientID", ""),
                "split": row.get("split", ""),
                "patient_age": age,
                "patient_sex": row.get("PatientSex_DICOM") or None,
                "study_is_benchmark": row.get("study_is_benchmark", "").lower() == "true",
                "study_is_validation": row.get("study_is_validation", "").lower() == "true",
            }
    return by_study


def _sentences_to_report(findings: list[PadChestGRBox], lang: str) -> str:
    """Concatenate per-finding sentences to reconstruct a pseudo-report."""
    attr = f"sentence_{lang}"
    parts = [getattr(f, attr, "") for f in findings]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _resolve_image_path(root: Path, image_id: str, image_subdir: str = "Padchest_GR_files") -> Path:
    p = root / image_subdir / image_id
    if p.exists():
        return p
    for ext in (".png", ".jpg", ".jpeg"):
        alt = root / image_subdir / f"{Path(image_id).stem}{ext}"
        if alt.exists():
            return alt
    return p


def _parse_bbox_list(raw) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    if not isinstance(raw, list):
        return out
    for b in raw:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            except (TypeError, ValueError):
                continue
    return out


def load_records(
    root: Path,
    *,
    grounded_json: str = "grounded_reports_20240819.json",
    master_csv: str = "master_table.csv",
    image_subdir: str = "Padchest_GR_files",
    max_records: int | None = None,
) -> list[PadChestGRRecord]:
    """Load PadChest-GR into a unified record list.

    Tolerates a missing master_table.csv (useful for partial downloads where
    only grounded_reports_20240819.json was fetched for silver validation).
    """
    root = Path(root)
    json_path = root / grounded_json
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found")
    raw_studies = _load_grounded_reports_json(json_path)

    csv_path = root / master_csv
    master = _load_master_table(csv_path) if csv_path.exists() else {}
    if not master:
        logger.warning("master_table.csv not loaded; splits + patient_id will be empty")

    records: list[PadChestGRRecord] = []
    for study in raw_studies:
        sid = str(study.get("StudyID", ""))
        if not sid:
            continue
        image_id = str(study.get("ImageID", ""))
        findings: list[PadChestGRBox] = []
        for f in study.get("findings", []) or []:
            findings.append(PadChestGRBox(
                sentence_es=str(f.get("sentence_es", "")),
                sentence_en=str(f.get("sentence_en", "")),
                labels=list(f.get("labels") or []),
                locations=list(f.get("locations") or []),
                bboxes=_parse_bbox_list(f.get("boxes")),
                extra_bboxes=_parse_bbox_list(f.get("extra_boxes")),
                is_positive=bool(f.get("abnormal", False)),
                progression=f.get("progression"),
            ))
        meta = master.get(sid, {})
        rec = PadChestGRRecord(
            study_id=sid,
            image_id=image_id,
            patient_id=str(meta.get("patient_id", "")),
            image_path=_resolve_image_path(root, image_id, image_subdir),
            split=str(meta.get("split", "")),
            report_es=_sentences_to_report(findings, "es"),
            report_en=_sentences_to_report(findings, "en"),
            findings=findings,
            patient_age=meta.get("patient_age"),
            patient_sex=meta.get("patient_sex"),
            previous_study_id=study.get("PreviousStudyID"),
            previous_image_id=study.get("PreviousImageID"),
        )
        records.append(rec)
        if max_records is not None and len(records) >= max_records:
            break
    logger.info(
        "loaded %d PadChest-GR records (%d with bboxes, %d with split, %d with EN)",
        len(records),
        sum(1 for r in records if any(f.bboxes for f in r.findings)),
        sum(1 for r in records if r.split),
        sum(1 for r in records if r.report_en),
    )
    return records


_TRANSLATE_SYSTEM = (
    "You are a professional medical translator. Translate the given Spanish "
    "radiology sentence to clinical English. Preserve all anatomical terms, "
    "severity qualifiers, laterality, and negation. Output only the translated "
    "text — no preamble."
)


def translate_missing_en_sentences(
    records: list[PadChestGRRecord],
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 200,
) -> int:
    """Populate any missing ``sentence_en`` via Claude Haiku (in place).

    Most of PadChest-GR already ships ``sentence_en`` pre-translated; this is
    only needed when a sentence's EN field is empty (rare).

    Returns:
        Number of translations performed.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package required for translate_missing_en_sentences") from exc
    client = anthropic.Anthropic()
    translated = 0
    for rec in records:
        for f in rec.findings:
            if f.sentence_en.strip() or not f.sentence_es.strip():
                continue
            for attempt in range(3):
                try:
                    msg = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=_TRANSLATE_SYSTEM,
                        messages=[{"role": "user", "content": f.sentence_es[:800]}],
                    )
                    f.sentence_en = msg.content[0].text if msg.content else ""
                    translated += 1
                    break
                except Exception as exc:
                    if attempt == 2:
                        logger.warning("translate failed for study %s: %s", rec.study_id, exc)
                        continue
                    time.sleep(2 ** attempt)
        rec.report_en = _sentences_to_report(rec.findings, "en")
    return translated


def records_to_jsonl(records: Iterable[PadChestGRRecord], out_path: Path) -> int:
    """Serialize records to a JSONL consumed by ``padchest_gr_validate``.

    The JSONL carries flat fields for downstream matching:

        study_id, image_id, patient_id, split, report_es, report_en,
        boxes: [{sentence_en, sentence_es, finding, bbox, is_positive}, ...]

    ``boxes`` uses the union of ``boxes`` + ``extra_boxes`` per sentence to
    carry the grounding region list. One entry per radiologist-authored
    sentence.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        for rec in records:
            boxes_out: list[dict] = []
            for f in rec.findings:
                finding_label = f.labels[0] if f.labels else ""
                box_list = list(f.bboxes) or list(f.extra_bboxes)
                if box_list:
                    for b in box_list:
                        boxes_out.append({
                            "sentence_es": f.sentence_es,
                            "sentence_en": f.sentence_en,
                            "finding": finding_label,
                            "bbox": list(b),
                            "is_positive": f.is_positive,
                        })
                else:
                    boxes_out.append({
                        "sentence_es": f.sentence_es,
                        "sentence_en": f.sentence_en,
                        "finding": finding_label,
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "is_positive": f.is_positive,
                    })
            row = {
                "site": rec.site,
                "study_id": rec.study_id,
                "image_id": rec.image_id,
                "patient_id": rec.patient_id,
                "image_path": str(rec.image_path),
                "split": rec.split,
                "report_es": rec.report_es,
                "report_en": rec.report_en,
                "patient_age": rec.patient_age,
                "patient_sex": rec.patient_sex,
                "boxes": boxes_out,
            }
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n
