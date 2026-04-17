"""PadChest-GR loader (BIMCV + Microsoft, grounded subset).

PadChest-GR releases per-sentence bounding boxes tied to specific
positive-finding sentences from the PadChest reports. Format (as of the
BIMCV Nov 2024 release):

  <root>/
    images/<study>_<view>.png
    padchest_gr_annotations.json        # top-level list of records
    padchest_gr_reports.tsv             # study_id -> full report text

Each record:
  {
    "study_id": "...",
    "image_file": "...",
    "image_width": int, "image_height": int,
    "patient_id": "...",
    "age": float, "sex": "M"|"F",
    "view": "PA"|"AP"|...,
    "grounded_sentences": [
        {"sentence": "...", "finding_es": "...", "boxes": [[x,y,w,h], ...]},
        ...
    ]
  }

Finding names are in Spanish. The ontology mapper has Spanish aliases.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .base import BaseLoader, GroundedStudy
from claimguard_nmi.grounding.grounding import Annotation
from claimguard_nmi.data import load_ontology


class PadChestGRLoader(BaseLoader):
    site_name = "padchest_gr"

    def __init__(
        self,
        image_root: Path,
        annotations_json: Path,
        reports_tsv: Optional[Path] = None,
    ):
        self.image_root = Path(image_root)
        self.annotations_json = Path(annotations_json)
        self.reports_tsv = Path(reports_tsv) if reports_tsv else None
        self._mapper = load_ontology()

    @property
    def label_schema(self) -> frozenset:
        # PadChest-GR covers a broad set — use intersection with canonical ontology.
        covered = set()
        for cid in self._mapper.canonical_findings():
            covered.add(cid)
        return frozenset(covered)

    @staticmethod
    def _assign_split(patient_id: str) -> str:
        h = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 85:
            return "cal"
        return "test"

    def _load_reports(self) -> Dict[str, str]:
        if self.reports_tsv is None or not self.reports_tsv.exists():
            return {}
        reports: Dict[str, str] = {}
        with open(self.reports_tsv, "r") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                sid = row.get("study_id") or row.get("StudyID")
                text = row.get("report") or row.get("Report")
                if sid and text:
                    reports[sid] = text
        return reports

    _EXPECTED_KEYS_ANY_OF = (
        # BIMCV is not fully stable on naming; accept any of these for the
        # grounded-sentences field. Schema validation asserts at least one
        # variant is present on the first record (v2 review caught silent
        # empty-annotations on key mismatch).
        ("grounded_sentences",),
        ("sentences_grounded_boxes",),
        ("sentences",),
    )
    _EXPECTED_KEYS_REQUIRED = ("image_file",)

    def _validate_schema(self, records):
        if not records:
            raise ValueError("PadChest-GR annotations_json contains zero records")
        first = records[0]
        missing_required = [k for k in self._EXPECTED_KEYS_REQUIRED if k not in first]
        has_any_sentence_key = any(
            any(k in first for k in variant)
            for variant in self._EXPECTED_KEYS_ANY_OF
        )
        if missing_required or not has_any_sentence_key:
            raise ValueError(
                f"PadChest-GR schema mismatch on first record. "
                f"Required keys missing: {missing_required}. "
                f"Sentence-field not found among {self._EXPECTED_KEYS_ANY_OF}. "
                f"Got keys: {list(first.keys())}"
            )
        # Return which sentence-field to read.
        for variant in self._EXPECTED_KEYS_ANY_OF:
            for key in variant:
                if key in first:
                    return key
        raise ValueError("unreachable")

    def _extract_sentences(self, rec: dict, sentence_field: str) -> list:
        sents = rec.get(sentence_field, []) or []
        # Some releases wrap sentences under a per-sentence dict with nested "boxes".
        # Others use "box" (singular) with xyxy corners. Normalize to common form:
        #   [{'finding_surface': str, 'boxes_xywh': [[x,y,w,h], ...]}]
        out = []
        for s in sents:
            surface = (
                s.get("finding_es")
                or s.get("finding")
                or s.get("label")
                or (s["labels"][0] if isinstance(s.get("labels"), list) and s["labels"] else None)
                or s.get("sentence")
            )
            boxes = []
            if "boxes" in s:
                # xywh list of lists
                for b in s["boxes"]:
                    if len(b) == 4:
                        boxes.append(tuple(b))
            if "box" in s:
                b = s["box"]
                if len(b) == 4:
                    # Some releases store xyxy — detect heuristically: if
                    # b[2] <= b[0] or b[3] <= b[1] it's probably xyxy
                    # stored as xywh garbage; skip. Otherwise assume xyxy if
                    # b[2] > image width would be unlikely; default to xyxy -> xywh.
                    x1, y1, x2, y2 = b
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2 - x1, y2 - y1))
            if surface is None:
                continue
            out.append({"finding_surface": surface, "boxes_xywh": boxes})
        return out

    def iter_studies(self) -> Iterator[GroundedStudy]:
        with open(self.annotations_json, "r") as fh:
            records = json.load(fh)
        sentence_field = self._validate_schema(records)
        reports = self._load_reports()

        for rec in records:
            study_id = str(rec.get("study_id", rec.get("image_file", "unknown")))
            patient_id = str(rec.get("patient_id", study_id))
            H = int(rec.get("image_height", 1024))
            W = int(rec.get("image_width", 1024))

            anns: Dict[str, List[Annotation]] = {}
            for sent in self._extract_sentences(rec, sentence_field):
                canonical = self._mapper.canonicalize(sent["finding_surface"])
                if canonical is None:
                    continue
                mask = np.zeros((H, W), dtype=bool)
                for (x, y, bw, bh) in sent["boxes_xywh"]:
                    x0, y0 = max(int(x), 0), max(int(y), 0)
                    x1, y1 = min(int(x + bw), W), min(int(y + bh), H)
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1, x0:x1] = True
                anns.setdefault(canonical, []).append(
                    Annotation(finding=canonical, mask=mask)
                )

            yield GroundedStudy(
                study_id=study_id,
                patient_id=patient_id,
                image_path=self.image_root / rec["image_file"],
                image_shape=(H, W),
                annotations=anns,
                dataset_label_schema=self.label_schema,
                report_text=reports.get(study_id),
                split=self._assign_split(patient_id),
                metadata={
                    "dataset": "padchest_gr",
                    "image_format": "png",
                    "age": rec.get("age"),
                    "sex": rec.get("sex"),
                    "view": rec.get("view"),
                },
            )
