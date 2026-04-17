"""NIH ChestX-ray14 BBox subset loader.

The NIH release ships a ``BBox_List_2017.csv`` with ~1000 radiologist-
drawn bounding boxes across 8 pathology classes. CSV columns:

  Image Index, Finding Label, Bbox [x, y, w, h]

Image files live in ``images_*.tar.gz`` archives; we expect the archives
have been unpacked to a flat ``<root>/images/`` directory.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .base import BaseLoader, GroundedStudy
from claimguard_nmi.grounding.grounding import Annotation
from claimguard_nmi.data import load_ontology


_NIH_TO_CANONICAL = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural_effusion",
    "Infiltrate": "infiltration",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "lung_opacity",
    "Pneumothorax": "pneumothorax",
}


class NIHBBoxLoader(BaseLoader):
    site_name = "nih_bbox"

    def __init__(
        self,
        image_root: Path,
        bbox_csv: Path,
        metadata_csv: Optional[Path] = None,
        image_shape: tuple = (1024, 1024),
    ):
        self.image_root = Path(image_root)
        self.bbox_csv = Path(bbox_csv)
        self.metadata_csv = Path(metadata_csv) if metadata_csv else None
        self._image_shape = image_shape
        self._mapper = load_ontology()

    @property
    def label_schema(self) -> frozenset:
        return frozenset(set(_NIH_TO_CANONICAL.values()))

    def _patient_from_image(self, image_index: str) -> str:
        # NIH image ids look like "00000001_000.png"; prefix before `_` is patient id.
        return image_index.split("_", 1)[0]

    def _assign_split(self, patient_id: str) -> str:
        h = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 85:
            return "cal"
        return "test"

    def _load_metadata(self) -> Dict[str, Dict[str, object]]:
        if self.metadata_csv is None or not self.metadata_csv.exists():
            return {}
        out: Dict[str, Dict[str, object]] = {}
        with open(self.metadata_csv, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                img = row.get("Image Index")
                if not img:
                    continue
                out[img] = {
                    "age": row.get("Patient Age"),
                    "sex": row.get("Patient Gender"),
                    "view": row.get("View Position"),
                }
        return out

    def iter_studies(self) -> Iterator[GroundedStudy]:
        H, W = self._image_shape
        per_image: Dict[str, List[tuple]] = {}
        with open(self.bbox_csv, "r") as fh:
            # Detect delimiter (NIH release CSV + community mirrors sometimes use `;`).
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            # Normalize fieldnames: strip whitespace and surrounding brackets that
            # NIH's CSV header carries (e.g., "Bbox [x" with trailing space).
            if reader.fieldnames:
                reader.fieldnames = [f.strip() for f in reader.fieldnames]

            def _num(row, *keys):
                for k in keys:
                    v = row.get(k)
                    if v is None:
                        continue
                    try:
                        return float(str(v).strip())
                    except ValueError:
                        continue
                return None

            for row in reader:
                img = (row.get("Image Index") or "").strip()
                finding = (row.get("Finding Label") or "").strip()
                x = _num(row, "Bbox [x", "x", "X")
                y = _num(row, "y", "Y")
                w = _num(row, "w", "W")
                h = _num(row, "h]", "h", "H")
                if None in (x, y, w, h) or not img:
                    continue
                per_image.setdefault(img, []).append((finding, x, y, w, h))

        meta_by_image = self._load_metadata()

        for img, boxes in per_image.items():
            anns: Dict[str, List[Annotation]] = {}
            for (finding, x, y, w, h) in boxes:
                canonical = _NIH_TO_CANONICAL.get(finding)
                if canonical is None:
                    continue
                mask = np.zeros((H, W), dtype=bool)
                x0, y0 = max(int(x), 0), max(int(y), 0)
                x1, y1 = min(int(x + w), W), min(int(y + h), H)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = True
                anns.setdefault(canonical, []).append(
                    Annotation(finding=canonical, mask=mask)
                )
            pid = self._patient_from_image(img)
            yield GroundedStudy(
                study_id=img,
                patient_id=pid,
                image_path=self.image_root / img,
                image_shape=(H, W),
                annotations=anns,
                dataset_label_schema=self.label_schema,
                split=self._assign_split(pid),
                metadata={
                    "dataset": "nih_bbox",
                    "image_format": "png",
                    **meta_by_image.get(img, {}),
                },
            )
