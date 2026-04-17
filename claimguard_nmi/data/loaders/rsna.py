"""RSNA Pneumonia Detection Challenge loader.

Dataset layout (as distributed via Kaggle):
  <root>/
    stage_2_train_images/*.dcm
    stage_2_test_images/*.dcm
    stage_2_train_labels.csv              # patientId,x,y,width,height,Target
    stage_2_detailed_class_info.csv       # patientId,class

Label schema: single canonical finding ``lung_opacity``.
Only positive rows (Target == 1) have bounding boxes; negatives have empty x/y/w/h.
Images are DICOM; mask rasterization uses the DICOM native shape.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import BaseLoader, GroundedStudy
from claimguard_nmi.grounding.grounding import Annotation


class RSNALoader(BaseLoader):
    site_name = "rsna_pneumonia"

    def __init__(
        self,
        image_root: Path,
        labels_csv: Path,
        image_shape: tuple = (1024, 1024),
        split_seeds: tuple = (17, 19, 23),
    ):
        self.image_root = Path(image_root)
        self.labels_csv = Path(labels_csv)
        self._image_shape = image_shape
        self._split_seeds = split_seeds

    @property
    def label_schema(self) -> frozenset:
        return frozenset({"lung_opacity"})

    def _assign_split(self, patient_id: str) -> str:
        """Deterministic patient-level train/cal/test split.

        Uses md5 hash of patient_id mod 100. 70 / 15 / 15.
        """
        h = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 85:
            return "cal"
        return "test"

    def _boxes_by_patient(self) -> dict:
        boxes: dict = {}
        with open(self.labels_csv, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = row["patientId"]
                if pid not in boxes:
                    boxes[pid] = []
                if row.get("Target", "0") != "1":
                    continue
                try:
                    x, y, w, h = (float(row[k]) for k in ("x", "y", "width", "height"))
                except (KeyError, ValueError):
                    continue
                boxes[pid].append((x, y, w, h))
        return boxes

    def iter_studies(self) -> Iterator[GroundedStudy]:
        boxes_by_pid = self._boxes_by_patient()
        H, W = self._image_shape
        n_missing = 0
        for pid, bbs in boxes_by_pid.items():
            image_path = self.image_root / f"{pid}.dcm"
            # Skip rows whose DICOM file is missing on disk. We only check
            # existence when the image_root exists — fixture-based tests pass
            # a non-existent root to avoid DICOM dependencies.
            if self.image_root.exists() and not image_path.exists():
                n_missing += 1
                continue
            if bbs:
                mask = np.zeros((H, W), dtype=bool)
                for (x, y, w, h) in bbs:
                    x0, y0 = max(int(x), 0), max(int(y), 0)
                    x1, y1 = min(int(x + w), W), min(int(y + h), H)
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1, x0:x1] = True
                anns = {"lung_opacity": [Annotation(finding="lung_opacity", mask=mask)]}
            else:
                anns = {}
            yield GroundedStudy(
                study_id=pid,
                patient_id=pid,
                image_path=image_path,
                image_shape=(H, W),
                annotations=anns,
                dataset_label_schema=self.label_schema,
                split=self._assign_split(pid),
                metadata={"dataset": "rsna_pneumonia", "image_format": "dicom"},
            )
