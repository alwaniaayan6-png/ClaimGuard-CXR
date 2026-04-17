"""SIIM-ACR Pneumothorax Segmentation loader.

Dataset layout (Kaggle):
  <root>/
    dicom-images-train/<series>/<study>/<instance>.dcm
    train-rle.csv                  # ImageId,EncodedPixels (RLE)
    test-rle.csv

Label schema: single canonical finding ``pneumothorax``.
Masks are RLE-encoded; ``-1`` means "no pneumothorax on this image".
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterator, List

import numpy as np

from .base import BaseLoader, GroundedStudy
from claimguard_nmi.grounding.grounding import Annotation


class SIIMLoader(BaseLoader):
    site_name = "siim_pneumothorax"

    def __init__(
        self,
        image_root: Path,
        rle_csv: Path,
        image_shape: tuple = (1024, 1024),
    ):
        self.image_root = Path(image_root)
        self.rle_csv = Path(rle_csv)
        self._image_shape = image_shape

    @property
    def label_schema(self) -> frozenset:
        return frozenset({"pneumothorax"})

    @staticmethod
    def _rle_to_mask(rle: str, shape: tuple) -> np.ndarray:
        H, W = shape
        mask = np.zeros(H * W, dtype=bool)
        if rle.strip() in {"-1", ""}:
            return mask.reshape((H, W))
        tokens = [int(t) for t in rle.split()]
        starts = tokens[0::2]
        lengths = tokens[1::2]
        for s, l in zip(starts, lengths):
            s = max(s - 1, 0)
            mask[s:s + l] = True
        # Kaggle uses column-major (Fortran) RLE.
        return mask.reshape((W, H)).T

    def _assign_split(self, image_id: str) -> str:
        h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 85:
            return "cal"
        return "test"

    def _rles_by_image(self) -> dict:
        rles: dict = {}
        with open(self.rle_csv, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                iid = row["ImageId"]
                rle = row.get("EncodedPixels", "") or row.get(" EncodedPixels", "")
                rles.setdefault(iid, []).append(rle.strip())
        return rles

    def _find_image(self, image_id: str) -> Path:
        # SIIM nests images deeply; use a suffix glob once per image at iteration.
        hits = list(self.image_root.rglob(f"{image_id}.dcm"))
        if not hits:
            return self.image_root / f"{image_id}.dcm"  # may not exist; downstream handles
        return hits[0]

    def iter_studies(self) -> Iterator[GroundedStudy]:
        H, W = self._image_shape
        for iid, rles in self._rles_by_image().items():
            masks = [self._rle_to_mask(r, (H, W)) for r in rles]
            positive = [m for m in masks if m.any()]
            if positive:
                anns = {
                    "pneumothorax": [
                        Annotation(finding="pneumothorax", mask=m) for m in positive
                    ]
                }
            else:
                anns = {}
            yield GroundedStudy(
                study_id=iid,
                patient_id=iid,  # patient ≡ image here
                image_path=self._find_image(iid),
                image_shape=(H, W),
                annotations=anns,
                dataset_label_schema=self.label_schema,
                split=self._assign_split(iid),
                metadata={"dataset": "siim_pneumothorax", "image_format": "dicom"},
            )
