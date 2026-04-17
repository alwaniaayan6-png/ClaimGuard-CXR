"""ChestX-Det10 loader (Deepwise-AILab, GitHub).

Dataset layout:
  <root>/
    train_data/*.png
    test_data/*.png
    ChestX_Det_train.json        # COCO-style: images, annotations, categories
    ChestX_Det_test.json

Categories (canonical mapping in configs/ontology.yaml):
  Atelectasis, Calcification, Consolidation, Effusion, Emphysema,
  Fibrosis, Fracture, Mass, Nodule, Pneumothorax
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np

from .base import BaseLoader, GroundedStudy
from claimguard_nmi.grounding.grounding import Annotation
from claimguard_nmi.data import load_ontology


def _rasterize_polygon_segmentation(segmentation, bbox, shape):
    """Rasterize a COCO-style polygon segmentation to a boolean mask.

    ``segmentation`` may be:
      - a list of polygons (each polygon is a flat list of x,y,x,y,...)
      - a single polygon (flat list of floats)
      - None / empty — fall back to bbox

    ``bbox`` is [x, y, w, h]. We fall back to a rectangular bbox mask only
    when no polygon segmentation is available.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    polygons = []
    if segmentation:
        if isinstance(segmentation, list) and segmentation:
            if isinstance(segmentation[0], (list, tuple)):
                polygons = list(segmentation)
            else:
                polygons = [segmentation]

    if polygons:
        try:
            from PIL import Image, ImageDraw  # type: ignore

            img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(img)
            for poly in polygons:
                pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, outline=1, fill=1)
            mask = np.asarray(img, dtype=bool)
            return mask
        except ImportError:
            pass  # fall through to bbox fallback

    # Bbox fallback only when no polygon was supplied or PIL missing.
    if bbox and len(bbox) == 4:
        x, y, bw, bh = bbox
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


# ChestX-Det10 native categories -> canonical ids. Unmapped classes are skipped.
_CATEGORY_MAP = {
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Consolidation": "lung_opacity",
    "Effusion": "pleural_effusion",
    "Fibrosis": "fibrosis",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumothorax": "pneumothorax",
    # Emphysema and Fracture do not map to our canonical ontology; excluded.
}


class ChestXDet10Loader(BaseLoader):
    site_name = "chestxdet10"

    def __init__(
        self,
        image_root: Path,
        coco_json_paths: List[Path],
        image_shape: tuple = (1024, 1024),
    ):
        self.image_root = Path(image_root)
        self.coco_json_paths = [Path(p) for p in coco_json_paths]
        self._image_shape = image_shape
        self._mapper = load_ontology()

    @property
    def label_schema(self) -> frozenset:
        return frozenset(set(_CATEGORY_MAP.values()))

    @staticmethod
    def _assign_split(study_id: str) -> str:
        h = int(hashlib.md5(study_id.encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 85:
            return "cal"
        return "test"

    def iter_studies(self) -> Iterator[GroundedStudy]:
        H, W = self._image_shape
        for json_path in self.coco_json_paths:
            with open(json_path, "r") as fh:
                data = json.load(fh)
            id_to_image = {img["id"]: img for img in data.get("images", [])}
            id_to_cat = {c["id"]: c["name"] for c in data.get("categories", [])}
            annotations_by_image: Dict[int, List[dict]] = {}
            for ann in data.get("annotations", []):
                annotations_by_image.setdefault(ann["image_id"], []).append(ann)

            for image_id, image_meta in id_to_image.items():
                h = image_meta.get("height", H)
                w = image_meta.get("width", W)
                file_name = image_meta["file_name"]
                anns: Dict[str, List[Annotation]] = {}
                for raw in annotations_by_image.get(image_id, []):
                    cat_name = id_to_cat.get(raw["category_id"])
                    canonical = _CATEGORY_MAP.get(cat_name)
                    if canonical is None:
                        continue
                    # ChestX-Det10 ships polygon segmentations. Rasterize the
                    # polygon (not the bbox) — using the bbox silently halves
                    # IoU on every grounded decision (v2 review).
                    mask = _rasterize_polygon_segmentation(
                        raw.get("segmentation"), raw.get("bbox"), (h, w),
                    )
                    anns.setdefault(canonical, []).append(
                        Annotation(finding=canonical, mask=mask)
                    )
                yield GroundedStudy(
                    study_id=str(image_id),
                    patient_id=file_name.rsplit(".", 1)[0],
                    image_path=self.image_root / file_name,
                    image_shape=(h, w),
                    annotations=anns,
                    dataset_label_schema=self.label_schema,
                    split=self._assign_split(file_name),
                    metadata={"dataset": "chestxdet10", "image_format": "png"},
                )
