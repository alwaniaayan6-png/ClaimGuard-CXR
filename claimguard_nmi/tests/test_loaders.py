"""Fixture-based loader tests. No real dataset downloads; we synthesize
minimal examples on disk to exercise each parser's core paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimguard_nmi.data.loaders import (
    ChestXDet10Loader,
    NIHBBoxLoader,
    PadChestGRLoader,
    RSNALoader,
    SIIMLoader,
)


def test_rsna_loader_parses_positive_and_negative(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "patientId,x,y,width,height,Target\n"
        "p001,100.0,200.0,50.0,60.0,1\n"
        "p002,,,,,0\n"
    )
    loader = RSNALoader(image_root=tmp_path / "images", labels_csv=labels)
    studies = list(loader.iter_studies())
    assert len(studies) == 2
    pos = next(s for s in studies if s.study_id == "p001")
    neg = next(s for s in studies if s.study_id == "p002")
    assert "lung_opacity" in pos.annotations
    assert pos.annotations["lung_opacity"][0].mask.any()
    assert neg.annotations == {}
    assert pos.dataset_label_schema == frozenset({"lung_opacity"})


def test_siim_loader_parses_rle_and_missing(tmp_path: Path):
    rle = tmp_path / "rle.csv"
    # Smallest possible positive RLE + one negative
    rle.write_text(
        "ImageId,EncodedPixels\n"
        "img1,1 4\n"
        "img2,-1\n"
    )
    loader = SIIMLoader(
        image_root=tmp_path / "images",
        rle_csv=rle,
        image_shape=(4, 4),
    )
    studies = {s.study_id: s for s in loader.iter_studies()}
    assert "img1" in studies and "img2" in studies
    assert studies["img1"].annotations["pneumothorax"][0].mask.any()
    assert studies["img2"].annotations == {}


def test_chestxdet10_loader_maps_only_canonical_classes(tmp_path: Path):
    coco = {
        "images": [
            {"id": 1, "file_name": "a.png", "height": 256, "width": 256},
        ],
        "categories": [
            {"id": 1, "name": "Atelectasis"},
            {"id": 2, "name": "Emphysema"},   # NOT in _CATEGORY_MAP -> should be dropped
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [100, 100, 40, 40]},
        ],
    }
    coco_path = tmp_path / "chestx.json"
    coco_path.write_text(json.dumps(coco))
    loader = ChestXDet10Loader(
        image_root=tmp_path / "imgs",
        coco_json_paths=[coco_path],
    )
    studies = list(loader.iter_studies())
    assert len(studies) == 1
    assert "atelectasis" in studies[0].annotations
    # Emphysema should NOT appear
    assert all(k != "emphysema" for k in studies[0].annotations)


def test_padchest_gr_loader_maps_spanish_surface_forms(tmp_path: Path):
    records = [
        {
            "study_id": "s1",
            "image_file": "s1.png",
            "image_width": 512,
            "image_height": 512,
            "patient_id": "p1",
            "age": 65, "sex": "F", "view": "PA",
            "grounded_sentences": [
                {"finding_es": "derrame pleural", "boxes": [[50, 50, 100, 100]], "sentence": "..."},
                {"finding_es": "unicorn_finding", "boxes": [[0, 0, 10, 10]], "sentence": "..."},
            ],
        }
    ]
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(records))
    loader = PadChestGRLoader(
        image_root=tmp_path / "imgs",
        annotations_json=ann_path,
    )
    studies = list(loader.iter_studies())
    assert len(studies) == 1
    anns = studies[0].annotations
    assert "pleural_effusion" in anns  # Spanish canonicalized
    assert all(k != "unicorn_finding" for k in anns)
    assert studies[0].metadata["sex"] == "F"


def test_nih_bbox_loader_parses_canonical_findings(tmp_path: Path):
    bbox = tmp_path / "boxes.csv"
    bbox.write_text(
        "Image Index,Finding Label,x,y,w,h\n"
        "00000001_000.png,Atelectasis,100,100,80,80\n"
        "00000001_000.png,Pneumothorax,200,200,40,40\n"
    )
    loader = NIHBBoxLoader(
        image_root=tmp_path / "imgs",
        bbox_csv=bbox,
    )
    studies = list(loader.iter_studies())
    assert len(studies) == 1
    anns = studies[0].annotations
    assert "atelectasis" in anns
    assert "pneumothorax" in anns
    # Patient id should be derived from "00000001_000.png" -> "00000001"
    assert studies[0].patient_id == "00000001"


def test_patient_split_deterministic(tmp_path: Path):
    labels = tmp_path / "l.csv"
    labels.write_text(
        "patientId,x,y,width,height,Target\n"
        + "\n".join(f"p{i},,,,,0" for i in range(200)) + "\n"
    )
    loader1 = RSNALoader(image_root=tmp_path / "x", labels_csv=labels)
    loader2 = RSNALoader(image_root=tmp_path / "x", labels_csv=labels)
    assignments1 = [s.split for s in loader1.iter_studies()]
    assignments2 = [s.split for s in loader2.iter_studies()]
    assert assignments1 == assignments2
    # Approximate 70/15/15 split
    n = len(assignments1)
    n_train = sum(1 for a in assignments1 if a == "train")
    assert 0.55 * n < n_train < 0.85 * n
