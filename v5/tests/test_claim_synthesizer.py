"""Unit tests for v5/data/claim_synthesizer.py + grounding target projection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from collections import Counter
from unittest.mock import patch

import pytest
import torch

from v5.data.claim_matcher import Annotation
from v5.data.claim_synthesizer import (
    default_site_findings,
    synthesize_claims_for_image,
)


# ---------------------------------------------------------------------------
# Label-distribution guarantees for each detection-only site
# ---------------------------------------------------------------------------


def _build_synth_for(site: str, present_findings: list[str]) -> list:
    anns = [
        Annotation(
            image_id=f"{site}_img_1",
            finding=f,
            laterality="L",
            bbox=(0.2, 0.3, 0.5, 0.6),
            source=site,
        )
        for f in present_findings
    ]
    return synthesize_claims_for_image(
        image_id=f"{site}_img_1",
        annotations=anns,
        source=site,
        all_findings_in_site=default_site_findings(site),
        emit_negatives=True,
        emit_contradicted_positives=True,
    )


def test_chestx_det10_produces_supported_and_contradicted():
    """Multi-finding taxonomy → both label classes represented."""
    claims = _build_synth_for("chestx_det10", ["pneumonia"])
    labels = Counter(c.gt_label for c in claims)
    assert labels["SUPPORTED"] >= 1
    assert labels["CONTRADICTED"] >= 1, f"no CONTRADICTED claim produced; got {labels}"


def test_rsna_single_finding_still_emits_contradicted_pair():
    """Single-finding site (RSNA) + finding-absent image → emits a
    contradictory pair ('no X'=SUPPORTED, 'there is X'=CONTRADICTED) that is
    pedagogically useful as hard negatives.
    """
    # Empty annotations → "pneumonia" is absent
    claims = synthesize_claims_for_image(
        image_id="rsna_img_1",
        annotations=[],
        source="rsna_pneumonia",
        all_findings_in_site=default_site_findings("rsna_pneumonia"),
        emit_negatives=True,
        emit_contradicted_positives=True,
    )
    labels = Counter(c.gt_label for c in claims)
    assert labels["SUPPORTED"] >= 1
    assert labels["CONTRADICTED"] >= 1


def test_siim_single_finding_absent_also_emits_pair():
    claims = synthesize_claims_for_image(
        image_id="siim_img_1",
        annotations=[],
        source="siim_acr",
        all_findings_in_site=default_site_findings("siim_acr"),
    )
    labels = Counter(c.gt_label for c in claims)
    assert labels["CONTRADICTED"] >= 1


def test_object_cxr_single_finding_absent_also_emits_pair():
    claims = synthesize_claims_for_image(
        image_id="obj_img_1",
        annotations=[],
        source="object_cxr",
        all_findings_in_site=default_site_findings("object_cxr"),
    )
    labels = Counter(c.gt_label for c in claims)
    assert labels["CONTRADICTED"] >= 1


def test_synthesizer_positive_claim_has_polarity_present():
    anns = [Annotation(image_id="img", finding="pneumonia", source="rsna")]
    claims = synthesize_claims_for_image(
        image_id="img",
        annotations=anns,
        source="rsna_pneumonia",
        all_findings_in_site=["pneumonia"],
        emit_negatives=False,
        emit_contradicted_positives=False,
    )
    assert len(claims) == 1
    assert claims[0].gt_label == "SUPPORTED"
    assert claims[0].structured.comparison == "present"


def test_synthesizer_negation_has_polarity_negated_tag():
    """Bug B1 regression guard: negative claims must carry the modifier tag
    that the matcher understands, in case the matcher is ever re-invoked.
    """
    claims = synthesize_claims_for_image(
        image_id="img",
        annotations=[],
        source="rsna_pneumonia",
        all_findings_in_site=["pneumonia"],
        emit_negatives=True,
        emit_contradicted_positives=False,
    )
    neg_claims = [c for c in claims if "negative" in c.structured.modifier_tags]
    assert len(neg_claims) == 1
    assert "polarity_negated" in neg_claims[0].structured.modifier_tags


# ---------------------------------------------------------------------------
# Grounding-target projection in GroundBenchDataset
# ---------------------------------------------------------------------------


def test_grounding_target_projection_non_null_bbox():
    """Bug B3 regression guard: when a row has grounding_bbox, the dataset
    must emit a (14,14) binary target with grounding_mask=True.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from v5.train import GroundBenchDataset, V5TrainConfig
    from v5.model import build_v5_tokenizer, V5Config

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jsonl = td / "mini.jsonl"
        rows = [
            {
                "image_path": "x.png",
                "claim_text": "test supporting claim",
                "evidence_text": "",
                "gt_label": "SUPPORTED",
                "grounding_bbox": [0.2, 0.3, 0.5, 0.6],
            },
            {
                "image_path": "y.png",
                "claim_text": "test contradicted claim",
                "evidence_text": "",
                "gt_label": "CONTRADICTED",
                "grounding_bbox": None,
            },
        ]
        with jsonl.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tok = build_v5_tokenizer(V5Config())
        cfg = V5TrainConfig(train_jsonl=jsonl, val_jsonl=jsonl, out_dir=td, image_root=td)
        with patch.object(GroundBenchDataset, "_load_image", return_value=torch.zeros(3, 224, 224)):
            ds = GroundBenchDataset(jsonl, td, tok, cfg)
            r0 = ds[0]
            r1 = ds[1]
        assert r0["grounding_mask"].item() is True
        assert r0["grounding_target"].shape == (14, 14)
        assert r0["grounding_target"].sum().item() > 0
        assert r1["grounding_mask"].item() is False
        assert r1["grounding_target"].sum().item() == 0


def test_grounding_target_degenerate_bbox_clamped():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from v5.train import GroundBenchDataset, V5TrainConfig
    from v5.model import build_v5_tokenizer, V5Config

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jsonl = td / "mini.jsonl"
        # Degenerate zero-area bbox: c2>c1 and r2>r1 must be enforced.
        rows = [
            {
                "image_path": "x.png",
                "claim_text": "c",
                "evidence_text": "",
                "gt_label": "SUPPORTED",
                "grounding_bbox": [0.5, 0.5, 0.5, 0.5],
            }
        ]
        with jsonl.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tok = build_v5_tokenizer(V5Config())
        cfg = V5TrainConfig(train_jsonl=jsonl, val_jsonl=jsonl, out_dir=td, image_root=td)
        with patch.object(GroundBenchDataset, "_load_image", return_value=torch.zeros(3, 224, 224)):
            ds = GroundBenchDataset(jsonl, td, tok, cfg)
            r = ds[0]
        # Zero-area bbox → mask remains False, target all zeros.
        assert r["grounding_mask"].item() is False
        assert r["grounding_target"].sum().item() == 0


# ---------------------------------------------------------------------------
# HO filter row-idx alignment guard
# ---------------------------------------------------------------------------


def test_ho_filter_row_ordering_matches_groundbench_dataset():
    """Regression guard on first-review concern 9(c): both datasets must
    filter rows identically and preserve file order so per-row weights align.
    """
    pytest.importorskip("transformers")
    from v5.train import GroundBenchDataset, V5TrainConfig
    from v5.ho_filter import _HODataset
    from v5.model import build_v5_tokenizer, V5Config

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jsonl = td / "mini.jsonl"
        rows = [
            {"image_path": "a.png", "claim_text": "s1", "evidence_text": "", "gt_label": "SUPPORTED"},
            {"image_path": "b.png", "claim_text": "?", "evidence_text": "", "gt_label": "NO_GT"},  # should be filtered
            {"image_path": "c.png", "claim_text": "s2", "evidence_text": "", "gt_label": "CONTRADICTED"},
            {"image_path": "d.png", "claim_text": "s3", "evidence_text": "", "gt_label": "SUPPORTED"},
        ]
        with jsonl.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tok = build_v5_tokenizer(V5Config())
        cfg = V5TrainConfig(train_jsonl=jsonl, val_jsonl=jsonl, out_dir=td, image_root=td)
        with patch.object(GroundBenchDataset, "_load_image", return_value=torch.zeros(3, 224, 224)):
            gb = GroundBenchDataset(jsonl, td, tok, cfg)
        ho = _HODataset(jsonl, tok, max_text_tokens=64)
        assert len(gb.rows) == len(ho.rows) == 3
        for i in range(3):
            assert gb.rows[i]["image_path"] == ho.rows[i]["image_path"]
