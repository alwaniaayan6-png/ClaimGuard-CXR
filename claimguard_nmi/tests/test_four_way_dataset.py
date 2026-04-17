"""Unit tests for the four-way training dataset + collate.

These tests avoid loading any real CXR images; they use a stub tokenizer
and the dataset's default synthetic-tensor image loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from claimguard_nmi.training.four_way_dataset import (
    FourWayTrainingDataset,
    TrainingPair,
    collate_four_way,
)


def _stub_tokenizer(text: str, seq_len: int = 8):
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[0] = 1  # fake BOS
    ids[1] = min(len(text), 100)
    mask = torch.ones(seq_len, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask}


def _make_pairs(n: int = 4):
    return [
        TrainingPair(
            claim_text=f"claim {i}",
            evidence_supp=f"supp evidence {i}",
            evidence_contra=f"contra evidence {i}",
            image_path=Path(f"/tmp/img_{i}.png"),
            patient_id=f"p{i}",
            claim_finding="lung_opacity",
            claim_laterality="left",
            claim_region="lower",
        )
        for i in range(n)
    ]


def test_dataset_yields_expected_keys_and_labels():
    pairs = _make_pairs(3)
    pool = {f"p{i}": [Path(f"/tmp/img_{i}.png")] for i in range(3)}
    ds = FourWayTrainingDataset(
        pairs=pairs, tokenizer=_stub_tokenizer,
        image_pool_by_patient=pool,
    )
    sample = ds[0]
    expected_keys = {
        "claim_ids", "claim_mask",
        "evidence_supp_ids", "evidence_supp_mask",
        "evidence_contra_ids", "evidence_contra_mask",
        "image_correct", "image_swap", "image_mask",
        "label_v1", "label_v2", "label_v3", "label_v4",
    }
    assert set(sample.keys()) == expected_keys
    assert sample["label_v1"].item() == 0
    assert sample["label_v2"].item() == 1
    assert sample["label_v3"].item() == 1
    assert sample["label_v4"].item() == 1


def test_image_swap_draws_from_other_patient():
    pairs = _make_pairs(2)
    pool = {
        "p0": [Path("/tmp/img_0.png")],
        "p1": [Path("/tmp/img_1.png")],
    }
    ds = FourWayTrainingDataset(
        pairs=pairs, tokenizer=_stub_tokenizer,
        image_pool_by_patient=pool,
    )
    sample = ds[0]  # p0 — should swap to p1's image
    # Deterministic stub image loader: different paths yield different tensors.
    assert not torch.allclose(sample["image_correct"], sample["image_swap"])


def test_image_mask_is_all_zeros():
    pairs = _make_pairs(1)
    pool = {"p0": [Path("/tmp/img_0.png")], "p_other": [Path("/tmp/other.png")]}
    ds = FourWayTrainingDataset(
        pairs=pairs, tokenizer=_stub_tokenizer,
        image_pool_by_patient=pool,
    )
    sample = ds[0]
    assert torch.all(sample["image_mask"] == 0)


def test_collate_four_way_stacks_correctly():
    pairs = _make_pairs(4)
    pool = {f"p{i}": [Path(f"/tmp/img_{i}.png")] for i in range(4)}
    ds = FourWayTrainingDataset(
        pairs=pairs, tokenizer=_stub_tokenizer,
        image_pool_by_patient=pool,
    )
    batch = collate_four_way([ds[i] for i in range(4)])
    assert batch["claim_ids"].shape == (4, 8)
    assert batch["image_correct"].dim() == 4
    assert batch["image_correct"].shape[0] == 4
    assert batch["label_v1"].shape == (4,)
