"""Unit tests for padchest_gr_validate (matching + Cohen kappa)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from v5.eval.padchest_gr_validate import (
    _compute_agreement_stats,
    _overlap,
    _tokenize,
    validate,
)


class TestTokenize:
    def test_strips_stopwords(self):
        assert "the" not in _tokenize("the cardiomegaly is present")
        assert "cardiomegaly" in _tokenize("the cardiomegaly is present")

    def test_short_words_dropped(self):
        assert _tokenize("is at of") == set()

    def test_lowercases(self):
        assert _tokenize("Cardiomegaly") == {"cardiomegaly"}


class TestOverlap:
    def test_identical(self):
        assert _overlap("mild cardiomegaly", "mild cardiomegaly") == pytest.approx(1.0)

    def test_disjoint(self):
        assert _overlap("cardiomegaly", "pneumothorax") == pytest.approx(0.0)

    def test_partial(self):
        sim = _overlap("mild cardiomegaly present", "cardiomegaly observed")
        assert 0.0 < sim < 1.0


class TestAgreementStats:
    def test_perfect_agreement(self):
        c = Counter({("SUPPORTED", "SUPPORTED"): 50, ("CONTRADICTED", "CONTRADICTED"): 50})
        stats = _compute_agreement_stats(c)
        assert stats["kappa"] == pytest.approx(1.0)
        assert stats["precision"] == pytest.approx(1.0)
        assert stats["recall"] == pytest.approx(1.0)
        assert stats["f1"] == pytest.approx(1.0)

    def test_chance_agreement(self):
        c = Counter({
            ("SUPPORTED", "SUPPORTED"): 25, ("SUPPORTED", "CONTRADICTED"): 25,
            ("CONTRADICTED", "SUPPORTED"): 25, ("CONTRADICTED", "CONTRADICTED"): 25,
        })
        stats = _compute_agreement_stats(c)
        assert stats["kappa"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_confusion(self):
        stats = _compute_agreement_stats(Counter())
        assert stats["kappa"] is None

    def test_uncertain_tracked_separately(self):
        c = Counter({
            ("SUPPORTED", "SUPPORTED"): 40,
            ("SUPPORTED", "UNCERTAIN"): 10,
            ("CONTRADICTED", "CONTRADICTED"): 40,
        })
        stats = _compute_agreement_stats(c)
        assert stats["n_uncertain"] == 10


class TestValidateEndToEnd:
    def test_simple_match(self, tmp_path):
        records = tmp_path / "padchest_records.jsonl"
        with open(records, "w") as f:
            f.write(json.dumps({
                "study_id": "sid1",
                "boxes": [
                    {"sentence_en": "mild cardiomegaly is present",
                     "finding": "cardiomegaly", "is_positive": True, "bbox": [0, 0, 1, 1]},
                    {"sentence_en": "no pleural effusion",
                     "finding": "pleural_effusion", "is_positive": False, "bbox": [0, 0, 1, 1]},
                ]
            }) + "\n")

        ensemble = tmp_path / "ensemble.jsonl"
        with open(ensemble, "w") as f:
            f.write(json.dumps({
                "claim_id": "c1", "image_id": "sid1",
                "claim_text": "cardiomegaly present mild",
                "final_label": "SUPPORTED", "confidence": "HIGH",
            }) + "\n")
            f.write(json.dumps({
                "claim_id": "c2", "image_id": "sid1",
                "claim_text": "pleural effusion absent",
                "final_label": "CONTRADICTED", "confidence": "HIGH",
            }) + "\n")

        out = tmp_path / "validation.jsonl"
        stats = validate(records, ensemble, out)
        assert stats["n_ground_truth_sentences"] == 2
        assert stats["n_matched_at_min_similarity"] >= 1
