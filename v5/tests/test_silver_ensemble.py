"""Unit tests for silver_ensemble (Krippendorff alpha + decision rule)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v5.eval.silver_ensemble import (
    _decide,
    combine_labels,
    krippendorff_alpha,
)


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        labels = ["SUPPORTED", "CONTRADICTED", "UNCERTAIN", "SUPPORTED"]
        alpha = krippendorff_alpha([labels, labels])
        assert alpha == pytest.approx(1.0)

    def test_three_coder_perfect_agreement(self):
        labels = ["SUPPORTED"] * 10 + ["CONTRADICTED"] * 10
        alpha = krippendorff_alpha([labels, labels, labels])
        assert alpha == pytest.approx(1.0)

    def test_positive_partial_agreement(self):
        a = ["S", "S", "S", "S", "C", "C", "C", "C"]
        b = ["S", "S", "S", "C", "S", "C", "C", "C"]
        alpha = krippendorff_alpha([a, b])
        assert 0.0 < alpha < 1.0

    def test_disagreement_is_negative(self):
        alpha = krippendorff_alpha([["S"] * 4, ["C"] * 4])
        assert alpha < 0

    def test_single_category_returns_none(self):
        alpha = krippendorff_alpha([["S"] * 4, ["S"] * 4])
        assert alpha is None

    def test_empty_input_returns_none(self):
        assert krippendorff_alpha([[]]) is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            krippendorff_alpha([["S"], ["S", "C"]])


class TestDecisionRule:
    def test_unanimous_supported(self):
        label, conf, flag = _decide("SUPPORTED", "SUPPORTED", "SUPPORTED")
        assert label == "SUPPORTED"
        assert conf == "HIGH"
        assert flag is False

    def test_unanimous_contradicted(self):
        label, conf, flag = _decide("CONTRADICTED", "CONTRADICTED", "CONTRADICTED")
        assert label == "CONTRADICTED"
        assert conf == "HIGH"

    def test_majority_with_uncertain_is_med(self):
        label, conf, flag = _decide("SUPPORTED", "SUPPORTED", "UNCERTAIN")
        assert label == "SUPPORTED"
        assert conf == "MED"
        assert flag is False

    def test_majority_with_conflict_is_low(self):
        label, conf, flag = _decide("SUPPORTED", "SUPPORTED", "CONTRADICTED")
        assert label == "SUPPORTED"
        assert conf == "LOW"
        assert flag is True

    def test_all_disagree(self):
        label, conf, flag = _decide("SUPPORTED", "CONTRADICTED", "UNCERTAIN")
        assert label == "UNCERTAIN"
        assert conf == "EXCLUDED"
        assert flag is True


class TestCombineLabels:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_combine_unanimous(self, tmp_path):
        rows = [
            {"claim_id": "c1", "image_id": "i1", "rrg_model": "maira", "claim_text": "x",
             "label": "SUPPORTED"},
            {"claim_id": "c2", "image_id": "i1", "rrg_model": "maira", "claim_text": "y",
             "label": "CONTRADICTED"},
        ]
        g = tmp_path / "green.jsonl"
        r = tmp_path / "radfact.jsonl"
        v = tmp_path / "vert.jsonl"
        self._write_jsonl(g, rows)
        self._write_jsonl(r, rows)
        self._write_jsonl(v, rows)
        out = tmp_path / "ensemble.jsonl"
        stats = combine_labels(g, r, v, out)
        assert stats["n_shared_claims"] == 2
        assert stats["final_label_counts"].get("SUPPORTED") == 1
        assert stats["final_label_counts"].get("CONTRADICTED") == 1
        assert stats["krippendorff_alpha"]["three_way"] == pytest.approx(1.0)

    def test_combine_disagreement(self, tmp_path):
        g_rows = [{"claim_id": "c1", "image_id": "i1", "rrg_model": "maira", "claim_text": "x",
                   "label": "SUPPORTED"}]
        r_rows = [{"claim_id": "c1", "image_id": "i1", "rrg_model": "maira", "claim_text": "x",
                   "label": "CONTRADICTED"}]
        v_rows = [{"claim_id": "c1", "image_id": "i1", "rrg_model": "maira", "claim_text": "x",
                   "label": "UNCERTAIN"}]
        g = tmp_path / "green.jsonl"
        r = tmp_path / "radfact.jsonl"
        v = tmp_path / "vert.jsonl"
        self._write_jsonl(g, g_rows)
        self._write_jsonl(r, r_rows)
        self._write_jsonl(v, v_rows)
        out = tmp_path / "ensemble.jsonl"
        stats = combine_labels(g, r, v, out)
        assert stats["disagreement_count"] == 1
