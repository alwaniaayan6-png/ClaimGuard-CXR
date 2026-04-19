"""Validate silver labels against PadChest-GR radiologist-placed bboxes.

PadChest-GR ships per-sentence radiologist bounding boxes with polarity
(is_positive). For a subset of v6 evaluation where the image comes from
PadChest-GR, we can validate the silver-label pipeline against this
radiologist ground truth without recruiting any clinician:

* Each PadChest-GR sentence with ``is_positive=true`` -> expected SUPPORTED
* Each PadChest-GR sentence with ``is_positive=false`` -> expected CONTRADICTED

We match silver-ensemble claims to PadChest-GR sentences by image_id and
semantic string similarity (normalized token overlap), then compute Cohen's
kappa, precision, recall, and F1 overall and per finding class.

This is the reviewer-defensibility artifact: the paper reports silver-label
agreement with the only public CXR dataset that carries radiologist
claim-level ground truth (not derived from MIMIC).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class ValidationRow:
    image_id: str
    finding: str
    sentence_text: str
    is_positive_gt: bool
    expected_label: str             # SUPPORTED | CONTRADICTED
    predicted_label: str            # SUPPORTED | CONTRADICTED | UNCERTAIN
    predicted_confidence: str       # HIGH | MED | LOW | EXCLUDED
    matched_claim_id: str | None
    similarity_score: float


_STOPWORDS = {"the", "a", "an", "is", "are", "in", "on", "of", "at", "to", "and", "or", "with",
              "no", "not", "for", "by", "be", "has", "have", "was", "were", "there"}


def _tokenize(text: str) -> set[str]:
    lower = text.lower()
    words = re.findall(r"[a-z]+", lower)
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_padchest_gr_sentences(records_jsonl: Path) -> list[dict]:
    """Load flat (image_id, sentence_en, finding, is_positive) rows from records JSONL."""
    rows: list[dict] = []
    with open(records_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = rec.get("study_id") or rec.get("image_id")
            if not image_id:
                continue
            for box in rec.get("boxes", []):
                rows.append({
                    "image_id": str(image_id),
                    "finding": str(box.get("finding", "")),
                    "sentence_en": str(box.get("sentence_en") or box.get("sentence_es") or ""),
                    "is_positive": bool(box.get("is_positive", True)),
                })
    return rows


def _load_ensemble_claims(ensemble_jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    with open(ensemble_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate(
    padchest_gr_records_jsonl: Path,
    silver_ensemble_jsonl: Path,
    out_jsonl: Path,
    *,
    min_similarity: float = 0.30,
) -> dict:
    """Match silver labels to radiologist sentences and compute agreement.

    Args:
        padchest_gr_records_jsonl: records JSONL from ``padchest_gr.records_to_jsonl``.
        silver_ensemble_jsonl: ensemble output from ``silver_ensemble.combine_labels``.
        out_jsonl: per-sentence validation detail rows.
        min_similarity: normalized-token-Jaccard threshold for claim-sentence matching.

    Returns:
        Dict with per-class and aggregate statistics (Cohen kappa, precision, recall, F1).
    """
    gt_rows = _load_padchest_gr_sentences(padchest_gr_records_jsonl)
    claim_rows = _load_ensemble_claims(silver_ensemble_jsonl)
    claims_by_image: dict[str, list[dict]] = defaultdict(list)
    for c in claim_rows:
        claims_by_image[str(c.get("image_id", ""))].append(c)

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    per_finding_counts: dict[str, Counter] = defaultdict(Counter)
    confusion = Counter()
    validated: list[ValidationRow] = []
    n_unmatched = 0

    with open(out_jsonl, "w") as fh:
        for gt in gt_rows:
            image_id = gt["image_id"]
            candidates = claims_by_image.get(image_id, [])
            best_sim = 0.0
            best_claim = None
            for claim in candidates:
                sim = _overlap(gt["sentence_en"], claim.get("claim_text", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_claim = claim
            expected = "SUPPORTED" if gt["is_positive"] else "CONTRADICTED"
            if best_claim is None or best_sim < min_similarity:
                n_unmatched += 1
                predicted = "UNMATCHED"
                confidence = "N/A"
                matched_id = None
            else:
                predicted = str(best_claim.get("final_label", "UNCERTAIN"))
                confidence = str(best_claim.get("confidence", "UNKNOWN"))
                matched_id = str(best_claim.get("claim_id", ""))
            row = ValidationRow(
                image_id=image_id,
                finding=gt["finding"],
                sentence_text=gt["sentence_en"],
                is_positive_gt=gt["is_positive"],
                expected_label=expected,
                predicted_label=predicted,
                predicted_confidence=confidence,
                matched_claim_id=matched_id,
                similarity_score=best_sim,
            )
            fh.write(json.dumps(row.__dict__) + "\n")
            validated.append(row)
            if predicted != "UNMATCHED":
                confusion[(expected, predicted)] += 1
                per_finding_counts[gt["finding"]][(expected, predicted)] += 1

    overall = _compute_agreement_stats(confusion)
    per_finding = {f: _compute_agreement_stats(c) for f, c in per_finding_counts.items()}
    return {
        "n_ground_truth_sentences": len(gt_rows),
        "n_matched_at_min_similarity": len(validated) - n_unmatched,
        "n_unmatched": n_unmatched,
        "min_similarity": min_similarity,
        "overall": overall,
        "per_finding": per_finding,
    }


def _compute_agreement_stats(confusion: Counter) -> dict:
    """Return kappa, precision, recall, F1 over SUPPORTED/CONTRADICTED labels.

    UNCERTAIN predictions are excluded from the binary classification stats
    but counted in ``n_uncertain`` for completeness.
    """
    cats = ("SUPPORTED", "CONTRADICTED")
    tp = confusion.get(("SUPPORTED", "SUPPORTED"), 0)
    fn = confusion.get(("SUPPORTED", "CONTRADICTED"), 0)
    fp = confusion.get(("CONTRADICTED", "SUPPORTED"), 0)
    tn = confusion.get(("CONTRADICTED", "CONTRADICTED"), 0)
    n_uncertain = sum(v for (e, p), v in confusion.items() if p == "UNCERTAIN")
    binary_total = tp + fn + fp + tn
    if binary_total == 0:
        return {"kappa": None, "precision": None, "recall": None, "f1": None,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_uncertain": n_uncertain}
    po = (tp + tn) / binary_total
    p_pos_exp = (tp + fn) / binary_total
    p_pos_pred = (tp + fp) / binary_total
    pe = p_pos_exp * p_pos_pred + (1 - p_pos_exp) * (1 - p_pos_pred)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
    return {
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "observed_agreement": po,
        "expected_agreement": pe,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_uncertain": n_uncertain,
    }
