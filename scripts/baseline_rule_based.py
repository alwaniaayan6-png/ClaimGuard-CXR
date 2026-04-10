"""Rule-based negation detector baseline for ClaimGuard-CXR.

Predicts Contradicted if evidence contains negation words within a small
context window of claim-content-overlap keywords. Otherwise NotContradicted.

This is a deliberately simple lexical baseline to show the value of the
trained verifier. Runs locally in seconds — no GPU, no API calls.

Evaluates on the same binary test set as the trained verifier and reports
the same metrics (accuracy, F1, AUROC, AP, confusion matrix).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

# Negation cues (standard medical NLP list, adapted for radiology reports)
NEGATION_CUES = {
    "no", "not", "without", "absent", "denies", "negative", "free of",
    "none", "ruled out", "no evidence", "no longer", "resolved",
    "cannot", "never", "unremarkable", "clear", "normal",
}

# Strong contradiction signals (specific to our hard-neg taxonomy)
LATERALITY_WORDS = {"left", "right", "bilateral", "unilateral"}
SEVERITY_SWAP_PAIRS = [
    ("small", "large"), ("mild", "severe"), ("minimal", "extensive"),
    ("tiny", "massive"),
]
TEMPORAL_SWAPS = [
    ("unchanged", "progressing"), ("stable", "worsening"),
    ("resolved", "new"), ("improved", "worsened"),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def content_overlap(claim_tokens: set[str], evidence_tokens: list[str]) -> set[int]:
    """Return positions in evidence_tokens where a claim content word appears."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "with", "and", "or", "but", "there", "has",
        "have", "had", "be", "been", "being", "this", "that",
    }
    content = {t for t in claim_tokens if t not in stopwords and len(t) > 2}
    return {i for i, t in enumerate(evidence_tokens) if t in content}


def has_nearby_negation(evidence_tokens: list[str], positions: set[int],
                        window: int = 4) -> bool:
    """Check if any negation cue is within `window` tokens of any content position."""
    for pos in positions:
        lo = max(0, pos - window)
        hi = min(len(evidence_tokens), pos + window + 1)
        window_text = " ".join(evidence_tokens[lo:hi])
        if any(cue in window_text for cue in NEGATION_CUES):
            return True
    return False


def detect_laterality_conflict(claim: str, evidence: str) -> bool:
    """Claim says 'left', evidence says 'right' (or vice versa)."""
    c_tokens = set(tokenize(claim))
    e_tokens = set(tokenize(evidence))
    c_lat = c_tokens & LATERALITY_WORDS
    e_lat = e_tokens & LATERALITY_WORDS
    if not c_lat or not e_lat:
        return False
    # Conflict if no overlap
    return len(c_lat & e_lat) == 0


def detect_severity_swap(claim: str, evidence: str) -> bool:
    c_lo = claim.lower()
    e_lo = evidence.lower()
    for a, b in SEVERITY_SWAP_PAIRS:
        if (a in c_lo and b in e_lo) or (b in c_lo and a in e_lo):
            return True
    return False


def detect_temporal_swap(claim: str, evidence: str) -> bool:
    c_lo = claim.lower()
    e_lo = evidence.lower()
    for a, b in TEMPORAL_SWAPS:
        if (a in c_lo and b in e_lo) or (b in c_lo and a in e_lo):
            return True
    return False


def predict_single(claim: str, evidence: str) -> tuple[int, float]:
    """Return (prediction, confidence_of_contra) for one claim.

    prediction: 0 = NotContra, 1 = Contra
    confidence: 0.0-1.0 (sum of fired rules / 4)
    """
    if isinstance(evidence, list):
        evidence = " ".join(evidence)

    claim_tokens_list = tokenize(claim)
    claim_tokens = set(claim_tokens_list)
    evidence_tokens = tokenize(evidence)

    score = 0
    positions = content_overlap(claim_tokens, evidence_tokens)

    # Rule 1: negation near shared content
    if positions and has_nearby_negation(evidence_tokens, positions, window=4):
        # Only flag if the CLAIM itself doesn't contain a negation cue
        if not any(cue in claim.lower() for cue in NEGATION_CUES):
            score += 1

    # Rule 2: laterality conflict
    if detect_laterality_conflict(claim, evidence):
        score += 1

    # Rule 3: severity swap
    if detect_severity_swap(claim, evidence):
        score += 1

    # Rule 4: temporal swap
    if detect_temporal_swap(claim, evidence):
        score += 1

    pred = 1 if score >= 1 else 0
    conf_contra = score / 4.0
    return pred, conf_contra


def evaluate(test_path: str, name: str) -> dict:
    with open(test_path) as f:
        claims = json.load(f)

    print(f"\n=== {name} ===")
    print(f"Loaded {len(claims)} claims from {test_path}")

    preds = []
    contra_scores = []
    labels_binary = []
    for claim in claims:
        pred, conf = predict_single(claim["claim"], claim["evidence"])
        preds.append(pred)
        contra_scores.append(conf)
        # Binary label remap
        labels_binary.append(1 if claim["label"] == 1 else 0)

    preds = np.array(preds)
    contra_scores = np.array(contra_scores)
    labels = np.array(labels_binary)

    # Binary not-contra score = 1 - contra_score
    not_contra_score = 1.0 - contra_scores

    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None).tolist()
    prec = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec = recall_score(labels, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    # AUROC / AP (need continuous scores)
    try:
        auroc = roc_auc_score((labels == 0).astype(int), not_contra_score)
        ap = average_precision_score((labels == 0).astype(int), not_contra_score)
    except ValueError:
        auroc, ap = float("nan"), float("nan")

    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Macro F1:            {mf1:.4f}")
    print(f"  NotContra F1:        {per_class_f1[0]:.4f}")
    print(f"  Contradicted F1:     {per_class_f1[1]:.4f}")
    print(f"  Contra Precision:    {prec[1]:.4f}")
    print(f"  Contra Recall:       {rec[1]:.4f}")
    print(f"  AUROC (NotContra):   {auroc:.4f}")
    print(f"  AP (NotContra):      {ap:.4f}")
    print(f"  Confusion matrix:    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"                        [FN={cm[1][0]}, TP={cm[1][1]}]]")

    return {
        "name": name,
        "n_claims": len(claims),
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "f1_notcontra": float(per_class_f1[0]),
        "f1_contra": float(per_class_f1[1]),
        "precision_contra": float(prec[1]),
        "recall_contra": float(rec[1]),
        "auroc": float(auroc),
        "average_precision": float(ap),
        "confusion_matrix": cm,
    }


def main():
    # Need to fetch test data from Modal — for now, use local cached copies
    test_paths = [
        ("/tmp/oracle_res/test_claims.json", "Rule-based (oracle evidence)"),
    ]

    # Download test_claims.json from Modal volume if not present
    import os, subprocess
    if not os.path.exists("/tmp/oracle_res/test_claims.json"):
        os.makedirs("/tmp/oracle_res", exist_ok=True)
        subprocess.run([
            "python3", "-m", "modal", "volume", "get",
            "claimguard-data", "/eval_data/test_claims.json",
            "/tmp/oracle_res/test_claims.json", "--force",
        ], check=True)

    results = []
    for path, name in test_paths:
        results.append(evaluate(path, name))

    # Also retrieval if available
    retr_path = "/tmp/oracle_res/test_claims_retrieved.json"
    if not os.path.exists(retr_path):
        subprocess.run([
            "python3", "-m", "modal", "volume", "get",
            "claimguard-data", "/eval_data/test_claims_retrieved.json",
            retr_path, "--force",
        ], check=False)
    if os.path.exists(retr_path):
        results.append(evaluate(retr_path, "Rule-based (retrieved evidence)"))

    # Save combined results
    out_path = Path("/Users/aayanalwani/VeriFact/verifact/figures/baseline_rule_based.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
