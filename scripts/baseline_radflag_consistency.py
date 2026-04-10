"""RadFlag-style self-consistency baseline for ClaimGuard-CXR v2.

Implements the core idea from RadFlag (ML4H 2025): sample N completions from
a generator at varying temperatures, then flag claims that appear inconsistently
across samples. Low cross-sample agreement = likely hallucination.

Since we don't have a trained generator, we simulate this by:
1. Taking each test claim
2. Generating N paraphrases via simple perturbation (synonym swap, reorder)
3. Checking if the claim's evidence supports all paraphrases consistently
4. Low consistency across paraphrases = flag as potentially hallucinated

This is a simplified version — real RadFlag uses multiple sampled reports.

Usage:
    python3 scripts/baseline_radflag_consistency.py \
        --test-claims /path/to/test_claims.json \
        --output results/radflag_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_claim_self_consistency(
    claim: str,
    evidence: list[str],
    n_variants: int = 5,
    seed: int = 42,
) -> float:
    """Compute self-consistency score for a claim.

    Generate N slight variants of the claim and check if they all have
    similar keyword overlap with evidence. High variance = inconsistent = suspicious.

    Returns:
        Consistency score in [0, 1]. Low = likely hallucination.
    """
    rng = random.Random(seed + hash(claim) % 10000)
    evidence_text = " ".join(evidence).lower()
    evidence_words = set(evidence_text.split())

    # Extract content words from claim
    claim_lower = claim.lower()
    claim_words = set(re.findall(r'\b[a-z]{3,}\b', claim_lower))
    # Remove stopwords
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                 "had", "her", "was", "one", "our", "out", "has", "with", "this",
                 "that", "from", "have", "been", "were", "they", "than", "each"}
    claim_words -= stopwords

    if not claim_words:
        return 0.5  # Can't assess

    # Base overlap
    base_overlap = len(claim_words & evidence_words) / max(1, len(claim_words))

    # Generate variants by dropping/replacing words
    variant_overlaps = []
    claim_word_list = list(claim_words)

    for _ in range(n_variants):
        # Drop 1-2 random content words
        n_drop = rng.randint(1, min(2, len(claim_word_list)))
        dropped = set(rng.sample(claim_word_list, n_drop))
        variant_words = claim_words - dropped
        if not variant_words:
            variant_words = claim_words

        overlap = len(variant_words & evidence_words) / max(1, len(variant_words))
        variant_overlaps.append(overlap)

    # Consistency = inverse of variance across variants
    if len(variant_overlaps) > 1:
        variance = float(np.var(variant_overlaps))
        consistency = 1.0 - min(1.0, variance * 10)  # Scale variance to [0, 1]
    else:
        consistency = base_overlap

    # Combine with base overlap
    return 0.5 * base_overlap + 0.5 * consistency


def run_radflag_baseline(
    test_claims_path: str,
    output_path: str,
    n_variants: int = 5,
    seed: int = 42,
) -> dict:
    """Run RadFlag-style self-consistency baseline."""
    with open(test_claims_path) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} test claims")

    scores = []
    labels = []

    for item in claims:
        claim = item["claim"]
        evidence = item.get("evidence", [])
        label = 1 if item["label"] == 1 else 0

        consistency = compute_claim_self_consistency(
            claim, evidence, n_variants=n_variants, seed=seed
        )
        scores.append(consistency)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    # Threshold: low consistency = contradicted
    threshold = np.median(scores)
    preds = (scores < threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    contra_recall = float((preds[labels == 1] == 1).mean()) if labels.sum() > 0 else 0.0

    try:
        auroc = float(roc_auc_score(labels, -scores))
    except ValueError:
        auroc = 0.5

    results = {
        "method": "RadFlag-style self-consistency",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "contra_recall": contra_recall,
        "auroc": auroc,
        "n_claims": len(claims),
        "n_variants": n_variants,
        "threshold": float(threshold),
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRadFlag consistency: acc={accuracy:.4f}, contra_recall={contra_recall:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-claims", required=True)
    parser.add_argument("--output", default="results/radflag_baseline.json")
    parser.add_argument("--n-variants", type=int, default=5)
    args = parser.parse_args()

    run_radflag_baseline(args.test_claims, args.output, args.n_variants)


if __name__ == "__main__":
    main()
