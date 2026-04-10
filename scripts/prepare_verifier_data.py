"""Prepare training data for the ClaimGuard-CXR claim verifier.

Generates (claim, evidence, label) triples with balanced classes:
- Supported: real claims matched to correct evidence
- Contradicted: hard negatives from 8 types
- Insufficient Evidence: claims about absent findings

Usage:
    python scripts/prepare_verifier_data.py \
        --data-root /path/to/chexpert-plus \
        --radgraph-json /path/to/radgraph-xl.json \
        --splits-dir /path/to/splits \
        --output /path/to/verifier_training_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_training_reports(data_root: str, splits_dir: str) -> pd.DataFrame:
    """Load reports for training split patients only."""
    from verifact.data.preprocessing.patient_splits import (
        load_chexpert_plus_metadata,
        load_splits,
    )

    meta = load_chexpert_plus_metadata(data_root)
    splits = load_splits(splits_dir)

    train_patients = set(str(pid) for pid in splits["train"])
    meta["patient_id"] = meta["patient_id"].astype(str)
    train_meta = meta[meta["patient_id"].isin(train_patients)]

    logger.info(f"Loaded {len(train_meta)} training reports from {len(train_patients)} patients")
    return train_meta


def build_evidence_pool(reports: pd.DataFrame) -> list[str]:
    """Build a pool of evidence passages from training reports.

    Each report is split into sentences to serve as evidence passages
    for the retriever simulation during verifier training.
    """
    from verifact.models.decomposer.claim_decomposer import sentence_split

    passages = []
    for _, row in reports.iterrows():
        report_col = None
        for col in ["report", "text", "impression", "findings", "Report"]:
            if col in row.index and pd.notna(row[col]):
                report_col = col
                break
        if report_col is None:
            continue

        sentences = sentence_split(str(row[report_col]))
        passages.extend(sentences)

    logger.info(f"Built evidence pool with {len(passages)} passages")
    return passages


def generate_training_examples(
    reports: pd.DataFrame,
    radgraph_path: str | None,
    evidence_pool: list[str],
    n_examples_per_class: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Generate balanced training data for the verifier.

    Creates equal numbers of Supported (0), Contradicted (1), and
    Insufficient Evidence (2) examples.
    """
    from verifact.data.preprocessing.radgraph_parser import (
        load_radgraph,
        parse_all_reports,
    )
    from verifact.models.decomposer.claim_decomposer import (
        SentenceDecomposer,
        classify_claim_pathology,
    )

    rng = random.Random(seed)
    decomposer = SentenceDecomposer()

    examples = []

    # =============================================
    # Class 0: Supported (real claims + matching evidence)
    # =============================================
    logger.info("Generating Supported examples...")
    supported = []

    report_col = None
    for col in ["report", "text", "impression", "findings", "Report"]:
        if col in reports.columns:
            report_col = col
            break

    if report_col:
        for _, row in tqdm(reports.iterrows(), total=min(len(reports), n_examples_per_class * 3),
                           desc="Supported"):
            if len(supported) >= n_examples_per_class:
                break

            report_text = str(row[report_col]) if pd.notna(row[report_col]) else ""
            if not report_text.strip():
                continue

            claims = decomposer.decompose(report_text)
            for claim in claims:
                if len(supported) >= n_examples_per_class:
                    break
                # Evidence: 2 random passages from the evidence pool that are
                # somewhat relevant (share words with the claim)
                claim_words = set(claim.text.lower().split())
                relevant = [p for p in rng.sample(evidence_pool, min(100, len(evidence_pool)))
                           if claim_words & set(p.lower().split())]
                evidence = relevant[:2] if relevant else rng.sample(evidence_pool, min(2, len(evidence_pool)))

                supported.append({
                    "claim": claim.text,
                    "evidence": evidence,
                    "label": 0,
                    "pathology": claim.pathology_category,
                    "negative_type": "none",
                })

    logger.info(f"  Generated {len(supported)} Supported examples")

    # =============================================
    # Class 1: Contradicted (hard negatives)
    # =============================================
    logger.info("Generating Contradicted examples (hard negatives)...")
    contradicted = []

    if radgraph_path and Path(radgraph_path).exists():
        from verifact.data.preprocessing.radgraph_parser import (
            load_radgraph,
            extract_entities,
            extract_relations,
            entities_to_claims,
        )
        from verifact.data.augmentation.hard_negative_generator import generate_hard_negatives

        radgraph_data = load_radgraph(radgraph_path)
        all_radgraph_claims = []

        for report_id, annotation in radgraph_data.items():
            entities = extract_entities(annotation)
            relations = extract_relations(annotation)
            claims = entities_to_claims(entities, relations)
            all_radgraph_claims.extend(claims)

        logger.info(f"  Extracted {len(all_radgraph_claims)} claims from RadGraph")

        # Generate hard negatives from RadGraph claims
        neg_types = [
            "laterality_swap", "finding_substitution", "negation",
            "region_swap", "severity_swap", "temporal_error", "device_error",
        ]
        hard_negs = generate_hard_negatives(
            all_radgraph_claims, types=neg_types, n_per_claim=3, seed=seed,
        )

        for neg_claim, neg_type, neg_label in hard_negs:
            if len(contradicted) >= n_examples_per_class:
                break
            evidence = rng.sample(evidence_pool, min(2, len(evidence_pool)))
            contradicted.append({
                "claim": neg_claim.text,
                "evidence": evidence,
                "label": 1,
                "pathology": neg_claim.pathology_category,
                "negative_type": neg_type,
            })
    else:
        # Fallback: generate contradicted examples from sentence-level negation
        logger.info("  No RadGraph data — using simple negation for contradicted examples")
        for ex in supported[:n_examples_per_class]:
            claim = ex["claim"]
            if "no " in claim.lower():
                negated = claim.lower().replace("no ", "", 1).capitalize()
            else:
                negated = "No " + claim[0].lower() + claim[1:]
            contradicted.append({
                "claim": negated,
                "evidence": ex["evidence"],
                "label": 1,
                "pathology": ex["pathology"],
                "negative_type": "simple_negation",
            })
            if len(contradicted) >= n_examples_per_class:
                break

    logger.info(f"  Generated {len(contradicted)} Contradicted examples")

    # =============================================
    # Class 2: Insufficient Evidence
    # =============================================
    logger.info("Generating Insufficient Evidence examples...")
    insufficient = []

    # Claims about findings NOT mentioned in a given report
    all_pathologies = [
        "Cardiomegaly", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Fracture",
    ]
    templates = [
        "There is {finding}.",
        "{finding} is present.",
        "The imaging shows {finding}.",
        "{finding} is identified in the {side} lung.",
    ]

    for _ in range(n_examples_per_class):
        finding = rng.choice(all_pathologies)
        template = rng.choice(templates)
        side = rng.choice(["left", "right"])
        claim = template.format(finding=finding.lower(), side=side)
        evidence = rng.sample(evidence_pool, min(2, len(evidence_pool)))

        insufficient.append({
            "claim": claim,
            "evidence": evidence,
            "label": 2,
            "pathology": finding,
            "negative_type": "insufficient",
        })

    logger.info(f"  Generated {len(insufficient)} Insufficient Evidence examples")

    # Combine and shuffle
    all_examples = supported + contradicted + insufficient
    rng.shuffle(all_examples)

    logger.info(
        f"\nTotal training examples: {len(all_examples)}\n"
        f"  Supported: {len(supported)}\n"
        f"  Contradicted: {len(contradicted)}\n"
        f"  Insufficient: {len(insufficient)}"
    )

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Prepare verifier training data")
    parser.add_argument("--data-root", type=str, required=True, help="Path to CheXpert Plus")
    parser.add_argument("--radgraph-json", type=str, default=None, help="Path to RadGraph-XL JSON")
    parser.add_argument("--splits-dir", type=str, required=True, help="Path to split files")
    parser.add_argument("--output", type=str, default="verifier_training_data.json")
    parser.add_argument("--n-per-class", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    reports = load_training_reports(args.data_root, args.splits_dir)
    evidence_pool = build_evidence_pool(reports)

    # Generate examples
    examples = generate_training_examples(
        reports=reports,
        radgraph_path=args.radgraph_json,
        evidence_pool=evidence_pool,
        n_examples_per_class=args.n_per_class,
        seed=args.seed,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Saved {len(examples)} training examples to {output_path}")


if __name__ == "__main__":
    main()
