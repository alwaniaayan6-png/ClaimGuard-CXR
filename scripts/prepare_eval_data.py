"""Prepare evaluation claim sets from CheXpert Plus test+calibration splits.

Creates controlled claim sets with labels {Supported=0, Contradicted=1,
Insufficient=2} and eight clinically-motivated hard-negative types.

Design contract:
  - Patient-level separation: no patient in more than one split
  - Class balance: 1/3 per label (33/33/33 target)
  - Hard-neg taxonomy (8 types per CLAIMGUARD_PROPOSAL.md §Methods):
      1. negation                2. laterality_swap
      3. severity_swap           4. temporal_error
      5. finding_substitution    6. region_swap
      7. device_line_error       8. omission_as_support
  - Evidence validation: contradicted evidence must NOT already support
    the perturbed claim (avoids label noise — Bug C6)
  - Insufficient = evidence from DIFFERENT patient AND DIFFERENT pathology
    (enforces true evidence mismatch — Bug C5)

This isolates verification quality from generator quality — standard
approach in NLI/hallucination detection papers (FactCC, SummaC, etc.)

Usage:
    python3 scripts/prepare_eval_data.py \
        --data-path /path/to/df_chexpert_plus_240401.csv \
        --splits-dir /path/to/splits_chexpert_plus/ \
        --output-dir /path/to/eval_data/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import clinical knowledge from sibling package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.augmentation.clinical_knowledge import (  # noqa: E402
    should_swap_laterality,
    get_confusable_finding,
    CHEXPERT_LABEL_NOISE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# CheXpert pathology keywords for claim classification
# =========================================================
PATHOLOGY_KEYWORDS = {
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart",
                     "heart size is enlarged", "heart is enlarged"],
    "Consolidation": ["consolidation", "consolidated"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Pneumonia": ["pneumonia", "infectious process"],
    "Pneumothorax": ["pneumothorax"],
    "Lung Opacity": ["opacity", "opacification", "opacities"],
    "Lung Lesion": ["lesion", "mass", "nodule"],
    "Fracture": ["fracture", "fractured"],
    "Support Devices": ["tube", "catheter", "line", "pacemaker", "picc",
                        "endotracheal", "nasogastric", "port-a-cath",
                        "tracheostomy"],
    "Enlarged Cardiomediastinum": ["mediastinal", "mediastinum"],
    "No Finding": ["normal", "unremarkable", "no acute", "clear"],
}

# =========================================================
# Hard negative perturbation vocabularies
# =========================================================
LATERALITY_TERMS = {"left": "right", "right": "left", "bilateral": "unilateral",
                    "unilateral": "bilateral"}
SEVERITY_TERMS = {"small": "large", "large": "small", "mild": "severe",
                  "severe": "mild", "moderate": "marked", "minimal": "extensive",
                  "subtle": "prominent", "tiny": "massive"}
TEMPORAL_TERMS = {"new": "unchanged", "unchanged": "worsening",
                  "worsening": "improved", "improved": "new",
                  "stable": "progressing", "resolving": "worsening",
                  "acute": "chronic", "chronic": "acute"}

# Region-swap vocabulary — anatomical location pairs for same-finding,
# wrong-region errors. Excludes laterality (covered separately).
REGION_TERMS = {
    "upper": "lower", "lower": "upper",
    "apical": "basal", "basal": "apical",
    "anterior": "posterior", "posterior": "anterior",
    "retrocardiac": "perihilar", "perihilar": "retrocardiac",
    "mediastinal": "paratracheal", "paratracheal": "mediastinal",
    "lobe": "base",  # weak swap, used only as last resort
}

# Device/line position-error vocabulary. Only applied to claims mentioning
# devices (ETT, NG, PICC, etc.). Swaps position phrases to clinically
# dangerous mis-placements.
DEVICE_POSITIONS = {
    "in appropriate position": "in the right mainstem",
    "in satisfactory position": "in the right mainstem",
    "in good position": "in the right mainstem bronchus",
    "terminates at the svc": "terminates in the right atrium",
    "terminates in the svc": "terminates in the right ventricle",
    "above the carina": "below the carina",
    "tip at the carina": "tip in the right mainstem",
    "tip projects over the svc": "tip projects over the right ventricle",
    "correctly positioned": "malpositioned",
    "properly positioned": "terminates in the right atrium",
}

# Keywords that mark a claim as device-related
DEVICE_KEYWORDS = {
    "ett", "endotracheal tube", "endotracheal",
    "ng tube", "nasogastric tube", "nasogastric",
    "ogt", "orogastric", "dobhoff",
    "picc", "picc line", "central line", "central venous catheter",
    "cvc", "port", "port-a-cath",
    "tracheostomy tube", "tracheostomy", "trach",
    "pacemaker", "icd", "pacer lead", "pacing wire",
    "chest tube", "pigtail",
}

# Fabrication vocabulary — findings to hallucinate for omission_as_support.
# Used to create a claim about something the report does NOT mention.
FABRICATION_FINDINGS = [
    ("pneumothorax", "left", "small"),
    ("pneumothorax", "right", "moderate"),
    ("pleural effusion", "left", "moderate"),
    ("pleural effusion", "right", "small"),
    ("consolidation", "right lower lobe", "dense"),
    ("consolidation", "left lower lobe", "patchy"),
    ("atelectasis", "bilateral lower lobe", "mild"),
    ("pulmonary edema", "bilateral", "moderate"),
    ("cardiomegaly", "", "moderate"),
    ("pulmonary nodule", "right upper lobe", "3 mm"),
    ("lung mass", "left upper lobe", "2 cm"),
    ("pneumonia", "right middle lobe", "new"),
    ("rib fracture", "left lateral 6th", "acute"),
    ("pneumomediastinum", "", "small"),
]


def classify_pathology(claim: str) -> str:
    """Map a claim to its CheXpert pathology category (word-boundary match)."""
    claim_lower = claim.lower()
    for pathology, keywords in PATHOLOGY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', claim_lower):
                return pathology
    return "Other"


def extract_claims(report: str) -> list[str]:
    """Extract individual atomic claims from a report section."""
    if not report or report == "nan" or len(report) < 10:
        return []

    sentences = re.split(r'(?<=[.!?])\s+|\n+', report.strip())
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= 15 and not sent.startswith("HISTORY"):
            claims.append(sent)
    return claims


def _word_boundary_sub(pattern: str, repl: str, text: str, count: int = 1) -> str:
    """Substitute `pattern` as a whole word, case-insensitive."""
    return re.sub(
        rf'\b{re.escape(pattern)}\b', repl, text,
        flags=re.IGNORECASE, count=count,
    )


def _claim_mentions_device(claim: str) -> bool:
    """Check if claim mentions a support device."""
    claim_lower = claim.lower()
    return any(re.search(rf'\b{re.escape(kw)}\b', claim_lower)
               for kw in DEVICE_KEYWORDS)


def generate_hard_negative(
    claim: str, neg_type: str, rng: random.Random | None = None,
) -> str | None:
    """Generate a single hard negative from a claim.

    Returns None if the perturbation cannot be meaningfully applied
    (e.g. laterality_swap on a midline finding). Callers should retry
    with a different type or claim.
    """
    if rng is None:
        rng = random.Random()
    claim_lower = claim.lower()

    if neg_type == "negation":
        # Flip present/absent markers where present
        if re.search(r'\bno\b', claim_lower):
            # Remove "no " (with trailing space) for grammatical output
            return re.sub(r'\bno\s+', '', claim, count=1, flags=re.IGNORECASE).strip()
        elif re.search(r'\bwithout\b', claim_lower):
            return _word_boundary_sub("without", "with", claim)
        elif re.search(r'\babsent\b', claim_lower):
            return _word_boundary_sub("absent", "present", claim)
        elif re.search(r'\bpresent\b', claim_lower):
            return _word_boundary_sub("present", "absent", claim)
        else:
            # Insert "no" after a copula verb (is/are/was/were) for grammaticality.
            # Fall back to inserting before the last noun-like token.
            for copula in ["is", "are", "was", "were"]:
                m = re.search(rf'\b{copula}\b', claim, flags=re.IGNORECASE)
                if m:
                    return claim[:m.end()] + " no" + claim[m.end():]
            # No copula — prepend "No " and lowercase the original first letter
            if claim and claim[0].isalpha():
                return "No " + claim[0].lower() + claim[1:]
            return "No " + claim

    elif neg_type == "laterality_swap":
        # Only apply to laterality-sensitive findings (bug M8 fix)
        if not should_swap_laterality(claim):
            return None
        for orig, swap in LATERALITY_TERMS.items():
            if re.search(rf'\b{re.escape(orig)}\b', claim_lower):
                return _word_boundary_sub(orig, swap, claim)
        return None

    elif neg_type == "severity_swap":
        for orig, swap in SEVERITY_TERMS.items():
            if re.search(rf'\b{re.escape(orig)}\b', claim_lower):
                return _word_boundary_sub(orig, swap, claim)
        return None

    elif neg_type == "temporal_error":
        for orig, swap in TEMPORAL_TERMS.items():
            if re.search(rf'\b{re.escape(orig)}\b', claim_lower):
                return _word_boundary_sub(orig, swap, claim)
        return None

    elif neg_type == "finding_substitution":
        # Prefer clinically confusable pairs (via clinical_knowledge)
        confusable = get_confusable_finding(claim)
        if confusable:
            # Find which term from the pair is in the claim and swap
            for term in [
                "consolidation", "atelectasis", "pneumonia", "pulmonary edema",
                "pleural effusion", "pneumothorax", "cardiomegaly", "lung mass",
                "hilar enlargement", "mediastinal widening", "rib fracture",
                "pulmonary nodule", "skin fold",
            ]:
                if re.search(rf'\b{re.escape(term)}\b', claim_lower):
                    return _word_boundary_sub(term, confusable, claim)
        # Fallback: dictionary-based substitution
        fallback = {
            "atelectasis": "consolidation",
            "consolidation": "atelectasis",
            "pneumonia": "pulmonary edema",
            "pulmonary edema": "pneumonia",
            "pleural effusion": "atelectasis",
            "pneumothorax": "skin fold artifact",
            "cardiomegaly": "pericardial effusion",
            "lung mass": "consolidation",
        }
        for orig, swap in fallback.items():
            if re.search(rf'\b{re.escape(orig)}\b', claim_lower):
                return _word_boundary_sub(orig, swap, claim)
        return None

    elif neg_type == "region_swap":
        # Swap anatomical region words (NOT laterality — that's its own type)
        for orig, swap in REGION_TERMS.items():
            if re.search(rf'\b{re.escape(orig)}\b', claim_lower):
                return _word_boundary_sub(orig, swap, claim)
        return None

    elif neg_type == "device_line_error":
        # Only apply to device-related claims
        if not _claim_mentions_device(claim):
            return None
        # Position phrases are multi-word; match them as-is but anchored with
        # word-boundaries on the outside so we don't match partial words.
        for orig, swap in DEVICE_POSITIONS.items():
            pattern = rf'\b{re.escape(orig)}\b'
            if re.search(pattern, claim_lower):
                return re.sub(pattern, swap, claim, flags=re.IGNORECASE, count=1)
        # No matching position phrase; synthesize one
        return claim.rstrip(".") + " terminates in the right mainstem bronchus."

    elif neg_type == "omission_as_support":
        # Fabricate a new claim about a finding NOT in the original
        finding, laterality, severity = rng.choice(FABRICATION_FINDINGS)
        parts = []
        if severity:
            parts.append(severity)
        if laterality:
            parts.append(laterality)
        parts.append(finding)
        return " ".join(parts).capitalize() + "."

    return None


def _evidence_supports_claim_text(claim: str, evidence_texts: list[str]) -> bool:
    """Heuristic check: does any evidence passage appear to support the claim?

    Used to filter contradicted hard-negatives where the perturbed claim
    is accidentally still supported by the original evidence (Bug C6).

    Strategy: look for the key finding/anatomy/severity terms of the
    perturbed claim in the evidence. If the claim's key terms are
    present in evidence verbatim, treat the pairing as AMBIGUOUS and
    reject it.
    """
    if not evidence_texts:
        return False
    evidence_joined = " ".join(str(e) for e in evidence_texts).lower()
    claim_lower = claim.lower()

    # Extract key clinical terms from claim (pathology keywords + laterality)
    key_terms = []
    for pathology_kws in PATHOLOGY_KEYWORDS.values():
        for kw in pathology_kws:
            if re.search(rf'\b{re.escape(kw)}\b', claim_lower):
                key_terms.append(kw)
    for lat in ["left", "right", "bilateral"]:
        if re.search(rf'\b{lat}\b', claim_lower):
            key_terms.append(lat)

    if not key_terms:
        return False

    # If ALL key terms appear in evidence, the evidence likely still
    # supports the perturbed claim — reject the pairing.
    matches = sum(
        1 for kw in key_terms
        if re.search(rf'\b{re.escape(kw)}\b', evidence_joined)
    )
    return matches == len(key_terms) and matches >= 2


def create_eval_claims(
    df: pd.DataFrame,
    n_supported: int = 5000,
    n_contradicted: int = 5000,
    n_insufficient: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Create balanced evaluation claim sets with all 8 hard-negative types.

    Args:
        df: CheXpert Plus DataFrame (filtered to desired split).
        n_supported: Number of supported claims (label=0).
        n_contradicted: Number of contradicted claims (label=1).
        n_insufficient: Number of insufficient-evidence claims (label=2).
        seed: Random seed.

    Returns:
        List of claim dicts with keys: claim, evidence, label,
        pathology, negative_type, patient_id.
    """
    rng = random.Random(seed)

    # Extract all claims with their evidence
    all_claims = []
    for _, row in df.iterrows():
        pid = str(row.get("deid_patient_id", ""))
        impression = str(row.get("section_impression", ""))
        findings = str(row.get("section_findings", ""))

        report_claims = extract_claims(impression)
        if not report_claims:
            report_claims = extract_claims(findings)

        for claim in report_claims:
            # Evidence = other sentences from same report
            evidence_pool = [c for c in report_claims if c != claim]
            if findings and findings != "nan":
                evidence_pool.extend(extract_claims(findings))

            all_claims.append({
                "claim": claim,
                "evidence": evidence_pool[:3],
                "pathology": classify_pathology(claim),
                "patient_id": pid,
            })

    logger.info(f"Extracted {len(all_claims)} total claims from {len(df)} reports")

    # =========================================================
    # 1. SUPPORTED claims (real claims with same-report evidence)
    # =========================================================
    supported = rng.sample(all_claims, min(n_supported, len(all_claims)))
    supported_examples = []
    for item in supported:
        supported_examples.append({
            "claim": item["claim"],
            "evidence": item["evidence"][:2],
            "label": 0,  # Supported
            "pathology": item["pathology"],
            "negative_type": "none",
            "patient_id": item["patient_id"],
        })

    # =========================================================
    # 2. CONTRADICTED claims (8 hard-negative types)
    # =========================================================
    # All 8 types; sampled uniformly. The per-type distribution is
    # determined by claim eligibility (not every claim supports every
    # perturbation type) — we accept whatever the data produces.
    neg_types = [
        "negation",
        "laterality_swap",
        "severity_swap",
        "temporal_error",
        "finding_substitution",
        "region_swap",
        "device_line_error",
        "omission_as_support",
    ]
    contradicted_examples = []
    neg_pool = list(all_claims)
    rng.shuffle(neg_pool)

    # Track per-type counts for balanced sampling
    type_quota = n_contradicted // len(neg_types)
    type_counts = {t: 0 for t in neg_types}

    attempts = 0
    max_attempts = n_contradicted * 20
    rejected_for_support = 0
    while len(contradicted_examples) < n_contradicted and attempts < max_attempts:
        item = neg_pool[attempts % len(neg_pool)]
        # Prefer under-represented types
        underrep = [t for t in neg_types if type_counts[t] < type_quota]
        neg_type = rng.choice(underrep if underrep else neg_types)

        neg_claim = generate_hard_negative(item["claim"], neg_type, rng)
        attempts += 1

        if not neg_claim or neg_claim == item["claim"]:
            continue

        evidence = item["evidence"][:2]

        # C6 FIX: validate evidence actually contradicts the perturbed claim.
        # If the evidence still contains all the perturbed claim's key terms,
        # the pairing is ambiguous — reject it.
        if _evidence_supports_claim_text(neg_claim, evidence):
            rejected_for_support += 1
            continue

        contradicted_examples.append({
            "claim": neg_claim,
            "evidence": evidence,
            "label": 1,  # Contradicted
            "pathology": item["pathology"],
            "negative_type": neg_type,
            "patient_id": item["patient_id"],
        })
        type_counts[neg_type] += 1

    logger.info(
        f"Generated {len(contradicted_examples)} contradicted claims from "
        f"{attempts} attempts ({rejected_for_support} rejected for "
        f"evidence-still-supports-claim); per-type: {type_counts}"
    )

    # =========================================================
    # 3. INSUFFICIENT EVIDENCE claims (truly mismatched)
    # =========================================================
    # C5 FIX: evidence must come from a DIFFERENT patient AND a DIFFERENT
    # pathology than the claim, with a retry loop + hard cap.
    insufficient_examples = []
    claims_list = list(all_claims)
    rng.shuffle(claims_list)

    # Build patient/pathology index for fast mismatch lookup
    by_pathology = {}
    for idx, c in enumerate(claims_list):
        by_pathology.setdefault(c["pathology"], []).append(idx)

    for i in range(min(n_insufficient, len(claims_list))):
        item = claims_list[i]
        target_pid = item["patient_id"]
        target_path = item["pathology"]

        # Find a claim from a different pathology (so different evidence)
        other_paths = [p for p in by_pathology if p != target_path]
        if not other_paths:
            # Extremely degenerate case — fall back to any-different-patient
            other_paths = list(by_pathology.keys())

        other = None
        for _ in range(20):  # max 20 retries
            chosen_path = rng.choice(other_paths)
            candidate_idx = rng.choice(by_pathology[chosen_path])
            candidate = claims_list[candidate_idx]
            if (candidate["patient_id"] != target_pid
                    and candidate["pathology"] != target_path
                    and candidate["evidence"]):
                other = candidate
                break

        if other is None:
            # Give up on this claim — skip rather than create bad label
            continue

        insufficient_examples.append({
            "claim": item["claim"],
            "evidence": other["evidence"][:2],
            "label": 2,  # Insufficient Evidence
            "pathology": item["pathology"],
            "negative_type": "mismatched_evidence",
            "patient_id": item["patient_id"],
        })

    logger.info(f"Generated {len(insufficient_examples)} insufficient-evidence claims")

    # Combine and shuffle
    all_examples = supported_examples + contradicted_examples + insufficient_examples
    rng.shuffle(all_examples)

    # Log distribution
    label_counts: dict[int, int] = {}
    pathology_counts: dict[str, int] = {}
    neg_type_counts: dict[str, int] = {}
    for ex in all_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
        pathology_counts[ex["pathology"]] = pathology_counts.get(ex["pathology"], 0) + 1
        neg_type_counts[ex["negative_type"]] = neg_type_counts.get(ex["negative_type"], 0) + 1

    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Top pathologies: {dict(sorted(pathology_counts.items(), key=lambda x: -x[1])[:8])}")
    logger.info(f"Negative types: {neg_type_counts}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation claim sets")
    parser.add_argument("--data-path", required=True,
                        help="Path to df_chexpert_plus_240401.csv")
    parser.add_argument("--splits-dir", required=True,
                        help="Path to splits directory")
    parser.add_argument("--output-dir", default="./eval_data",
                        help="Output directory for eval data")
    parser.add_argument("--n-claims", type=int, default=5000,
                        help="Number of claims per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading CheXpert Plus data...")
    df = pd.read_csv(args.data_path, low_memory=False)
    df["deid_patient_id"] = df["deid_patient_id"].astype(str)

    # Process calibration and test splits
    for split_name in ["calibration", "test"]:
        split_file = Path(args.splits_dir) / f"{split_name}_patients.csv"
        if not split_file.exists():
            logger.warning(f"No {split_name} split file at {split_file}")
            continue

        split_pids = set(
            pd.read_csv(split_file)["patient_id"].astype(str).tolist()
        )
        split_df = df[df["deid_patient_id"].isin(split_pids)]
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {split_name}: {len(split_df)} reports, "
                     f"{len(split_pids)} patients")

        claims = create_eval_claims(
            split_df,
            n_supported=args.n_claims,
            n_contradicted=args.n_claims,
            n_insufficient=args.n_claims,
            seed=args.seed,
        )

        out_file = output_dir / f"{split_name}_claims.json"
        with open(out_file, "w") as f:
            json.dump(claims, f, indent=2)
        logger.info(f"Saved {len(claims)} claims to {out_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
