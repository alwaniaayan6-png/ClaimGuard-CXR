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
from data.augmentation.hard_negative_generator import (  # noqa: E402
    MEASUREMENT_TARGET_NOUNS as _MEASUREMENT_TARGET_NOUNS_SHARED,
)
from inference.provenance import (  # noqa: E402
    EvidenceSourceType,
    default_provenance,
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

# =========================================================
# v3 taxonomy — fabrication vocabularies
# =========================================================
# Injected into claims that had no measurement / prior / temporal
# language in the original text. Attacks the ReXErr 2024 + Chen et al.
# 2025 failure modes not covered by the 8-type v2 taxonomy.
FABRICATED_MEASUREMENT_UNITS = [
    "3 mm", "5 mm", "7 mm", "9 mm",
    "1.2 cm", "1.5 cm", "1.8 cm", "2.4 cm", "3 cm",
    "4 x 6 mm", "5 x 8 mm", "1 x 2 cm",
]
FABRICATED_PRIOR_PHRASES = [
    "compared to the prior exam from 2 weeks ago",
    "since the previous study",
    "unchanged from the prior film",
    "stable compared to the comparison exam",
    "new relative to the most recent comparison",
    "interval change from the baseline study",
    "compared with the prior radiograph from last month",
]
FABRICATED_TEMPORAL_DATES = [
    "since 3 days ago",
    "from last week's study",
    "compared to yesterday's film",
    "noted on the prior exam from 2 days ago",
    "present since the study on 01/12/2026",
    "new since the admission radiograph",
    "interval change over the past 48 hours",
]

# Regex guards — skip fabrication when the claim already has similar text.
# These mirror the v3.1 broadened patterns in
# ``data/augmentation/hard_negative_generator.py`` (self-check HIGH #1, #2,
# #5).  Kept in lockstep so the Claim-based and string-based fabrication
# paths make the same accept/reject decisions.
_EXISTING_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?"                                 # leading size
    r"(?:\s*[x×-]\s*\d+(?:\.\d+)?)?"                   # optional x/× dim-2
    r"\s*[- ]?"                                         # optional separator
    r"(?:mm|cm|millimeter|centimeter)s?\b",             # unit
    re.IGNORECASE,
)
_EXISTING_PRIOR_RE = re.compile(
    r"(?:"
    # Connective + prior/previous/... noun phrase (original behaviour)
    r"\b(?:compared (?:to|with)|since|from|relative to"
    r"|interval (?:change|worsening|improvement|development|progression|increase|decrease))"
    r"\s+(?:the\s+)?(?:prior|previous|last|baseline|comparison|most recent)\b"
    r"|"
    # Bare prior/previous/baseline exam/study/film phrase — catches
    # outputs of fabricate_temporal like "noted on the prior exam"
    r"\b(?:prior|previous|baseline|comparison)\s+"
    r"(?:exam|study|film|radiograph|imaging|report|chest x-?ray|cxr)\b"
    r"|"
    # Standalone radiology prior-reference idioms
    r"\b(?:previously\s+(?:noted|seen|described|demonstrated)"
    r"|in\s+the\s+interim"
    r"|grossly\s+similar"
    r"|no\s+(?:significant\s+)?interval\s+change"
    r"|(?:stable|unchanged)\s+(?:from|compared|since|in\s+the)"
    r")\b"
    r")",
    re.IGNORECASE,
)
_EXISTING_TEMPORAL_RE = re.compile(
    r"\b(?:\d+\s+(?:day|week|month|year)s?\s+ago"
    r"|yesterday|last\s+(?:week|month|year)"
    r"|since\s+\d|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:noted|present|new)\s+(?:on|since|from)\s+(?:the\s+)?"
    r"(?:prior|previous|baseline)\s+"
    r"(?:exam|study|film|radiograph))\b",
    re.IGNORECASE,
)

# Target nouns onto which a fabricated measurement is grammatically pinned.
# v3.1: re-exported from ``data.augmentation.hard_negative_generator`` so the
# Claim-based and string-based fabrication paths cannot drift (self-check
# MEDIUM #4).  The Claim path prefers observation entities and falls back
# to this list; the string path uses this list directly.
_MEASUREMENT_TARGET_NOUNS = list(_MEASUREMENT_TARGET_NOUNS_SHARED)

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


#: Module-level singleton for the LLM extractor — lazy-loaded the first
#: time ``extract_claims(report, use_llm_extractor=True)`` is called so a
#: user can opt in from the CLI without paying the Phi-3 load cost when
#: the flag is off.  Initialised to ``None``; set by
#: ``_get_llm_extractor`` below.
_LLM_EXTRACTOR = None


def _get_llm_extractor():
    """Lazily construct the shared ``LLMClaimExtractor`` instance."""
    global _LLM_EXTRACTOR
    if _LLM_EXTRACTOR is None:
        from models.decomposer.llm_claim_extractor import LLMClaimExtractor

        _LLM_EXTRACTOR = LLMClaimExtractor(use_llm=True)
    return _LLM_EXTRACTOR


def extract_claims(
    report: str,
    use_llm_extractor: bool = False,
) -> list[str]:
    """Extract individual atomic claims from a report section.

    Args:
        report: Raw report text (``section_impression`` or
            ``section_findings``) from the CheXpert Plus DataFrame.
        use_llm_extractor: If True, route through the v2 context-aware
            ``LLMClaimExtractor`` (Phi-3-mini + rule-based fallback).
            Default False preserves v1 behaviour.

    Returns:
        List of claim strings.  Empty list for reports shorter than 10
        characters or after HISTORY-line filtering.
    """
    if not report or report == "nan" or len(report) < 10:
        return []

    if use_llm_extractor:
        extractor = _get_llm_extractor()
        # LLMClaimExtractor returns list[dict]; we only need the text.
        return [
            c["claim"] for c in extractor.extract_claims(report)
            if c.get("claim")
        ]

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

    # ---------------------------------------------------------------
    # v3 taxonomy — fabrication perturbations (types 9–11)
    # ---------------------------------------------------------------
    elif neg_type == "fabricate_measurement":
        # Skip if the claim already has a numeric measurement
        if _EXISTING_MEASUREMENT_RE.search(claim):
            return None
        target_noun = None
        for noun in _MEASUREMENT_TARGET_NOUNS:
            if re.search(rf'\b{re.escape(noun)}\b', claim_lower):
                target_noun = noun
                break
        if not target_noun:
            return None
        measurement = rng.choice(FABRICATED_MEASUREMENT_UNITS)
        new_text, n = re.subn(
            rf'\b{re.escape(target_noun)}\b',
            f"{measurement} {target_noun}",
            claim,
            count=1,
            flags=re.IGNORECASE,
        )
        if n == 0 or new_text == claim:
            return None
        return new_text

    elif neg_type == "fabricate_prior":
        if _EXISTING_PRIOR_RE.search(claim):
            return None
        stripped = claim.rstrip().rstrip(".")
        if not stripped:
            return None
        phrase = rng.choice(FABRICATED_PRIOR_PHRASES)
        return f"{stripped}, {phrase}."

    elif neg_type == "fabricate_temporal":
        if _EXISTING_TEMPORAL_RE.search(claim) or _EXISTING_PRIOR_RE.search(claim):
            return None
        stripped = claim.rstrip().rstrip(".")
        if not stripped:
            return None
        phrase = rng.choice(FABRICATED_TEMPORAL_DATES)
        return f"{stripped}, {phrase}."

    # ---------------------------------------------------------------
    # v3 taxonomy — compound (type 12) — two-error / three-error variants
    # ---------------------------------------------------------------
    elif neg_type == "compound_2err":
        return _compound_perturbation(claim, rng, n_errors=2)
    elif neg_type == "compound_3err":
        return _compound_perturbation(claim, rng, n_errors=3)

    return None


# Deterministic order in which _compound_perturbation tries single-type
# perturbations.  Mirrors the order used by
# data/augmentation/hard_negative_generator.py:_COMPOUND_ORDER so the two
# code paths stay in sync.
_COMPOUND_SINGLE_TYPES = [
    "laterality_swap",
    "finding_substitution",
    "severity_swap",
    "temporal_error",
    "negation",
    "region_swap",
    "device_line_error",
    "fabricate_measurement",
    "fabricate_prior",
    "fabricate_temporal",
]


def _compound_perturbation(
    claim: str, rng: random.Random, n_errors: int,
) -> str | None:
    """Chain *n_errors* distinct single-type perturbations on *claim*.

    Returns None if any sub-step fails, if the chained perturbations
    don't change the text, or if n_errors < 2.
    """
    if n_errors < 2:
        return None
    # Sample n_errors distinct types; apply in the fixed _COMPOUND_SINGLE_TYPES
    # order so the result is deterministic given rng state.
    pool_size = min(n_errors, len(_COMPOUND_SINGLE_TYPES))
    selected = set(rng.sample(_COMPOUND_SINGLE_TYPES, k=pool_size))
    ordered = [t for t in _COMPOUND_SINGLE_TYPES if t in selected]
    if len(ordered) < n_errors:
        return None

    current = claim
    changed_any = False
    for neg_type in ordered:
        nxt = generate_hard_negative(current, neg_type, rng)
        if nxt is None or nxt == current:
            return None
        current = nxt
        changed_any = True

    if not changed_any or current == claim:
        return None
    return current


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
    use_llm_extractor: bool = False,
) -> list[dict]:
    """Create balanced evaluation claim sets with all 13 hard-negative types.

    Args:
        df: CheXpert Plus DataFrame (filtered to desired split).
        n_supported: Number of supported claims (label=0).
        n_contradicted: Number of contradicted claims (label=1).
        n_insufficient: Number of insufficient-evidence claims (label=2).
        seed: Random seed.
        use_llm_extractor: If True, use ``LLMClaimExtractor`` (Phi-3-mini
            context-aware decomposer) instead of the naive regex
            sentence splitter.  Routed directly into ``extract_claims``.

    Returns:
        List of claim dicts with keys: claim, evidence, label,
        pathology, negative_type, patient_id.
    """
    rng = random.Random(seed)

    if use_llm_extractor:
        logger.info(
            "extract_claims: routing through LLMClaimExtractor "
            "(Phi-3-mini, context-aware)"
        )

    # Extract all claims with their evidence
    all_claims = []
    for _, row in df.iterrows():
        pid = str(row.get("deid_patient_id", ""))
        impression = str(row.get("section_impression", ""))
        findings = str(row.get("section_findings", ""))

        report_claims = extract_claims(impression, use_llm_extractor=use_llm_extractor)
        if not report_claims:
            report_claims = extract_claims(findings, use_llm_extractor=use_llm_extractor)

        for claim in report_claims:
            # Evidence = other sentences from same report
            evidence_pool = [c for c in report_claims if c != claim]
            if findings and findings != "nan":
                evidence_pool.extend(
                    extract_claims(findings, use_llm_extractor=use_llm_extractor)
                )

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
        example = {
            "claim": item["claim"],
            "evidence": item["evidence"][:2],
            "label": 0,  # Supported
            "pathology": item["pathology"],
            "negative_type": "none",
            "patient_id": item["patient_id"],
        }
        # Provenance: evidence is the same radiologist-written report the
        # claim was extracted from. Oracle text, trust tier TRUSTED.
        example.update(default_provenance(
            source_type=EvidenceSourceType.ORACLE_REPORT_TEXT,
            claim_generator_id=None,
            evidence_generator_id=None,
        ))
        supported_examples.append(example)

    # =========================================================
    # 2. CONTRADICTED claims (v3 taxonomy — 13 hard-negative types)
    # =========================================================
    # v2 had 8 single-type perturbations.  v3 adds:
    #   9. fabricate_measurement  — inject "3 mm" / "1.2 cm" into claims
    #                                that had no measurement
    #  10. fabricate_prior        — inject "compared to the prior film"
    #  11. fabricate_temporal     — inject "since 3 days ago"
    #  12. compound_2err          — chain 2 distinct single-type perturbations
    #  13. compound_3err          — chain 3 distinct single-type perturbations
    # The compound_2err / compound_3err pair is split 60 / 40 per the
    # realistic-complexity distribution from Chen et al. 2025 — in a
    # pool of real multi-error radiology claims most have exactly 2
    # errors, fewer have 3+.
    neg_types = [
        "negation",
        "laterality_swap",
        "severity_swap",
        "temporal_error",
        "finding_substitution",
        "region_swap",
        "device_line_error",
        "omission_as_support",
        "fabricate_measurement",
        "fabricate_prior",
        "fabricate_temporal",
        "compound_2err",
        "compound_3err",
    ]
    #: Base quota per type = n_contradicted / 13 (even split).  The
    #: compound pair then gets rebalanced 60/40 of whatever combined
    #: share it would have received under the even split.
    contradicted_examples = []
    neg_pool = list(all_claims)
    rng.shuffle(neg_pool)

    # Track per-type counts for balanced sampling.
    # v3.1 (self-check MEDIUM #3): quota must sum to EXACTLY ``n_contradicted``
    # — the earlier ``base_quota = n // 13`` formulation lost the floor
    # remainder, leaking e.g. 8 examples on n=5000. We fold the leftover
    # into compound_2err, which is the most abundant type.
    base_quota = n_contradicted // len(neg_types)
    type_quota: dict[str, int] = {t: base_quota for t in neg_types}
    # Rebalance compound pair to 60/40 of their combined 2/13 share
    compound_combined = base_quota * 2
    type_quota["compound_2err"] = int(round(compound_combined * 0.60))
    type_quota["compound_3err"] = compound_combined - type_quota["compound_2err"]
    remainder = n_contradicted - sum(type_quota.values())
    if remainder > 0:
        # Absorb the floor-division remainder into the most generator-friendly
        # bucket (compound_2err has the widest sampling surface).
        type_quota["compound_2err"] += remainder
    assert sum(type_quota.values()) == n_contradicted, (
        f"type_quota sums to {sum(type_quota.values())}, "
        f"expected {n_contradicted}"
    )
    type_counts = {t: 0 for t in neg_types}

    attempts = 0
    max_attempts = n_contradicted * 20
    rejected_for_support = 0
    while len(contradicted_examples) < n_contradicted and attempts < max_attempts:
        item = neg_pool[attempts % len(neg_pool)]
        # Prefer under-represented types (type_quota is a per-type dict in v3)
        underrep = [t for t in neg_types if type_counts[t] < type_quota[t]]
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

        example = {
            "claim": neg_claim,
            "evidence": evidence,
            "label": 1,  # Contradicted
            "pathology": item["pathology"],
            "negative_type": neg_type,
            "patient_id": item["patient_id"],
        }
        # Provenance: evidence is still the same radiologist-written oracle
        # report. The claim is a hard-negative perturbation of the original,
        # but the evidence itself is trusted oracle text.
        example.update(default_provenance(
            source_type=EvidenceSourceType.ORACLE_REPORT_TEXT,
            claim_generator_id=None,
            evidence_generator_id=None,
        ))
        contradicted_examples.append(example)
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

        example = {
            "claim": item["claim"],
            "evidence": other["evidence"][:2],
            "label": 2,  # Insufficient Evidence
            "pathology": item["pathology"],
            "negative_type": "mismatched_evidence",
            "patient_id": item["patient_id"],
        }
        # Provenance: evidence comes from a DIFFERENT radiologist-written
        # report (different patient + different pathology). Still oracle
        # text, so trust tier TRUSTED — the independence issue we care
        # about is "same generator wrote both" and neither side was
        # generated by a model here.
        example.update(default_provenance(
            source_type=EvidenceSourceType.ORACLE_REPORT_TEXT,
            claim_generator_id=None,
            evidence_generator_id=None,
        ))
        insufficient_examples.append(example)

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
    parser.add_argument(
        "--use-llm-extractor", action="store_true",
        help=(
            "Route claim extraction through LLMClaimExtractor (Phi-3-mini, "
            "context-aware) instead of the naive regex sentence splitter. "
            "Default: off, preserving v1 behaviour. Turn on for v3 "
            "regeneration (adds ~2s/report on CPU)."
        ),
    )
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
            use_llm_extractor=args.use_llm_extractor,
        )

        out_file = output_dir / f"{split_name}_claims.json"
        with open(out_file, "w") as f:
            json.dump(claims, f, indent=2)
        logger.info(f"Saved {len(claims)} claims to {out_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
