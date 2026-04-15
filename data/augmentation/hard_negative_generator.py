"""Hard negative generator for ClaimGuard-CXR.

Constructs 12 types of clinically motivated hard negatives from
RadGraph-annotated radiology claims for training the ClaimGuard-CXR verifier.
Each perturbation is anatomically or semantically grounded to produce
plausible-but-wrong claims that stress-test the verifier beyond simple
syntactic paraphrases.

Negative types (v3 taxonomy — Chen et al. 2025 + ReXErr 2024 coverage):
     1. laterality_swap        — flip left/right (incl. abbreviations)
     2. finding_substitution   — swap pathology for a co-regional alternative
     3. negation               — insert or remove negation
     4. region_swap            — relocate finding to a different anatomical region
     5. severity_swap          — substitute severity grade
     6. temporal_error         — substitute temporal descriptor
     7. device_error           — swap device type or position descriptor
     8. omission               — fabricate a claim about a finding absent from
                                  the image (produces "Insufficient Evidence")
     9. fabricate_measurement  — inject a fabricated size / measurement
                                  (e.g. "3 mm", "1.2 cm") into a claim that
                                  had no measurement in the original text
    10. fabricate_prior        — inject a fabricated comparison to a prior
                                  exam ("compared to the prior study from
                                  2 weeks ago")
    11. fabricate_temporal     — inject a fabricated absolute / relative date
                                  ("since 3 days ago", "from yesterday's film")
    12. compound_perturbation  — apply 2 or 3 distinct single-type perturbations
                                  in sequence (realistic multi-error mode).
                                  Registered under two string keys so the
                                  training-data pipeline can hit the 60/40
                                  (2-error / 3-error) quota cleanly:
                                      compound_2err
                                      compound_3err

Types 9-12 were added in the v3 sprint to close the gap against real-world
hallucinations that the 8-type v2 taxonomy missed (Chen et al. 2025 flagged
fabricated_temporal as a distinct omission subtype; ReXErr 2024 flagged
measurements/priors as the most common ungrounded fabrication modes).
"""

from __future__ import annotations

import copy
import logging
import random
import re
from dataclasses import replace
from typing import Callable, Optional

from ..preprocessing.radgraph_parser import (
    CHEXPERT_ONTOLOGY,
    Claim,
    Entity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants referenced by downstream modules
# ---------------------------------------------------------------------------

#: Maps broad anatomical regions to pathology findings that commonly occur there.
#: Used by finding_substitution and omission generators.
FINDING_SUBSTITUTION_MAP: dict[str, list[str]] = {
    "lung": [
        "pneumonia",
        "atelectasis",
        "consolidation",
        "pulmonary edema",
        "pneumothorax",
        "lung opacity",
        "pulmonary nodule",
        "pulmonary mass",
        "air-space disease",
        "interstitial opacities",
        "ground-glass opacity",
    ],
    "pleura": [
        "pleural effusion",
        "pleural thickening",
        "pleural plaque",
        "pneumothorax",
        "hydropneumothorax",
        "hemothorax",
        "pleural calcification",
    ],
    "mediastinum": [
        "mediastinal widening",
        "enlarged cardiomediastinum",
        "mediastinal mass",
        "pneumomediastinum",
        "mediastinal lymphadenopathy",
        "mediastinal shift",
    ],
    "heart": [
        "cardiomegaly",
        "cardiac enlargement",
        "pericardial effusion",
        "right heart enlargement",
        "left ventricular enlargement",
    ],
    "bone": [
        "rib fracture",
        "clavicle fracture",
        "vertebral fracture",
        "compression fracture",
        "lytic lesion",
        "sclerotic lesion",
        "osteopenia",
    ],
    "hilum": [
        "hilar enlargement",
        "hilar lymphadenopathy",
        "hilar prominence",
        "hilar mass",
        "vascular prominence",
    ],
    "diaphragm": [
        "elevated hemidiaphragm",
        "subphrenic air",
        "free air under diaphragm",
        "diaphragmatic hernia",
        "hemidiaphragm flattening",
    ],
    "airway": [
        "tracheal deviation",
        "tracheal narrowing",
        "bronchial obstruction",
        "endobronchial lesion",
        "tracheal shift",
    ],
}

#: Canonical device type tokens.  Swaps are drawn from this list.
DEVICE_TYPES: list[str] = [
    "endotracheal tube",
    "ETT",
    "PICC line",
    "PICC",
    "central venous catheter",
    "CVC",
    "nasogastric tube",
    "NG tube",
    "orogastric tube",
    "chest tube",
    "pleural drain",
    "pacemaker",
    "ICD",
    "cardiac monitor leads",
    "Port-a-Cath",
    "port",
    "Swan-Ganz catheter",
    "pulmonary artery catheter",
    "intra-aortic balloon pump",
    "IABP",
    "tracheostomy tube",
]

#: Position / placement descriptor tokens used in device claims.
DEVICE_POSITIONS: list[str] = [
    "in appropriate position",
    "in satisfactory position",
    "in good position",
    "appropriately positioned",
    "well-positioned",
    "malpositioned",
    "in malposition",
    "tip in the right atrium",
    "tip in the right ventricle",
    "tip above the carina",
    "tip below the carina",
    "tip at the carina",
    "tip at the thoracic inlet",
    "coiled in the esophagus",
    "advanced too far distally",
    "positioned too proximally",
    "tip at the cavoatrial junction",
    "tip in the superior vena cava",
]

#: Fabricated measurement strings for the fabricate_measurement generator.
#: Each string is a numeric size that can be grammatically suffixed onto an
#: existing finding ("a 3 mm pulmonary nodule").  The ranges are chosen to be
#: clinically plausible (sub-cm nodules, small masses) but completely
#: ungrounded — any verifier trained only on hard-negatives without
#: measurement fabrication has no defence against this attack.
FABRICATED_MEASUREMENT_UNITS: list[str] = [
    "3 mm",
    "5 mm",
    "7 mm",
    "9 mm",
    "1.2 cm",
    "1.5 cm",
    "1.8 cm",
    "2.4 cm",
    "3 cm",
    "4 x 6 mm",
    "5 x 8 mm",
    "1 x 2 cm",
]

#: Fabricated prior-exam comparison phrases.  Each phrase asserts a
#: comparison to a prior study that may or may not exist; the key
#: failure mode is a model that self-consistently generates "compared to
#: the prior film..." even when no prior is available.
FABRICATED_PRIOR_PHRASES: list[str] = [
    "compared to the prior exam from 2 weeks ago",
    "since the previous study",
    "unchanged from the prior film",
    "stable compared to the comparison exam",
    "new relative to the most recent comparison",
    "interval change from the baseline study",
    "compared with the prior radiograph from last month",
]

#: Fabricated temporal / date phrases.  Distinct from FABRICATED_PRIOR_PHRASES
#: in that these inject an absolute or relative calendar reference without
#: claiming a prior-exam comparison.
FABRICATED_TEMPORAL_DATES: list[str] = [
    "since 3 days ago",
    "from last week's study",
    "compared to yesterday's film",
    "noted on the prior exam from 2 days ago",
    "present since the study on 01/12/2026",
    "new since the admission radiograph",
    "interval change over the past 48 hours",
]

#: Target nouns onto which a fabricated measurement is grammatically pinned.
#: The Claim-based ``fabricate_measurement`` path prefers ``claim.entities``
#: when available, falling back to a substring match from this list;
#: ``scripts/prepare_eval_data.py`` operates on raw strings and uses this
#: list directly.  Kept as a single module-level constant so both paths
#: cannot drift (self-check MEDIUM #4).
MEASUREMENT_TARGET_NOUNS: list[str] = [
    "nodule",
    "mass",
    "opacity",
    "lesion",
    "effusion",
    "consolidation",
    "infiltrate",
    "atelectasis",
]

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

# All left-token variants  →  corresponding right-token variants (and vice versa)
_LEFT_TOKENS: list[str] = [
    "left",
    "left-sided",
    "left sided",
    "LLL",  # left lower lobe
    "LUL",  # left upper lobe
    "LML",  # left middle lobe (rare but used in some reports)
    "L ",   # abbreviated lateral marker (careful with word boundary below)
]
_RIGHT_TOKENS: list[str] = [
    "right",
    "right-sided",
    "right sided",
    "RLL",  # right lower lobe
    "RUL",  # right upper lobe
    "RML",  # right middle lobe
    "R ",
]

# Ordered severity grades (low → high)
_SEVERITY_GROUPS: list[list[str]] = [
    ["mild", "minimal", "trace", "slight", "small", "tiny"],
    ["moderate", "medium", "moderate-sized"],
    ["severe", "large", "massive", "extensive", "marked"],
]

_SEVERITY_FLAT: list[str] = [w for grp in _SEVERITY_GROUPS for w in grp]

# Temporal descriptor tokens
_TEMPORAL_TOKENS: list[str] = [
    "new",
    "newly",
    "unchanged",
    "stable",
    "improved",
    "improving",
    "worsening",
    "worsened",
    "worse",
    "resolved",
    "resolving",
    "persistent",
    "progressed",
    "increasing",
    "decreased",
    "decreasing",
]

# Negation patterns (order matters: more specific patterns first)
_NEGATION_INSERTIONS: list[str] = ["no ", "no evidence of ", "without "]
_NEGATION_REMOVALS: list[re.Pattern] = [
    re.compile(r"\bno evidence of\s+", re.IGNORECASE),
    re.compile(r"\bno signs? of\s+", re.IGNORECASE),
    re.compile(r"\bwithout\s+", re.IGNORECASE),
    re.compile(r"\bno\s+", re.IGNORECASE),
    re.compile(r"\babsence of\s+", re.IGNORECASE),
]

# Anatomy regions used for region_swap
_ANATOMY_REGIONS: list[str] = [
    "right lower lobe",
    "left lower lobe",
    "right upper lobe",
    "left upper lobe",
    "right middle lobe",
    "left mid lung",
    "right lung base",
    "left lung base",
    "bilateral lung bases",
    "right hemithorax",
    "left hemithorax",
    "right costophrenic angle",
    "left costophrenic angle",
    "right hilum",
    "left hilum",
    "mediastinum",
    "left pleural space",
    "right pleural space",
    "pericardium",
    "retrocardiac region",
    "perihilar region",
    "right paratracheal region",
]

# Label constants
_LABEL_CONTRADICTED = "Contradicted"
_LABEL_INSUFFICIENT = "Insufficient Evidence"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _clone_claim(claim: Claim, new_text: str) -> Claim:
    """Return a shallow copy of *claim* with text replaced by *new_text*."""
    return Claim(
        text=new_text,
        pathology_category=claim.pathology_category,
        entities=list(claim.entities),
        relations=list(claim.relations),
        is_negated=claim.is_negated,
        laterality=claim.laterality,
        severity=claim.severity,
        anatomy=claim.anatomy,
    )


def _has_any_token(text: str, tokens: list[str]) -> bool:
    """Return True if *text* contains any token (case-insensitive word match)."""
    text_lower = text.lower()
    for tok in tokens:
        pattern = re.escape(tok.strip())
        if re.search(rf"\b{pattern}\b", text_lower):
            return True
    return False


def _replace_first(text: str, old: str, new: str, *, case_insensitive: bool = True) -> str:
    """Replace the first occurrence of *old* with *new* in *text*.

    Uses word-boundary matching for single-word tokens to avoid partial matches.
    Returns the original *text* unchanged if *old* is not found.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = rf"\b{re.escape(old.strip())}\b"
    result, count = re.subn(pattern, new, text, count=1, flags=flags)
    if count == 0:
        return text
    return result


def _replace_all_tokens(text: str, src_list: list[str], dst_list: list[str]) -> Optional[str]:
    """Replace the first matched token from *src_list* with the corresponding
    token in *dst_list* at the same index.

    Returns the modified string, or None if no replacement was made.
    """
    for i, src in enumerate(src_list):
        dst = dst_list[i]
        new_text = _replace_first(text, src.strip(), dst.strip())
        if new_text != text:
            return new_text
    return None


# ---------------------------------------------------------------------------
# Hard negative generators — types 1-7 return "Contradicted"
# ---------------------------------------------------------------------------


def laterality_swap(claim: Claim) -> Optional[Claim]:
    """Swap all left/right tokens in *claim* (incl. lobe abbreviations).

    Handles:  left <-> right,  left-sided <-> right-sided,
              LLL/LUL <-> RLL/RUL,  bilateral (returns None, ambiguous).

    Args:
        claim: Source claim.

    Returns:
        Modified Claim with label "Contradicted", or None if swap is not
        applicable (no laterality found, or claim is bilateral).
    """
    if claim.laterality is None:
        return None
    if claim.laterality == "bilateral":
        return None

    text = claim.text

    # Multi-pass replacement using a placeholder to avoid double-swapping.
    # Replace LEFT → placeholder, RIGHT → LEFT, placeholder → RIGHT.
    PLACEHOLDER = "___LATERALITY_PLACEHOLDER___"

    # Define paired patterns ordered longest-first to avoid partial overlaps.
    left_patterns = ["left-sided", "left sided", "LLL", "LUL", "LML", "left"]
    right_patterns = ["right-sided", "right sided", "RLL", "RUL", "RML", "right"]

    new_text = text
    changed = False

    if claim.laterality == "left":
        for lp, rp in zip(left_patterns, right_patterns):
            candidate = _replace_first(new_text, lp, rp)
            if candidate != new_text:
                new_text = candidate
                changed = True
                break
    elif claim.laterality == "right":
        for lp, rp in zip(left_patterns, right_patterns):
            candidate = _replace_first(new_text, rp, lp)
            if candidate != new_text:
                new_text = candidate
                changed = True
                break

    if not changed or new_text == text:
        return None

    modified = _clone_claim(claim, new_text)
    modified.laterality = "right" if claim.laterality == "left" else "left"
    return modified


def finding_substitution(claim: Claim, rng: Optional[random.Random] = None) -> Optional[Claim]:
    """Replace the primary pathology finding with a different finding from the
    same anatomical region family.

    Args:
        claim: Source claim.
        rng: Optional seeded Random instance for reproducibility.

    Returns:
        Modified Claim with label "Contradicted", or None if no suitable
        substitution exists.
    """
    if rng is None:
        rng = random.Random()

    if claim.pathology_category in ("No Finding", "Support Devices"):
        return None

    # Determine anatomical region bucket
    region = _infer_region(claim)
    candidates = FINDING_SUBSTITUTION_MAP.get(region, [])

    # Collect entity text to exclude from candidates
    entity_texts = {e.tokens.lower() for e in claim.entities}

    # Filter out anything that looks like the current finding
    current_lower = claim.text.lower()
    filtered = [
        c for c in candidates
        if c.lower() not in entity_texts
        and c.lower() not in current_lower
        and not any(et in c.lower() for et in entity_texts)
    ]

    if not filtered:
        return None

    replacement = rng.choice(filtered)

    # Build new text: replace the first entity token that matches an OBS entity
    new_text = claim.text
    replaced = False
    for entity in claim.entities:
        if entity.is_observation and entity.tokens:
            candidate_text = _replace_first(new_text, entity.tokens, replacement)
            if candidate_text != new_text:
                new_text = candidate_text
                replaced = True
                break

    if not replaced:
        # Fallback: append the substitution after trimming the original
        new_text = f"{replacement} noted"

    modified = _clone_claim(claim, new_text)
    modified.pathology_category = _category_for_finding(replacement)
    return modified


def negation(claim: Claim) -> Optional[Claim]:
    """Flip the negation polarity of *claim*.

    - Present claim  → insert "No evidence of" prefix.
    - Negated claim  → strip the negation marker.

    Args:
        claim: Source claim.

    Returns:
        Modified Claim with label "Contradicted".
    """
    text = claim.text

    if claim.is_negated:
        # Strategy 1: "without X" → "with X"  (grammatically cleanest for this pattern)
        candidate = re.sub(r"\bwithout\b", "with", text, count=1, flags=re.IGNORECASE)
        if candidate != text:
            modified = _clone_claim(claim, candidate)
            modified.is_negated = False
            return modified

        # Strategy 2: strip explicit negation markers ("no evidence of", "no signs of",
        # "no X", "absence of X")
        new_text = text
        for pattern in _NEGATION_REMOVALS:
            candidate = pattern.sub("", new_text, count=1).strip()
            if candidate != new_text:
                new_text = candidate
                break

        if new_text == text:
            return None

        # Capitalize first letter
        new_text = new_text[0].upper() + new_text[1:] if new_text else new_text
        modified = _clone_claim(claim, new_text)
        modified.is_negated = False
        return modified
    else:
        # Insert negation
        # Handle "with X" → "without X" first
        candidate = re.sub(r"\bwith\b", "without", text, count=1, flags=re.IGNORECASE)
        if candidate != text:
            modified = _clone_claim(claim, candidate)
            modified.is_negated = True
            return modified

        # Prepend "No evidence of"
        new_text = f"No evidence of {text[0].lower()}{text[1:]}"
        modified = _clone_claim(claim, new_text)
        modified.is_negated = True
        return modified


def region_swap(claim: Claim, rng: Optional[random.Random] = None) -> Optional[Claim]:
    """Attribute the finding to a different anatomical region.

    Args:
        claim: Source claim.
        rng: Optional seeded Random instance.

    Returns:
        Modified Claim with label "Contradicted", or None if no suitable
        region swap is possible.
    """
    if rng is None:
        rng = random.Random()

    current_anatomy = claim.anatomy

    # Build candidate pool excluding current anatomy
    candidates = [r for r in _ANATOMY_REGIONS if r != current_anatomy]
    if not candidates:
        return None

    new_region = rng.choice(candidates)

    if current_anatomy:
        new_text = _replace_first(claim.text, current_anatomy, new_region)
        if new_text == claim.text:
            # Try appending if the original anatomy text doesn't appear verbatim
            new_text = f"{claim.text} in the {new_region}"
    else:
        new_text = f"{claim.text} in the {new_region}"

    modified = _clone_claim(claim, new_text)
    modified.anatomy = new_region
    return modified


def severity_swap(claim: Claim, rng: Optional[random.Random] = None) -> Optional[Claim]:
    """Substitute the severity descriptor with one from a different severity tier.

    Mild → severe or moderate; moderate → mild or severe; severe → mild or
    moderate.  A token from a *different* tier is always chosen to ensure the
    swap is semantically meaningful.

    Args:
        claim: Source claim.
        rng: Optional seeded Random instance.

    Returns:
        Modified Claim with label "Contradicted", or None if no severity token
        is present.
    """
    if rng is None:
        rng = random.Random()

    if claim.severity is None:
        return None

    text_lower = claim.text.lower()

    # Find the first severity token present in the claim and its tier index
    found_token: Optional[str] = None
    current_tier: int = -1
    for tier_idx, tier in enumerate(_SEVERITY_GROUPS):
        for token in tier:
            if re.search(rf"\b{re.escape(token)}\b", text_lower):
                found_token = token
                current_tier = tier_idx
                break
        if found_token:
            break

    if found_token is None or current_tier == -1:
        return None

    # Pick a replacement from a different tier
    other_tiers = [i for i in range(len(_SEVERITY_GROUPS)) if i != current_tier]
    if not other_tiers:
        return None

    target_tier_idx = rng.choice(other_tiers)
    replacement_token = rng.choice(_SEVERITY_GROUPS[target_tier_idx])

    new_text = _replace_first(claim.text, found_token, replacement_token)
    if new_text == claim.text:
        return None

    modified = _clone_claim(claim, new_text)
    # Infer new severity tier label
    tier_to_label = {0: "mild", 1: "moderate", 2: "severe"}
    modified.severity = tier_to_label[target_tier_idx]
    return modified


def temporal_error(claim: Claim, rng: Optional[random.Random] = None) -> Optional[Claim]:
    """Swap a temporal descriptor (new/unchanged/resolved/worsening/stable).

    Args:
        claim: Source claim.
        rng: Optional seeded Random instance.

    Returns:
        Modified Claim with label "Contradicted", or None if no temporal token
        is found.
    """
    if rng is None:
        rng = random.Random()

    text_lower = claim.text.lower()

    found_token: Optional[str] = None
    for tok in _TEMPORAL_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", text_lower):
            found_token = tok
            break

    if found_token is None:
        return None

    # Pick a replacement that is semantically opposite or at least different
    _OPPOSITES: dict[str, list[str]] = {
        "new": ["unchanged", "stable", "resolved"],
        "newly": ["unchanged", "stable"],
        "unchanged": ["new", "worsening", "resolved"],
        "stable": ["new", "worsening", "progressed"],
        "improved": ["worsening", "worsened", "new"],
        "improving": ["worsening", "progressed"],
        "worsening": ["stable", "improved", "resolved"],
        "worsened": ["stable", "improved", "resolved"],
        "worse": ["stable", "improved", "resolved"],
        "resolved": ["new", "worsening", "unchanged"],
        "resolving": ["worsening", "new"],
        "persistent": ["resolved", "new"],
        "progressed": ["stable", "improved", "resolved"],
        "increasing": ["decreasing", "stable", "resolved"],
        "decreased": ["increased", "worsening", "new"],
        "decreasing": ["increasing", "worsening", "new"],
    }

    replacements = _OPPOSITES.get(found_token.lower(), [])
    # Fallback: any other temporal token
    if not replacements:
        replacements = [t for t in _TEMPORAL_TOKENS if t != found_token.lower()]

    if not replacements:
        return None

    replacement = rng.choice(replacements)
    new_text = _replace_first(claim.text, found_token, replacement)
    if new_text == claim.text:
        return None

    return _clone_claim(claim, new_text)


def device_error(claim: Claim, rng: Optional[random.Random] = None) -> Optional[Claim]:
    """Introduce a device-related error: swap device type or position descriptor.

    Strategy:
        1. If the claim contains a known device position descriptor, swap it
           for a conflicting one (e.g., "appropriate position" → "malpositioned").
        2. If the claim contains a known device type, swap it for a different
           device type.
        3. Otherwise return None (claim is not device-related).

    Args:
        claim: Source claim.
        rng: Optional seeded Random instance.

    Returns:
        Modified Claim with label "Contradicted", or None if the claim does
        not mention a device.
    """
    if rng is None:
        rng = random.Random()

    if claim.pathology_category != "Support Devices":
        return None

    text = claim.text

    # --- Try position swap first ---
    found_pos: Optional[str] = None
    for pos in DEVICE_POSITIONS:
        if pos.lower() in text.lower():
            found_pos = pos
            break

    if found_pos:
        # Swap between "appropriate/good/satisfactory" and "malpositoned/distal/proximal"
        _GOOD_POSITIONS = [
            "in appropriate position",
            "in satisfactory position",
            "in good position",
            "appropriately positioned",
            "well-positioned",
            "tip at the cavoatrial junction",
            "tip in the superior vena cava",
        ]
        _BAD_POSITIONS = [
            "malpositioned",
            "in malposition",
            "tip in the right atrium",
            "tip in the right ventricle",
            "tip above the carina",
            "coiled in the esophagus",
            "advanced too far distally",
            "positioned too proximally",
        ]
        if found_pos in _GOOD_POSITIONS:
            replacement = rng.choice(_BAD_POSITIONS)
        elif found_pos in _BAD_POSITIONS:
            replacement = rng.choice(_GOOD_POSITIONS)
        else:
            # Pick any different position
            replacement = rng.choice([p for p in DEVICE_POSITIONS if p != found_pos])

        # Case-insensitive replacement preserving original capitalisation position
        new_text = re.sub(re.escape(found_pos), replacement, text, count=1, flags=re.IGNORECASE)
        if new_text != text:
            return _clone_claim(claim, new_text)

    # --- Try device type swap ---
    found_dev: Optional[str] = None
    for dev in DEVICE_TYPES:
        if re.search(rf"\b{re.escape(dev)}\b", text, flags=re.IGNORECASE):
            found_dev = dev
            break

    if found_dev:
        candidates = [d for d in DEVICE_TYPES if d.lower() != found_dev.lower()]
        if not candidates:
            return None
        replacement_dev = rng.choice(candidates)
        new_text = re.sub(
            rf"\b{re.escape(found_dev)}\b",
            replacement_dev,
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if new_text != text:
            return _clone_claim(claim, new_text)

    return None


def omission(
    claim: Claim,
    all_claims: list[Claim],
    rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Generate a claim about a finding NOT present in the image.

    Selects a finding from FINDING_SUBSTITUTION_MAP that does not appear in
    *all_claims* and creates a positive-polarity claim asserting its presence.
    This produces an "Insufficient Evidence" negative (the verifier cannot
    confirm the finding because the image simply doesn't show it).

    Args:
        claim: The reference claim providing context (anatomy, category).
        all_claims: All claims extracted from the same report/image, used to
            exclude findings already mentioned.
        rng: Optional seeded Random instance.

    Returns:
        A new Claim asserting an absent finding, with label
        "Insufficient Evidence", or None if no suitable absent finding can
        be constructed.
    """
    if rng is None:
        rng = random.Random()

    # Collect all finding text already present in the report
    present_texts = {c.text.lower() for c in all_claims}
    present_entity_tokens = {
        e.tokens.lower() for c in all_claims for e in c.entities if e.is_observation
    }

    region = _infer_region(claim)
    # Expand search to all regions if needed
    all_candidates: list[tuple[str, str]] = []  # (finding, region)
    for reg, findings in FINDING_SUBSTITUTION_MAP.items():
        for f in findings:
            all_candidates.append((f, reg))

    # Filter out anything already present in the report
    absent = [
        (f, reg)
        for f, reg in all_candidates
        if f.lower() not in present_entity_tokens
        and not any(f.lower() in pt for pt in present_texts)
    ]

    if not absent:
        return None

    # Prefer candidates from the same anatomical region for plausibility
    same_region = [(f, reg) for f, reg in absent if reg == region]
    pool = same_region if same_region else absent

    chosen_finding, chosen_region = rng.choice(pool)

    # Build a credible positive claim about the absent finding
    anatomy_phrases = {
        "lung": "in the lung",
        "pleura": "in the pleural space",
        "mediastinum": "in the mediastinum",
        "heart": "at the cardiac silhouette",
        "bone": "involving the osseous structures",
        "hilum": "at the hilum",
        "diaphragm": "at the diaphragm",
        "airway": "in the airway",
    }
    anatomy_phrase = anatomy_phrases.get(chosen_region, "")
    if anatomy_phrase:
        new_text = f"{chosen_finding.capitalize()} {anatomy_phrase}"
    else:
        new_text = chosen_finding.capitalize()

    return Claim(
        text=new_text,
        pathology_category=_category_for_finding(chosen_finding),
        entities=[],
        relations=[],
        is_negated=False,
        laterality=None,
        severity=None,
        anatomy=anatomy_phrase.replace("in the ", "").replace("at the ", "").strip() or None,
    )


# ---------------------------------------------------------------------------
# Hard negative generators — v3 extensions (fabrication + compound)
# ---------------------------------------------------------------------------


# A regex that matches any textual measurement already in a claim.
# Used by fabricate_measurement to skip claims that already have a size.
# v3.1: broadened to catch "3x4 mm", "6-mm", "6 × 4 mm", and hyphenated
# forms which an earlier version missed (self-check HIGH #5).
_EXISTING_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?"                                # leading size
    r"(?:\s*[x×-]\s*\d+(?:\.\d+)?)?"                  # optional x/× dim-2
    r"\s*[- ]?"                                        # optional separator
    r"(?:mm|cm|millimeter|centimeter)s?\b",            # unit
    re.IGNORECASE,
)

# Regex that matches any existing prior-exam language in a claim.  v3.1:
# added a bare "prior exam / previous study / in the interim / previously
# noted / grossly similar" alternation so the fabrication guards catch
# natural radiologist phrasing that the earlier connective-only form
# missed (self-check HIGH #1 + #2).
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


def fabricate_measurement(
    claim: Claim, rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Inject a fabricated measurement into a claim that had none.

    Rationale:
        A common real-world hallucination mode is a VLM confidently
        reporting a size for a finding even when it has no way to measure
        one ("1.2 cm pulmonary nodule").  Any verifier trained only on
        perturbations that edit existing text is blind to this attack —
        there is no corresponding "correct" measurement to compare to.

    Transformation:
        If the claim has no existing measurement and contains a nominal
        pathology entity, insert a randomly-chosen measurement from
        FABRICATED_MEASUREMENT_UNITS in front of the entity.  Return
        None if the claim already has a measurement or has nothing to
        attach one to.

    Args:
        claim: Source claim.
        rng:   Optional seeded Random instance for reproducibility.

    Returns:
        Modified Claim with label "Contradicted", or None if the
        fabrication is not applicable.
    """
    if rng is None:
        rng = random.Random()

    text = claim.text
    if _EXISTING_MEASUREMENT_RE.search(text):
        return None

    # Choose a target noun to prepend the measurement to.  Prefer an
    # observation entity with a known token string; fall back to the
    # pathology category if no entity is usable.
    target_noun: Optional[str] = None
    entity_candidates = [
        e.tokens for e in claim.entities
        if getattr(e, "is_observation", False) and e.tokens
    ]
    if entity_candidates:
        target_noun = entity_candidates[0]
    else:
        for kw in MEASUREMENT_TARGET_NOUNS:
            if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
                target_noun = kw
                break

    if not target_noun:
        return None

    measurement = rng.choice(FABRICATED_MEASUREMENT_UNITS)
    # Insert "<measurement> " directly in front of the first occurrence of
    # the target noun (case-insensitive), preserving the original casing
    # of the rest of the string.
    pattern = rf"\b{re.escape(target_noun)}\b"
    new_text, n = re.subn(
        pattern, f"{measurement} {target_noun}", text, count=1, flags=re.IGNORECASE,
    )
    if n == 0 or new_text == text:
        return None

    return _clone_claim(claim, new_text)


def fabricate_prior(
    claim: Claim, rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Inject a fabricated comparison-to-prior clause.

    Attaches a ``compared to the prior exam …`` style clause to the end
    of the claim.  Skips claims that already have comparison language.

    Args:
        claim: Source claim.
        rng:   Optional seeded Random instance for reproducibility.

    Returns:
        Modified Claim with label "Contradicted", or None if the claim
        already has prior-exam language.
    """
    if rng is None:
        rng = random.Random()

    text = claim.text.rstrip()
    if _EXISTING_PRIOR_RE.search(text):
        return None
    if not text:
        return None

    phrase = rng.choice(FABRICATED_PRIOR_PHRASES)
    # Drop a trailing period if present, add the phrase, end with a period.
    stripped = text.rstrip(".")
    new_text = f"{stripped}, {phrase}."
    return _clone_claim(claim, new_text)


def fabricate_temporal(
    claim: Claim, rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Inject a fabricated absolute or relative date clause.

    Distinct from fabricate_prior: this one injects a temporal phrase
    without claiming any specific comparison exam exists.  Covers the
    "new nodule since yesterday's film" failure mode (Chen et al. 2025).

    Args:
        claim: Source claim.
        rng:   Optional seeded Random instance for reproducibility.

    Returns:
        Modified Claim with label "Contradicted", or None if the claim
        already has temporal / date language.
    """
    if rng is None:
        rng = random.Random()

    text = claim.text.rstrip()
    if _EXISTING_TEMPORAL_RE.search(text) or _EXISTING_PRIOR_RE.search(text):
        return None
    if not text:
        return None

    phrase = rng.choice(FABRICATED_TEMPORAL_DATES)
    stripped = text.rstrip(".")
    new_text = f"{stripped}, {phrase}."
    return _clone_claim(claim, new_text)


#: Deterministic order in which compound_perturbation attempts to chain
#: single-type perturbations.  The order is chosen so that earlier
#: generators touch more structural properties (laterality, finding) and
#: later generators touch surface properties (fabrications), which
#: maximises the chance that a 3-error compound still produces a
#: clinically distinguishable claim.
_COMPOUND_ORDER: list[str] = [
    "laterality_swap",
    "finding_substitution",
    "severity_swap",
    "temporal_error",
    "negation",
    "region_swap",
    "device_error",
    "fabricate_measurement",
    "fabricate_prior",
    "fabricate_temporal",
]


def _apply_single_generator(
    neg_type: str, claim: Claim, rng: random.Random,
) -> Optional[Claim]:
    """Dispatch a single-type perturbation within compound_perturbation."""
    if neg_type == "laterality_swap":
        return laterality_swap(claim)
    if neg_type == "negation":
        return negation(claim)
    if neg_type == "finding_substitution":
        return finding_substitution(claim, rng=rng)
    if neg_type == "region_swap":
        return region_swap(claim, rng=rng)
    if neg_type == "severity_swap":
        return severity_swap(claim, rng=rng)
    if neg_type == "temporal_error":
        return temporal_error(claim, rng=rng)
    if neg_type == "device_error":
        return device_error(claim, rng=rng)
    if neg_type == "fabricate_measurement":
        return fabricate_measurement(claim, rng=rng)
    if neg_type == "fabricate_prior":
        return fabricate_prior(claim, rng=rng)
    if neg_type == "fabricate_temporal":
        return fabricate_temporal(claim, rng=rng)
    return None


def compound_perturbation(
    claim: Claim,
    rng: Optional[random.Random] = None,
    n_errors: int = 2,
) -> Optional[Claim]:
    """Chain *n_errors* distinct single-type perturbations.

    Picks *n_errors* distinct single-type generators in the fixed order
    defined by ``_COMPOUND_ORDER`` and applies them sequentially.  If
    any step returns None the whole compound is aborted (returns None);
    compounds that successfully mutate the text are returned as a
    Contradicted claim.

    Args:
        claim:    Source claim.
        rng:      Optional seeded Random instance.
        n_errors: Number of single-type perturbations to chain
                  (typically 2 or 3).

    Returns:
        Modified Claim with label "Contradicted", or None if the
        compound cannot be produced cleanly.
    """
    if rng is None:
        rng = random.Random()
    if n_errors < 2:
        return None

    # Randomly pick *n_errors* generators from _COMPOUND_ORDER without
    # replacement, but apply them in the _COMPOUND_ORDER-defined order
    # (not shuffled) so the operation is deterministic given a fixed
    # rng.choices seed.
    selected = set(rng.sample(_COMPOUND_ORDER, k=min(n_errors, len(_COMPOUND_ORDER))))
    ordered = [t for t in _COMPOUND_ORDER if t in selected]
    if len(ordered) < n_errors:
        return None

    current = claim
    applied: list[str] = []
    for neg_type in ordered:
        next_claim = _apply_single_generator(neg_type, current, rng)
        if next_claim is None:
            return None
        if next_claim.text == current.text:
            return None
        current = next_claim
        applied.append(neg_type)

    if not applied:
        return None
    if current.text == claim.text:
        return None
    return current


def compound_perturbation_2err(
    claim: Claim, rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Convenience wrapper — compound_perturbation with n_errors=2."""
    return compound_perturbation(claim, rng=rng, n_errors=2)


def compound_perturbation_3err(
    claim: Claim, rng: Optional[random.Random] = None,
) -> Optional[Claim]:
    """Convenience wrapper — compound_perturbation with n_errors=3."""
    return compound_perturbation(claim, rng=rng, n_errors=3)


# ---------------------------------------------------------------------------
# Dispatcher and batch generator
# ---------------------------------------------------------------------------

#: Mapping from string name to generator function (excluding omission which
#: requires an extra argument).
_GENERATORS: dict[str, Callable] = {
    "laterality_swap": laterality_swap,
    "finding_substitution": finding_substitution,
    "negation": negation,
    "region_swap": region_swap,
    "severity_swap": severity_swap,
    "temporal_error": temporal_error,
    "device_error": device_error,
    "fabricate_measurement": fabricate_measurement,
    "fabricate_prior": fabricate_prior,
    "fabricate_temporal": fabricate_temporal,
    "compound_2err": compound_perturbation_2err,
    "compound_3err": compound_perturbation_3err,
}

_LABELS: dict[str, str] = {
    "laterality_swap": _LABEL_CONTRADICTED,
    "finding_substitution": _LABEL_CONTRADICTED,
    "negation": _LABEL_CONTRADICTED,
    "region_swap": _LABEL_CONTRADICTED,
    "severity_swap": _LABEL_CONTRADICTED,
    "temporal_error": _LABEL_CONTRADICTED,
    "device_error": _LABEL_CONTRADICTED,
    "fabricate_measurement": _LABEL_CONTRADICTED,
    "fabricate_prior": _LABEL_CONTRADICTED,
    "fabricate_temporal": _LABEL_CONTRADICTED,
    "compound_2err": _LABEL_CONTRADICTED,
    "compound_3err": _LABEL_CONTRADICTED,
    "omission": _LABEL_INSUFFICIENT,
}

ALL_NEGATIVE_TYPES: list[str] = list(_GENERATORS.keys()) + ["omission"]

#: Generator keys whose functions accept a keyword `rng` argument.  Used
#: by the `generate_hard_negatives` dispatcher to route rng correctly.
_RNG_AWARE_GENERATORS: frozenset[str] = frozenset({
    "finding_substitution",
    "region_swap",
    "severity_swap",
    "temporal_error",
    "device_error",
    "fabricate_measurement",
    "fabricate_prior",
    "fabricate_temporal",
    "compound_2err",
    "compound_3err",
})


def generate_hard_negatives(
    claims: list[Claim],
    types: list[str],
    n_per_claim: int = 3,
    seed: Optional[int] = None,
) -> list[tuple[Claim, str, str]]:
    """Generate hard negatives for a list of claims.

    For each claim, attempts to produce up to *n_per_claim* distinct hard
    negatives drawn from the requested *types*.  Types that are not applicable
    to a particular claim (e.g., laterality_swap on a claim with no laterality)
    are silently skipped.

    Args:
        claims: Source claims extracted from RadGraph annotations.
        types: Which negative types to apply.  Valid values are the keys in
            ``ALL_NEGATIVE_TYPES``.
        n_per_claim: Maximum number of hard negatives to produce per claim.
        seed: Optional random seed for reproducibility.

    Returns:
        List of ``(modified_claim, negative_type, ground_truth_label)`` tuples.

    Raises:
        ValueError: If any entry in *types* is not a recognised negative type.
    """
    unknown = set(types) - set(ALL_NEGATIVE_TYPES)
    if unknown:
        raise ValueError(f"Unknown negative types: {unknown}. Valid: {ALL_NEGATIVE_TYPES}")

    rng = random.Random(seed)
    results: list[tuple[Claim, str, str]] = []

    for claim in claims:
        # Shuffle types per claim to get diverse coverage
        shuffled_types = list(types)
        rng.shuffle(shuffled_types)

        produced = 0
        seen_texts: set[str] = {claim.text.lower()}  # avoid duplicates

        for neg_type in shuffled_types:
            if produced >= n_per_claim:
                break

            modified: Optional[Claim] = None

            if neg_type == "omission":
                modified = omission(claim, claims, rng=rng)
            elif neg_type in _GENERATORS:
                fn = _GENERATORS[neg_type]
                # Functions that accept rng (v3 taxonomy includes fabrication
                # + compound generators, all of which are rng-aware).
                if neg_type in _RNG_AWARE_GENERATORS:
                    modified = fn(claim, rng=rng)
                else:
                    modified = fn(claim)
            else:
                logger.warning("Unhandled negative type: %s", neg_type)
                continue

            if modified is None:
                continue
            if modified.text.lower() in seen_texts:
                continue  # skip exact duplicate texts

            seen_texts.add(modified.text.lower())
            label = _LABELS[neg_type]
            results.append((modified, neg_type, label))
            produced += 1

    logger.info(
        "Generated %d hard negatives from %d claims using types %s",
        len(results),
        len(claims),
        types,
    )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_region(claim: Claim) -> str:
    """Map a claim's anatomy / pathology category to a FINDING_SUBSTITUTION_MAP
    region key.

    Args:
        claim: A Claim object.

    Returns:
        Region key string (defaults to "lung" when no better match is found).
    """
    anatomy_lower = (claim.anatomy or "").lower()
    category = claim.pathology_category

    if any(kw in anatomy_lower for kw in ["pleura", "pleural", "costophrenic"]):
        return "pleura"
    if any(kw in anatomy_lower for kw in ["mediastin", "paratracheal"]):
        return "mediastinum"
    if any(kw in anatomy_lower for kw in ["cardiac", "heart", "pericardial", "pericardium"]):
        return "heart"
    if any(kw in anatomy_lower for kw in ["rib", "clavicle", "vertebra", "bone", "scapula"]):
        return "bone"
    if any(kw in anatomy_lower for kw in ["hilum", "hilar"]):
        return "hilum"
    if any(kw in anatomy_lower for kw in ["diaphragm", "hemidiaphragm", "subphrenic"]):
        return "diaphragm"
    if any(kw in anatomy_lower for kw in ["trachea", "bronch", "airway"]):
        return "airway"

    # Fall back to category
    if category == "Pleural Effusion":
        return "pleura"
    if category in ("Enlarged Cardiomediastinum",):
        return "mediastinum"
    if category == "Cardiomegaly":
        return "heart"
    if category == "Fracture":
        return "bone"

    return "lung"


def _category_for_finding(finding_text: str) -> str:
    """Map a free-text finding to a CheXpert ontology category.

    Args:
        finding_text: Text description of the finding.

    Returns:
        CheXpert category string.
    """
    from ..preprocessing.radgraph_parser import map_to_chexpert_ontology
    return map_to_chexpert_ontology(finding_text)
