"""Clinical knowledge base for hard negative construction.

Encodes which findings are laterality-sensitive, which CheXpert labels
have known noise, difficulty tiers for hard negatives, and clinical
severity rankings. This prevents generating clinically nonsensical
negatives that reviewers with radiology knowledge would catch.
"""

from __future__ import annotations

# ============================================================
# LATERALITY SENSITIVITY
# ============================================================
# Which findings are laterality-sensitive (left/right matters)?
# Swapping laterality on non-sensitive findings creates obvious
# nonsense, not a hard negative.

LATERALITY_SENSITIVE = {
    # These findings are commonly lateralized — swapping creates real confounders
    "pneumothorax",
    "pleural effusion",
    "consolidation",
    "atelectasis",
    "pneumonia",
    "lung opacity",
    "pulmonary nodule",
    "pulmonary mass",
    "rib fracture",
    "pleural thickening",
    "lung lesion",
}

LATERALITY_INSENSITIVE = {
    # These are midline or bilateral — laterality swap is nonsensical
    "cardiomegaly",
    "cardiac enlargement",
    "pericardial effusion",
    "mediastinal widening",
    "enlarged cardiomediastinum",
    "pulmonary edema",  # typically bilateral
    "hilar enlargement",
    "tracheal deviation",  # midline structure
    "scoliosis",
}

# ============================================================
# CHEXPERT LABEL RELIABILITY
# ============================================================
# Known noise rates for CheXpert/CheXbert labels.
# Used to weight confidence in "Insufficient Evidence" labels —
# if CheXbert is unreliable for a finding, absence of label is
# weak evidence of absence.

CHEXPERT_LABEL_NOISE = {
    # Finding: estimated false negative rate (CheXbert misses it)
    "Atelectasis": 0.18,         # commonly overcalled or missed
    "Consolidation": 0.15,       # confused with atelectasis
    "Pleural Effusion": 0.08,    # relatively reliable
    "Pneumothorax": 0.12,        # small PTX often missed
    "Cardiomegaly": 0.06,        # reliable
    "Edema": 0.10,               # moderate reliability
    "Pneumonia": 0.20,           # poor — often conflated with consolidation
    "Lung Opacity": 0.12,        # broad category, moderate noise
    "Lung Lesion": 0.15,         # often missed if subtle
    "Fracture": 0.25,            # often missed by automated labelers
    "Support Devices": 0.05,     # very reliable
    "Enlarged Cardiomediastinum": 0.15,
    "Pleural Other": 0.20,
    "No Finding": 0.10,          # sometimes assigned when subtle findings exist
}

# ============================================================
# HARD NEGATIVE DIFFICULTY TIERS
# ============================================================
# Not all negatives should be equally hard. Easy negatives help
# the model learn basic distinctions; hard negatives push it on
# clinically dangerous confounders.
#
# Curriculum: start training with mostly easy+medium, gradually
# increase hard proportion.

DIFFICULTY_TIERS = {
    "easy": {
        # Obvious changes that even a rule-based system would catch
        "laterality_swap": 0.3,       # only on laterality-sensitive findings
        "negation": 0.4,              # flip present/absent
        "finding_substitution": 0.3,  # swap with unrelated finding
    },
    "medium": {
        # Requires understanding anatomy and clinical context
        "region_swap": 0.3,           # wrong anatomy, same finding
        "severity_swap": 0.3,         # mild vs severe
        "device_error": 0.2,          # wrong device type
        "temporal_error": 0.2,        # new vs unchanged
    },
    "hard": {
        # Subtle errors that require deep clinical knowledge
        "finding_substitution_same_region": 0.3,  # swap with similar finding
        "partial_negation": 0.2,       # "no large" -> "small" (subtle change)
        "laterality_swap_bilateral": 0.2,  # bilateral -> unilateral
        "severity_downgrade": 0.3,     # "tension pneumothorax" -> "small pneumothorax"
    },
}

# ============================================================
# CLINICAL SEVERITY RANKING
# ============================================================
# Findings ranked by clinical urgency. Used to weight verification
# errors — missing a tension pneumothorax is far worse than
# missing mild atelectasis.

CLINICAL_SEVERITY = {
    # 1 = life-threatening, 5 = incidental
    "tension pneumothorax": 1,
    "pneumothorax": 2,
    "pneumomediastinum": 2,
    "aortic dissection": 1,
    "pulmonary embolism": 1,
    "large pleural effusion": 2,
    "pulmonary edema": 2,
    "pneumonia": 3,
    "consolidation": 3,
    "cardiomegaly": 3,
    "pleural effusion": 3,
    "atelectasis": 4,
    "lung opacity": 4,
    "lung nodule": 4,
    "fracture": 3,
    "mild atelectasis": 5,
    "surgical changes": 5,
    "degenerative changes": 5,
}

# ============================================================
# CONFUSABLE FINDING PAIRS
# ============================================================
# Findings that are commonly confused with each other in real
# radiology errors. These make the BEST hard negatives because
# they represent actual clinical failure modes.

CONFUSABLE_PAIRS = [
    ("consolidation", "atelectasis"),       # most common confusion
    ("pneumonia", "consolidation"),         # overlapping definitions
    ("pulmonary edema", "pneumonia"),       # similar appearance
    ("pleural effusion", "atelectasis"),    # both cause opacity at base
    ("lung mass", "consolidation"),         # round pneumonia vs mass
    ("pneumothorax", "skin fold"),          # classic mimic on CXR
    ("cardiomegaly", "pericardial effusion"),  # both enlarge cardiac silhouette
    ("hilar enlargement", "mediastinal widening"),
    ("rib fracture", "bone island"),        # subtle distinction
    ("pulmonary nodule", "nipple shadow"),  # classic false positive
]


def should_swap_laterality(finding: str) -> bool:
    """Check if laterality swap makes clinical sense for this finding.

    Args:
        finding: The finding text (lowercase).

    Returns:
        True if laterality swap creates a realistic confounder.
    """
    finding_lower = finding.lower()
    # Check against sensitive list
    for sensitive in LATERALITY_SENSITIVE:
        if sensitive in finding_lower:
            return True
    # Check against insensitive list (explicit reject)
    for insensitive in LATERALITY_INSENSITIVE:
        if insensitive in finding_lower:
            return False
    # Default: allow swap if finding mentions anatomy
    return any(w in finding_lower for w in ["left", "right", "lobe", "lung"])


def get_confusable_finding(finding: str) -> str | None:
    """Get a clinically confusable finding for hard negative construction.

    Returns a finding that is commonly confused with the input in
    real radiology practice, rather than a random substitution.

    Args:
        finding: The original finding text.

    Returns:
        A confusable finding string, or None if no match.
    """
    finding_lower = finding.lower()
    for a, b in CONFUSABLE_PAIRS:
        if a in finding_lower:
            return b
        if b in finding_lower:
            return a
    return None


def get_label_confidence(pathology: str) -> float:
    """Get confidence weight for a CheXpert label.

    Lower confidence means the label is more likely to be wrong,
    so we should down-weight "Insufficient Evidence" examples
    derived from absence of this label.

    Args:
        pathology: CheXpert pathology name.

    Returns:
        Confidence in [0, 1]. 1.0 = very reliable label.
    """
    noise = CHEXPERT_LABEL_NOISE.get(pathology, 0.15)
    return 1.0 - noise
