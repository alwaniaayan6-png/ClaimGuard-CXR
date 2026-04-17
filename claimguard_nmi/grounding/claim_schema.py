"""Structured claim representation for image-grounded verification.

A claim is a single factual assertion extracted from a radiology report.
Every claim carries structured fields that let the grounding module
locate the claim's referent in radiologist-drawn image annotations.

See ARCHITECTURE_PATH_B.md Section 5 for the grounding algorithm that
consumes these structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimCertainty(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"
    UNCERTAIN = "uncertain"


class ClaimType(str, Enum):
    FINDING = "finding"                        # "there is a left pleural effusion"
    NEGATION = "negation"                      # "no pneumothorax"
    PRIOR_COMPARISON = "prior_comparison"      # "increased since prior"
    MEASUREMENT = "measurement"                # "3 mm nodule"
    IMPLICIT_GLOBAL_NEGATION = "implicit_global_negation"  # "lungs are clear"
    HEDGE = "hedge"                            # "possible atelectasis"
    GLOBAL_DESCRIPTION = "global_description"  # "mediastinal contours normal"
    DEVICE = "device"                          # "ET tube in good position"


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    UNSPECIFIED = "unspecified"


class Region(str, Enum):
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"
    APICAL = "apical"
    BASAL = "basal"
    RETROCARDIAC = "retrocardiac"
    PERIHILAR = "perihilar"
    COSTOPHRENIC = "costophrenic"
    MEDIASTINAL = "mediastinal"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class Claim:
    """Structured radiology claim suitable for image grounding.

    Attributes
    ----------
    raw_text : str
        Original surface-form sentence (or sentence fragment) as extracted.
    finding : str
        Canonicalized finding id (e.g., 'lung_opacity'). If the claim does
        not map to any canonical id, downstream code sets this to the
        original surface form and grounding returns UNGROUNDED.
    claim_type : ClaimType
        Category gating whether this claim is groundable.
    certainty : ClaimCertainty
        Whether the claim asserts presence, absence, or something weaker.
    laterality : Laterality
        Left / right / bilateral / unspecified.
    region : Region
        Anatomical region in which the finding is asserted.
    severity : str | None
        Optional severity descriptor ("small", "large", "trace", ...).
    size_mm : float | None
        Optional numeric size if the claim is a measurement.
    source_report_id : str | None
        Provenance — which generated report this claim came from.
    source_model_id : str | None
        Which VLM generated the report (for provenance gate integration).
    """

    raw_text: str
    finding: str
    claim_type: ClaimType = ClaimType.FINDING
    certainty: ClaimCertainty = ClaimCertainty.PRESENT
    laterality: Laterality = Laterality.UNSPECIFIED
    region: Region = Region.UNSPECIFIED
    severity: Optional[str] = None
    size_mm: Optional[float] = None
    source_report_id: Optional[str] = None
    source_model_id: Optional[str] = None

    def is_groundable(self) -> bool:
        """True iff this claim type can be checked against a still-image annotation."""
        ungroundable = {
            ClaimType.PRIOR_COMPARISON,
            ClaimType.MEASUREMENT,
            ClaimType.IMPLICIT_GLOBAL_NEGATION,
            ClaimType.HEDGE,
            ClaimType.GLOBAL_DESCRIPTION,
        }
        return self.claim_type not in ungroundable
