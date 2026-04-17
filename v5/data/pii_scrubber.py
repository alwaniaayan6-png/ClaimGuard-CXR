"""PII scrubber for text artifacts released in ClaimGuard-GroundBench.

Two-stage pipeline:
1. Presidio analyzer + anonymizer with medical recognizers.
2. Custom CXR-specific regex for dates, accession numbers, institution names,
   honorifics, and common identifiers that Presidio misses.

Manual audit of ~500 samples expected before release; false-negative rate
documented in the data card.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PHI_REGEXES: list[tuple[re.Pattern[str], str]] = [
    # Dates in common radiology formats
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"), "[DATE]"),
    (re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b", re.IGNORECASE), "[DATE]"),
    # Accession / MRN / study numbers
    (re.compile(r"\bmrn[:\s#]*\d{4,}\b", re.IGNORECASE), "[MRN]"),
    (re.compile(r"\bacc(?:ession)?[:\s#]*\d{4,}\b", re.IGNORECASE), "[ACCESSION]"),
    (re.compile(r"\bstudy\s*#?\s*\d{4,}\b", re.IGNORECASE), "[STUDY_ID]"),
    # Honorifics + name patterns
    (re.compile(r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), "[PROVIDER]"),
    (re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), "[PERSON]"),
    # Phone
    (re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Known institution names in our datasets (not exhaustive)
    (re.compile(r"\b(?:Stanford|Beth\s+Israel|Indiana\s+University|BIMCV|Fleury|Valencia|ValencIA)\b", re.IGNORECASE), "[INSTITUTION]"),
    # Times that look like appointment slots
    (re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b"), "[TIME]"),
    # Room / bed numbers
    (re.compile(r"\b(?:room|rm|bed)[:\s#]*\d{1,4}[A-Z]?\b", re.IGNORECASE), "[LOCATION]"),
]


@dataclass
class ScrubResult:
    text: str
    n_redactions: int
    via_presidio: int = 0
    via_regex: int = 0


class PIIScrubber:
    """Lazy loader for Presidio (heavyweight) with regex fallback."""

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None

    def _lazy_init(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Presidio initialized")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Presidio unavailable (%s); using regex-only scrub", exc)

    def scrub(self, text: str) -> ScrubResult:
        self._lazy_init()
        presidio_redactions = 0
        out = text
        if self._analyzer is not None and self._anonymizer is not None:
            try:
                results = self._analyzer.analyze(
                    text=text,
                    language="en",
                    entities=[
                        "PERSON",
                        "PHONE_NUMBER",
                        "EMAIL_ADDRESS",
                        "DATE_TIME",
                        "LOCATION",
                        "MEDICAL_LICENSE",
                        "US_SSN",
                    ],
                )
                presidio_redactions = len(results)
                anon = self._anonymizer.anonymize(text=text, analyzer_results=results)
                out = anon.text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Presidio scrub failed (%s); regex fallback", exc)
                out = text
        regex_redactions = 0
        for pat, repl in _PHI_REGEXES:
            out, n = pat.subn(repl, out)
            regex_redactions += n
        return ScrubResult(
            text=out,
            n_redactions=presidio_redactions + regex_redactions,
            via_presidio=presidio_redactions,
            via_regex=regex_redactions,
        )
