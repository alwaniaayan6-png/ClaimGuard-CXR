"""LLM-backed claim extractor.

Input: free-text radiology report.
Output: a list of structured ``Claim`` records.

The extractor is intentionally backend-agnostic. Callers pass an
``LLMBackend`` instance that knows how to turn a prompt into JSON.
That lets us swap GPT-4o / Claude / Llama-3.1-405B / local-Qwen without
touching the parser, and lets tests inject a deterministic stub.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from claimguard_nmi.grounding.claim_schema import (
    Claim,
    ClaimCertainty,
    ClaimType,
    Laterality,
    Region,
)


CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
You are a radiology claim extractor. Given a chest radiology report, return a
JSON array of atomic claims. Each claim is one factual assertion.

Each claim must have these fields:
  raw_text       string   - the original sentence or fragment
  finding        string   - short surface form of the pathology/device
                             ("pneumothorax", "left pleural effusion", ...)
  claim_type     string   - one of: finding, negation, prior_comparison,
                             measurement, implicit_global_negation, hedge,
                             global_description, device
  certainty      string   - one of: present, absent, possible, uncertain
  laterality     string   - one of: left, right, bilateral, unspecified
  region         string   - one of: upper, middle, lower, apical, basal,
                             retrocardiac, perihilar, costophrenic,
                             mediastinal, unspecified
  severity       string|null - optional, e.g. "small", "trace", "large"
  size_mm        float|null  - optional, only for explicit measurements

Return ONLY the JSON array, no prose.
"""


class LLMBackend(ABC):
    """Contract for a pluggable LLM backend used for claim extraction."""

    @abstractmethod
    def generate_json(self, system: str, user: str) -> str:
        """Return raw JSON string output (may include fences; parser strips them)."""


@dataclass
class ExtractionResult:
    """Result of extracting claims from one report.

    Attributes
    ----------
    claims : list[Claim]
        Successfully-parsed structured claims.
    n_skipped : int
        Records the extractor returned but we could not convert to a Claim
        (missing required fields, type errors). A high count is a quality
        signal — not zero ≠ broken, but >10% usually is.
    parse_failed : bool
        True iff the LLM output was not parseable as JSON at all.
    raw_response : str
        The raw LLM output (kept for debugging; do not log at INFO).
    """
    claims: List[Claim]
    n_skipped: int = 0
    parse_failed: bool = False
    raw_response: str = ""


@dataclass
class ClaimExtractor:
    """Extract structured Claim records from a radiology report."""
    backend: LLMBackend
    source_model_id: Optional[str] = None
    source_report_id: Optional[str] = None

    def extract(self, report_text: str) -> List[Claim]:
        """Back-compatible convenience wrapper that returns just the claims list.

        For production use prefer ``extract_with_diagnostics`` so the caller
        can detect silent parse failures (v2 review caught the
        double-silent-skip bug).
        """
        return self.extract_with_diagnostics(report_text).claims

    def extract_with_diagnostics(self, report_text: str) -> ExtractionResult:
        raw = self.backend.generate_json(
            system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            user=f"Report:\n{report_text}\n\nReturn JSON array.",
        )
        records, parse_failed = _parse_json_array_with_diag(raw)
        claims: List[Claim] = []
        n_skipped = 0
        for rec in records:
            try:
                claims.append(self._to_claim(rec))
            except Exception:
                n_skipped += 1
        return ExtractionResult(
            claims=claims,
            n_skipped=n_skipped,
            parse_failed=parse_failed,
            raw_response=raw,
        )

    def _to_claim(self, rec: dict) -> Claim:
        return Claim(
            raw_text=str(rec.get("raw_text", "")),
            finding=str(rec.get("finding", "")).strip().lower(),
            claim_type=_parse_enum(rec.get("claim_type"), ClaimType, ClaimType.FINDING),
            certainty=_parse_enum(rec.get("certainty"), ClaimCertainty, ClaimCertainty.PRESENT),
            laterality=_parse_enum(rec.get("laterality"), Laterality, Laterality.UNSPECIFIED),
            region=_parse_enum(rec.get("region"), Region, Region.UNSPECIFIED),
            severity=rec.get("severity"),
            size_mm=_parse_float_or_none(rec.get("size_mm")),
            source_report_id=self.source_report_id,
            source_model_id=self.source_model_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_array_with_diag(raw: str):
    """Return (records, parse_failed) so callers can distinguish
    'extractor returned no claims' from 'JSON parsing failed entirely'."""
    import json
    import re

    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if not s:
        return [], True
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return [], True
    if isinstance(obj, list):
        return obj, False
    if isinstance(obj, dict) and "claims" in obj and isinstance(obj["claims"], list):
        return obj["claims"], False
    return [], False


def _parse_json_array(raw: str) -> list:
    """Back-compat wrapper returning just the records list."""
    records, _ = _parse_json_array_with_diag(raw)
    return records


def _parse_enum(value, enum_cls, default):
    if value is None:
        return default
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return default


def _parse_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Stub backend for tests and development.
# ---------------------------------------------------------------------------
@dataclass
class StubBackend(LLMBackend):
    """Returns a fixed canned response. Useful for unit tests."""
    canned: str

    def generate_json(self, system: str, user: str) -> str:  # noqa: ARG002
        return self.canned
