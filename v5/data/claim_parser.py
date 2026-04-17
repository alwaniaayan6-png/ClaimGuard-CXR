"""Claim parser: free-text claim → structured (finding, location, laterality, severity, ...).

Primary implementation uses an LLM (Claude/GPT-4o) with a locked prompt. Fallback
uses a rule-based parser for offline/CI usage (deterministic, no API). Every claim
that passes through v5 is parsed into the schema from ARCHITECTURE_V5_IMAGE_GROUNDED.md
Appendix C.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v5.0-claim-parser"
ONTOLOGY_PATH = Path(__file__).parent.parent / "annotations" / "finding_ontology_v5.yaml"


Laterality = Literal["L", "R", "bilateral", "none", "unknown"]
Severity = Literal["mild", "moderate", "severe", "unknown"]
Temporality = Literal["new", "chronic", "improving", "worsening", "stable", "unknown"]
Comparison = Literal["present", "absent", "unknown"]


@dataclass
class StructuredClaim:
    claim_id: str
    raw_text: str
    report_id: str
    finding: str = "unknown"
    finding_family: str = "unknown"
    location: str = "unknown"
    laterality: Laterality = "unknown"
    severity: Severity = "unknown"
    temporality: Temporality = "unknown"
    comparison: Comparison = "unknown"
    modifier_tags: list[str] = field(default_factory=list)
    evidence_source_type: str = "unknown"
    generator_id: str = "unknown"
    generator_temperature: float | None = None
    generator_seed: int | None = None
    parser_confidence: float = 0.0
    parser_version: str = PROMPT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


_LATERALITY_MAP = {
    "left": "L",
    "right": "R",
    "bilateral": "bilateral",
    "both": "bilateral",
    "bilaterally": "bilateral",
}

_SEVERITY_MAP = {
    "small": "mild",
    "trace": "mild",
    "minimal": "mild",
    "mild": "mild",
    "moderate": "moderate",
    "large": "severe",
    "massive": "severe",
    "severe": "severe",
    "extensive": "severe",
}

_TEMPORAL_MAP = {
    "new": "new",
    "acute": "new",
    "chronic": "chronic",
    "old": "chronic",
    "improved": "improving",
    "improving": "improving",
    "resolving": "improving",
    "worsened": "worsening",
    "worsening": "worsening",
    "progressing": "worsening",
    "progressed": "worsening",
    "unchanged": "stable",
    "stable": "stable",
    "similar": "stable",
}

_COMPARISON_PATTERNS = [
    r"\bcompared (?:to|with)\b",
    r"\bsince (?:the )?prior\b",
    r"\bversus\s+prior\b",
    r"\bprior (?:exam|study|film|radiograph)\b",
    r"\bfrom (?:last|the previous)\b",
]

_NEG_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bruled? out\b",
    r"\babsen(?:t|ce)\b",
    r"\bfree of\b",
    r"\bclear of\b",
]


def load_ontology() -> dict:
    with ONTOLOGY_PATH.open() as f:
        return yaml.safe_load(f)


def _finding_family(finding: str, ontology: dict) -> str:
    for family, members in ontology.get("finding_family_groups", {}).items():
        if finding in members:
            return family
    return "unknown"


def _rule_find_finding(text: str, ontology: dict) -> str:
    """Scan for finding keywords in unified vocabulary."""
    t = text.lower()
    # Preference: explicit no_finding indicators
    if re.search(r"\b(?:no acute|unremarkable|normal|clear lungs?|no finding)\b", t):
        return "no_finding"
    # Match longest keyword first for specificity
    candidates = []
    for unified_class in ontology.get("unified_classes", []):
        if unified_class == "no_finding":
            continue
        phrase = unified_class.replace("_", " ")
        if phrase in t:
            candidates.append((len(phrase), unified_class))
    # a few hand-tuned aliases
    aliases = {
        "opacity": "lung_opacity",
        "opacities": "lung_opacity",
        "infiltrate": "infiltration",
        "infiltrates": "infiltration",
        "nodule": "lung_lesion",
        "nodules": "lung_lesion",
        "effusion": "pleural_effusion",
        "effusions": "pleural_effusion",
        "pneumo": "pneumothorax",
        "ptx": "pneumothorax",
        "cardiomediastinal": "enlarged_cardiomediastinum",
        "heart size": "cardiomegaly",
        "heart enlarged": "cardiomegaly",
        "heart is enlarged": "cardiomegaly",
        "et tube": "support_device",
        "endotracheal": "support_device",
        "picc": "support_device",
        "central line": "support_device",
        "pacemaker": "support_device",
        "ng tube": "support_device",
        "foreign body": "foreign_object",
    }
    for alias, mapped in aliases.items():
        if alias in t:
            candidates.append((len(alias), mapped))
    if not candidates:
        return "unknown"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _rule_find_laterality(text: str) -> Laterality:
    t = text.lower()
    for keyword, lat in _LATERALITY_MAP.items():
        if re.search(rf"\b{keyword}\b", t):
            return lat  # type: ignore[return-value]
    return "none"


def _rule_find_severity(text: str) -> Severity:
    t = text.lower()
    for keyword, sev in _SEVERITY_MAP.items():
        if re.search(rf"\b{keyword}\b", t):
            return sev  # type: ignore[return-value]
    return "unknown"


def _rule_find_temporality(text: str) -> Temporality:
    t = text.lower()
    for keyword, temp in _TEMPORAL_MAP.items():
        if re.search(rf"\b{keyword}\b", t):
            return temp  # type: ignore[return-value]
    return "unknown"


def _rule_find_comparison(text: str) -> Comparison:
    for pat in _COMPARISON_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "present"
    return "absent"


def _rule_find_location(text: str) -> str:
    """Extract coarse anatomical location. Returns the most specific match."""
    t = text.lower()
    patterns = [
        ("left_upper_lobe", r"left upper( lobe)?"),
        ("left_lower_lobe", r"left lower( lobe)?"),
        ("left_mid_lung", r"left (mid|middle|midzone|mid zone|mid-zone)"),
        ("right_upper_lobe", r"right upper( lobe)?"),
        ("right_middle_lobe", r"right middle( lobe)?"),
        ("right_lower_lobe", r"right lower( lobe)?"),
        ("right_mid_lung", r"right (mid|midzone|mid zone|mid-zone)"),
        ("lung_base_left", r"left (lung )?base"),
        ("lung_base_right", r"right (lung )?base"),
        ("apex_left", r"left (lung )?apex"),
        ("apex_right", r"right (lung )?apex"),
        ("lingula", r"\blingula\b"),
        ("mediastinum", r"\bmediastin"),
        ("hilum_left", r"left hil"),
        ("hilum_right", r"right hil"),
        ("heart", r"\b(cardiac silhouette|heart)\b"),
        ("diaphragm", r"\bdiaphragm"),
        ("pleural_space", r"\bpleural\b"),
    ]
    best = None
    for loc_id, pat in patterns:
        m = re.search(pat, t)
        if m:
            # prefer longer actual matched text (more specific)
            span_len = m.end() - m.start()
            if best is None or span_len > best[1]:
                best = (loc_id, span_len)
    return best[0] if best else "unspecified"


def _rule_find_modifiers(text: str) -> list[str]:
    tags: list[str] = []
    t = text.lower()
    if re.search(r"\bsmall\b|\btrace\b|\bminimal\b", t):
        tags.append("size_small")
    if re.search(r"\blarge\b|\bmassive\b|\bextensive\b", t):
        tags.append("size_large")
    if re.search(r"\b\d+\s?mm\b", t):
        tags.append("size_measured_mm")
    if re.search(r"\b\d+\s?cm\b", t):
        tags.append("size_measured_cm")
    if re.search(r"\bnew\b", t):
        tags.append("temporality_new")
    for pat in _NEG_PATTERNS:
        if re.search(pat, t):
            tags.append("polarity_negated")
            break
    return tags


def rule_parse(
    raw_text: str,
    claim_id: str,
    report_id: str,
    *,
    ontology: dict | None = None,
    evidence_source_type: str = "unknown",
    generator_id: str = "unknown",
    generator_temperature: float | None = None,
    generator_seed: int | None = None,
) -> StructuredClaim:
    """Deterministic rule-based parser. Used as fallback and for CI tests."""
    ontology = ontology or load_ontology()
    finding = _rule_find_finding(raw_text, ontology)
    is_negated = any(re.search(pat, raw_text, re.IGNORECASE) for pat in _NEG_PATTERNS)
    if is_negated and finding != "no_finding":
        finding = "no_finding" if finding == "unknown" else finding
    claim = StructuredClaim(
        claim_id=claim_id,
        raw_text=raw_text,
        report_id=report_id,
        finding=finding,
        finding_family=_finding_family(finding, ontology),
        location=_rule_find_location(raw_text),
        laterality=_rule_find_laterality(raw_text),
        severity=_rule_find_severity(raw_text),
        temporality=_rule_find_temporality(raw_text),
        comparison=_rule_find_comparison(raw_text),
        modifier_tags=_rule_find_modifiers(raw_text),
        evidence_source_type=evidence_source_type,
        generator_id=generator_id,
        generator_temperature=generator_temperature,
        generator_seed=generator_seed,
        parser_confidence=0.6,  # rule parser is adequate but not as strong as LLM
        parser_version=PROMPT_VERSION + "-rule",
    )
    return claim


# ---------------------------------------------------------------------------
# LLM-based parser (primary). Kept separate to avoid importing network libs in
# CI/test environments unless actually used.
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a radiology claim structuring tool. Given a sentence \
extracted from a chest X-ray report, you return a strict JSON object with these fields: \
finding (unified 23-class vocabulary), location (anatomical region id), laterality \
(L | R | bilateral | none | unknown), severity (mild | moderate | severe | unknown), \
temporality (new | chronic | improving | worsening | stable | unknown), comparison \
(present | absent | unknown), modifier_tags (list of strings), parser_confidence \
(0..1). Respond ONLY with a JSON object. No prose."""


def llm_parse(
    raw_text: str,
    claim_id: str,
    report_id: str,
    *,
    provider: Literal["anthropic", "openai"] = "anthropic",
    model: str | None = None,
    ontology: dict | None = None,
    evidence_source_type: str = "unknown",
    generator_id: str = "unknown",
    generator_temperature: float | None = None,
    generator_seed: int | None = None,
) -> StructuredClaim:
    """LLM-based parser. Falls back to rule_parse on any API failure."""
    ontology = ontology or load_ontology()
    prompt_user = (
        f"Sentence: {raw_text!r}\n"
        f"Unified finding vocabulary: {', '.join(ontology['unified_classes'])}\n"
        "Return JSON only."
    )
    raw_json: str | None = None
    try:
        if provider == "anthropic":
            import anthropic  # type: ignore[import]

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model or "claude-sonnet-4-5",
                max_tokens=400,
                system=_LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_user}],
            )
            raw_json = resp.content[0].text
        elif provider == "openai":
            from openai import OpenAI  # type: ignore[import]

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_user},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            raw_json = resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM parser failed (%s); falling back to rule parser", exc)

    if raw_json is None:
        return rule_parse(
            raw_text,
            claim_id,
            report_id,
            ontology=ontology,
            evidence_source_type=evidence_source_type,
            generator_id=generator_id,
            generator_temperature=generator_temperature,
            generator_seed=generator_seed,
        )

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; falling back")
        return rule_parse(
            raw_text,
            claim_id,
            report_id,
            ontology=ontology,
            evidence_source_type=evidence_source_type,
            generator_id=generator_id,
            generator_temperature=generator_temperature,
            generator_seed=generator_seed,
        )

    finding = parsed.get("finding", "unknown")
    return StructuredClaim(
        claim_id=claim_id,
        raw_text=raw_text,
        report_id=report_id,
        finding=finding,
        finding_family=_finding_family(finding, ontology),
        location=parsed.get("location", "unspecified"),
        laterality=parsed.get("laterality", "unknown"),
        severity=parsed.get("severity", "unknown"),
        temporality=parsed.get("temporality", "unknown"),
        comparison=parsed.get("comparison", "unknown"),
        modifier_tags=parsed.get("modifier_tags", []),
        evidence_source_type=evidence_source_type,
        generator_id=generator_id,
        generator_temperature=generator_temperature,
        generator_seed=generator_seed,
        parser_confidence=float(parsed.get("parser_confidence", 0.8)),
        parser_version=PROMPT_VERSION + "-llm",
    )
