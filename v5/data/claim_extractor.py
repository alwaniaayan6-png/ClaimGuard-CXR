"""Claim extractor: radiology report → list of atomic sentence-level claims.

Primary implementation: GPT-4o-mini with a locked prompt.
Fallback: deterministic sentence splitter (used in CI/tests).

Both return a list of (claim_text, section, sentence_index) triples plus a
`claim_id` that is deterministic over (report_id, sentence_index, claim_text).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v5.0-claim-extract"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+")
_MIN_CLAIM_LEN = 5  # chars


@dataclass
class ExtractedClaim:
    claim_id: str
    claim_text: str
    report_id: str
    section: str  # findings | impression | technique | other
    sentence_index: int
    extractor_version: str = PROMPT_VERSION


def _deterministic_claim_id(report_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{report_id}::{idx}::{text}".encode()).hexdigest()
    return h[:16]


def rule_extract(report_text: str, report_id: str, section: str = "report") -> list[ExtractedClaim]:
    """Deterministic sentence splitter. No LLM required."""
    out: list[ExtractedClaim] = []
    # Split into sentences
    sentences = _SENTENCE_SPLIT_RE.split(report_text.strip()) if report_text else []
    for i, s in enumerate(sentences):
        s = s.strip()
        if len(s) < _MIN_CLAIM_LEN:
            continue
        # Drop section headers
        if s.isupper() and s.endswith(":"):
            continue
        out.append(
            ExtractedClaim(
                claim_id=_deterministic_claim_id(report_id, i, s),
                claim_text=s,
                report_id=report_id,
                section=section,
                sentence_index=i,
                extractor_version=PROMPT_VERSION + "-rule",
            )
        )
    return out


_EXTRACT_SYSTEM = """You are a radiology claim extractor. Given a chest X-ray \
radiology report, break it into atomic sentence-level claims. Each claim should \
assert ONE finding or observation. Merge or split the original sentences as \
needed. Drop section headers. Return JSON: {"claims": [{"text": "...", \
"section": "findings|impression|technique|other"}]}"""


def llm_extract(
    report_text: str,
    report_id: str,
    *,
    provider: Literal["openai", "anthropic"] = "openai",
    model: str | None = None,
) -> list[ExtractedClaim]:
    """LLM extraction. Falls back to rule_extract on failure."""
    raw: str | None = None
    try:
        if provider == "openai":
            from openai import OpenAI  # type: ignore[import]

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": report_text},
                ],
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content
        else:
            import anthropic  # type: ignore[import]

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model or "claude-sonnet-4-5",
                max_tokens=1200,
                system=_EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": report_text}],
            )
            raw = resp.content[0].text
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM extractor failed (%s); falling back to rule", exc)

    if raw is None:
        return rule_extract(report_text, report_id)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {"claims": []}

    out: list[ExtractedClaim] = []
    for i, c in enumerate(parsed.get("claims", [])):
        text = str(c.get("text", "")).strip()
        if len(text) < _MIN_CLAIM_LEN:
            continue
        out.append(
            ExtractedClaim(
                claim_id=_deterministic_claim_id(report_id, i, text),
                claim_text=text,
                report_id=report_id,
                section=str(c.get("section", "other")),
                sentence_index=i,
                extractor_version=PROMPT_VERSION + "-llm",
            )
        )
    if not out:
        return rule_extract(report_text, report_id)
    return out
