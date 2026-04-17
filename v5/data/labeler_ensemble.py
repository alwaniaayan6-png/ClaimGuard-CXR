"""Silver-standard LLM-ensemble labeler for (image, claim) pairs.

Three labelers: GPT-4o vision, Claude Sonnet 4.5 vision, Llama-3.1-405B (text-only).
Two-of-three majority with UNCERTAIN on ties. Validated against radiologist-
drawn annotations (MS-CXR, RSNA, SIIM, ChestX-Det10) before being trusted for
silver training data. Only categories with κ ≥ 0.70 vs radiologist annotations
are shipped as silver training signal.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Label = Literal["SUPPORTED", "CONTRADICTED", "NOVEL_PLAUSIBLE", "NOVEL_HALLUCINATED", "UNCERTAIN"]
VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "NOVEL_PLAUSIBLE", "NOVEL_HALLUCINATED", "UNCERTAIN"}


@dataclass
class LabelerOutput:
    grader_id: str
    grader_version: str
    label: str
    confidence: float
    rationale: str
    prompt_version: str


@dataclass
class EnsembleLabel:
    image_id: str
    claim_id: str
    majority_label: str
    min_confidence: float
    per_labeler: list[LabelerOutput]
    agreement_fraction: float
    ensemble_rule: str = "2-of-3 majority, ties→UNCERTAIN"


SYSTEM_PROMPT = (
    "You are a radiology claim-verification grader. Given a chest X-ray image "
    "(or its description if you cannot view images), a radiology claim, and the "
    "original radiologist-written report, decide whether the claim is:\n"
    " - SUPPORTED: explicitly consistent with the report and/or image\n"
    " - CONTRADICTED: contradicts the report and/or image\n"
    " - NOVEL_PLAUSIBLE: not in the report but plausibly true from the image\n"
    " - NOVEL_HALLUCINATED: not in the report and unsupported by the image\n"
    " - UNCERTAIN: insufficient information\n"
    "Return strict JSON: {\"label\": ..., \"confidence\": 0..1, \"rationale\": \"<=30 words\"}."
)


def _encode_image_to_b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def grade_gpt4o_mini(image_path: Path, claim_text: str, report_text: str) -> LabelerOutput:
    """Primary budget labeler — gpt-4o-mini with vision (~10× cheaper than gpt-4o)."""
    from openai import OpenAI  # type: ignore[import]

    client = OpenAI()
    img_b64 = _encode_image_to_b64(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"Report: {report_text[:2000]}\nClaim: {claim_text}"},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    text = resp.choices[0].message.content or "{}"
    parsed = json.loads(text)
    return LabelerOutput(
        grader_id="gpt-4o-mini",
        grader_version=resp.model,
        label=str(parsed.get("label", "UNCERTAIN")).upper(),
        confidence=float(parsed.get("confidence", 0.5)),
        rationale=str(parsed.get("rationale", ""))[:200],
        prompt_version="v5.0",
    )


def grade_gpt4o(image_path: Path, claim_text: str, report_text: str) -> LabelerOutput:
    """High-quality labeler — reserved for κ-validation subset only (budget guard)."""
    from openai import OpenAI  # type: ignore[import]

    client = OpenAI()
    img_b64 = _encode_image_to_b64(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"Report: {report_text[:2000]}\nClaim: {claim_text}"},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    text = resp.choices[0].message.content or "{}"
    parsed = json.loads(text)
    return LabelerOutput(
        grader_id="gpt-4o",
        grader_version=resp.model,
        label=str(parsed.get("label", "UNCERTAIN")).upper(),
        confidence=float(parsed.get("confidence", 0.5)),
        rationale=str(parsed.get("rationale", ""))[:200],
        prompt_version="v5.0",
    )


def grade_claude_sonnet_45(image_path: Path, claim_text: str, report_text: str) -> LabelerOutput:
    import anthropic  # type: ignore[import]

    client = anthropic.Anthropic()
    img_b64 = _encode_image_to_b64(image_path)
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                    },
                    {"type": "text", "text": f"Report: {report_text[:2000]}\nClaim: {claim_text}"},
                ],
            }
        ],
    )
    text = resp.content[0].text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Find the first {...} in text
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}
    return LabelerOutput(
        grader_id="claude-sonnet-4-5",
        grader_version=resp.model,
        label=str(parsed.get("label", "UNCERTAIN")).upper(),
        confidence=float(parsed.get("confidence", 0.5)),
        rationale=str(parsed.get("rationale", ""))[:200],
        prompt_version="v5.0",
    )


def grade_llama_405b(claim_text: str, report_text: str) -> LabelerOutput:
    """Text-only labeler (no image). Used only as tiebreaker on non-visual claims.

    Wraps Together.ai's Llama-3.1-405B-Instruct by default.
    """
    try:
        from openai import OpenAI  # type: ignore[import]

        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=None,  # pulled from TOGETHER_API_KEY env var by the SDK if re-exported to OPENAI_API_KEY
        )
        resp = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n(Text-only; you cannot see the image.)"},
                {"role": "user", "content": f"Report: {report_text[:2000]}\nClaim: {claim_text}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        text = resp.choices[0].message.content or "{}"
        parsed = json.loads(text)
        return LabelerOutput(
            grader_id="llama-3.1-405b",
            grader_version=resp.model,
            label=str(parsed.get("label", "UNCERTAIN")).upper(),
            confidence=float(parsed.get("confidence", 0.4)),
            rationale=str(parsed.get("rationale", ""))[:200],
            prompt_version="v5.0",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Llama-405B labeler failed: %s", exc)
        return LabelerOutput(
            grader_id="llama-3.1-405b",
            grader_version="unknown",
            label="UNCERTAIN",
            confidence=0.0,
            rationale=f"api_error: {exc}",
            prompt_version="v5.0",
        )


def ensemble_label(
    image_path: Path,
    claim_id: str,
    claim_text: str,
    report_text: str,
    *,
    image_id: str | None = None,
) -> EnsembleLabel:
    # Budget-conscious ensemble: gpt-4o-mini (primary) + Claude Sonnet (secondary)
    # + Llama-405B text-only tiebreaker. gpt-4o reserved for κ-validation subset.
    outs: list[LabelerOutput] = []
    for fn in (grade_gpt4o_mini, grade_claude_sonnet_45):
        try:
            outs.append(fn(image_path, claim_text, report_text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("labeler %s failed: %s", fn.__name__, exc)
            outs.append(
                LabelerOutput(
                    grader_id=fn.__name__,
                    grader_version="error",
                    label="UNCERTAIN",
                    confidence=0.0,
                    rationale=f"exception: {exc}",
                    prompt_version="v5.0",
                )
            )
    outs.append(grade_llama_405b(claim_text, report_text))

    labels = [o.label for o in outs if o.label in VALID_LABELS]
    if not labels:
        return EnsembleLabel(
            image_id=image_id or image_path.stem,
            claim_id=claim_id,
            majority_label="UNCERTAIN",
            min_confidence=0.0,
            per_labeler=outs,
            agreement_fraction=0.0,
        )
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    if top_count < 2:
        majority = "UNCERTAIN"
        agreement = max(counts.values()) / len(labels)
    else:
        majority = top_label
        agreement = top_count / len(labels)
    agreeing_conf = [o.confidence for o in outs if o.label == majority]
    min_conf = min(agreeing_conf) if agreeing_conf else 0.0
    return EnsembleLabel(
        image_id=image_id or image_path.stem,
        claim_id=claim_id,
        majority_label=majority,
        min_confidence=float(min_conf),
        per_labeler=outs,
        agreement_fraction=agreement,
    )
