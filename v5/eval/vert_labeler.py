"""VERT-style silver labeler (Claude Opus backbone, MIMIC-free at inference).

VERT (Tang et al., arXiv:2604.03376, April 2026) is a structured-prompt LLM
judge that benchmarks as the strongest LLM-based radiology report grader
currently published (Kendall tau = 0.4633 with radiologists on RaTE-Eval
using Claude Opus 4.6). The original paper is prompt-engineering around an
LLM; we re-implement from the paper description with Claude Opus 4.7.

Critical property for the ClaimGuard-CXR v6.0 paper: VERT is **MIMIC-free at
both training and inference** (unlike GREEN, which was trained on MIMIC-CXR
report pairs). Including VERT in the silver-label ensemble decouples the
labels from MIMIC-pretrained RRG outputs, addressing pre-flight reviewer
blocker B2.

The VERT prompt follows a structured five-step analysis:
1. Parse both reports into finding entities.
2. Check each finding in candidate for factual correctness.
3. Check each finding in reference for coverage.
4. Categorize errors by severity.
5. Output a verdict per atomic claim.

For our claim-level usage we simplify: for each atomic claim, ask Opus to
verify it against the reference report using the VERT five-step protocol,
and emit a SUPPORTED / CONTRADICTED / UNCERTAIN label.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class VertLabel:
    """Per-claim VERT silver label."""

    claim_id: str
    image_id: str
    rrg_model: str
    claim_text: str
    label: str                  # SUPPORTED | CONTRADICTED | UNCERTAIN
    severity: str               # minor | major | none
    rationale: str
    latency_s: float


_VERT_SYSTEM = (
    "You are a radiology evaluation expert performing the VERT protocol to assess a "
    "candidate claim against a reference radiology report. Follow these steps exactly:\n"
    "1) Identify all findings in the reference report (with anatomy, severity, certainty).\n"
    "2) Identify the finding(s) asserted by the candidate claim.\n"
    "3) Check whether the candidate claim's finding is factually consistent with the reference "
    "(same finding, compatible anatomy, compatible severity, compatible certainty).\n"
    "4) Categorize any discrepancy as: minor (style/wording), major (factually different finding, "
    "wrong anatomy, wrong laterality, opposite severity, opposite certainty), or none.\n"
    "5) Emit a final verdict.\n\n"
    "Output format, exactly three lines:\n"
    "VERDICT: SUPPORTED | CONTRADICTED | UNCERTAIN\n"
    "SEVERITY: minor | major | none\n"
    "RATIONALE: <one sentence explaining the decision>"
)


class _LazyAnthropic:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client


class VertLabeler:
    """VERT-protocol labeler using Claude Opus 4.7."""

    model = "claude-opus-4-7"
    max_tokens = 180

    def __init__(self):
        self._client = _LazyAnthropic()

    def _call(self, system: str, user: str) -> str:
        for attempt in range(3):
            try:
                msg = self._client.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text if msg.content else ""
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("VERT Anthropic call failed (attempt %d): %s; waiting %ds",
                               attempt + 1, exc, wait)
                time.sleep(wait)
        return ""

    def label_claim(
        self,
        *,
        claim_id: str,
        image_id: str,
        rrg_model: str,
        claim_text: str,
        reference_report: str,
    ) -> VertLabel:
        t0 = time.time()
        user = (
            f"REFERENCE REPORT:\n{reference_report[:2500]}\n\n"
            f"CANDIDATE CLAIM: {claim_text[:600]}"
        )
        raw = self._call(_VERT_SYSTEM, user)
        m_v = re.search(r"VERDICT:\s*(SUPPORTED|CONTRADICTED|UNCERTAIN)", raw, re.IGNORECASE)
        m_s = re.search(r"SEVERITY:\s*(minor|major|none)", raw, re.IGNORECASE)
        m_r = re.search(r"RATIONALE:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
        verdict = m_v.group(1).upper() if m_v else "UNCERTAIN"
        severity = m_s.group(1).lower() if m_s else "none"
        rationale = m_r.group(1).strip() if m_r else raw.strip()
        return VertLabel(
            claim_id=claim_id,
            image_id=image_id,
            rrg_model=rrg_model,
            claim_text=claim_text,
            label=verdict,
            severity=severity,
            rationale=rationale,
            latency_s=time.time() - t0,
        )


def run_vert_sweep(
    labeler: VertLabeler,
    claims: Iterable[dict],
    references: dict[str, str],
    out_jsonl: Path,
    *,
    log_every: int = 25,
) -> dict[str, int]:
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNCERTAIN": 0, "ERROR": 0}
    claim_list = list(claims)
    with open(out_jsonl, "a") as fh:
        for i, c in enumerate(claim_list):
            ref = references.get(str(c["image_id"]), "")
            if not ref:
                counts["ERROR"] += 1
                continue
            try:
                label = labeler.label_claim(
                    claim_id=str(c["claim_id"]),
                    image_id=str(c["image_id"]),
                    rrg_model=str(c.get("rrg_model", "unknown")),
                    claim_text=str(c["claim_text"]),
                    reference_report=ref,
                )
            except Exception as exc:
                counts["ERROR"] += 1
                logger.warning("VERT label failed for claim %s: %s", c.get("claim_id"), exc)
                continue
            fh.write(json.dumps(asdict(label)) + "\n")
            fh.flush()
            counts[label.label] += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("vert_sweep progress %d/%d counts=%s", i + 1, len(claim_list), counts)
    return counts
