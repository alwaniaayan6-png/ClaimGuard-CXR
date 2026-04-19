"""RadFact-style silver labeler with cost-optimized LLM backbone split.

RadFact (Bannur et al. 2024, MAIRA-2 tech report, arXiv:2406.04449;
github.com/microsoft/RadFact, MIT) decomposes each report into atomic phrases
then performs bidirectional entailment checking via an LLM.

Cost analysis (corrected per pre-flight reviewer B1): a report with ~6 atomic
claims requires ~1 decomposition call + ~6 bidirectional entailment calls =
~13 LLM calls per report pair. To keep the 1000-claim agreement run under
the $120 budget line, we split:

* **Decomposition** uses Claude Haiku (cheap, good at atomic phrase extraction).
* **Bidirectional entailment** uses Claude Opus 4.7 (highest Kendall tau
  among LLM judges on RaTE-Eval per VERT 2026).

For our claim-level usage: each atomic claim from an RRG-generated report is
paired against the reference OpenI report. We ask Opus whether the reference
entails the claim (claim_entailed), and whether the claim implies anything
contradictory to the reference (claim_contradicted). SUPPORTED iff entailed
and not contradicted; CONTRADICTED iff contradicted; UNCERTAIN otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class RadFactLabel:
    """Per-claim RadFact silver label."""

    claim_id: str
    image_id: str
    rrg_model: str
    claim_text: str
    label: str                      # SUPPORTED | CONTRADICTED | UNCERTAIN
    claim_entailed: bool
    claim_contradicted: bool
    reference_phrases: list[str]
    entailment_rationale: str
    latency_s: float


_DECOMPOSE_SYSTEM = (
    "You are a radiology report decomposer. Split the input report into a bulleted list "
    "of atomic clinical phrases. Each phrase must assert a single finding or absence. "
    "Return one phrase per line, prefixed with '- '. No other output."
)

_ENTAIL_SYSTEM = (
    "You are a radiology claim verifier performing bidirectional entailment. "
    "Given a reference set of atomic radiology phrases and a candidate claim, "
    "decide: (a) does the reference ENTAIL the claim (i.e., the claim must be true "
    "if the reference is true)? (b) does the claim CONTRADICT the reference (i.e., "
    "the claim and reference cannot both be true)? "
    "Respond in exactly this format on three lines:\n"
    "ENTAILED: yes|no\n"
    "CONTRADICTED: yes|no\n"
    "RATIONALE: <one sentence>"
)


class _LazyAnthropic:
    """Lazy Anthropic client init so the module imports without credentials."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client


class RadFactLabeler:
    """RadFact-style decomposer (Haiku) + entailment (Opus) pipeline."""

    decompose_model = "claude-haiku-4-5-20251001"
    entail_model = "claude-opus-4-7"
    max_tokens_decompose = 400
    max_tokens_entail = 120

    def __init__(self):
        self._client = _LazyAnthropic()

    def _call(self, model: str, system: str, user: str, max_tokens: int) -> str:
        for attempt in range(3):
            try:
                msg = self._client.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text if msg.content else ""
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("Anthropic call failed (attempt %d, model=%s): %s; waiting %ds",
                               attempt + 1, model, exc, wait)
                time.sleep(wait)
        return ""

    def decompose(self, report_text: str) -> list[str]:
        """Decompose a report into atomic phrases via Claude Haiku."""
        if not report_text.strip():
            return []
        raw = self._call(
            self.decompose_model,
            _DECOMPOSE_SYSTEM,
            report_text.strip()[:4000],
            self.max_tokens_decompose,
        )
        phrases: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ")):
                phrases.append(line[2:].strip())
            elif line and line[0].isdigit() and "." in line[:3]:
                phrases.append(line.split(".", 1)[1].strip())
        return [p for p in phrases if p]

    def verify(
        self,
        *,
        reference_phrases: list[str],
        candidate_claim: str,
    ) -> tuple[bool, bool, str]:
        """Ask Opus for bidirectional entailment. Returns (entailed, contradicted, rationale)."""
        ref_block = "\n".join(f"- {p}" for p in reference_phrases[:30])
        user = (
            f"REFERENCE PHRASES:\n{ref_block}\n\n"
            f"CANDIDATE CLAIM: {candidate_claim.strip()[:600]}"
        )
        raw = self._call(self.entail_model, _ENTAIL_SYSTEM, user, self.max_tokens_entail)
        m_ent = re.search(r"ENTAILED:\s*(yes|no)", raw, re.IGNORECASE)
        m_con = re.search(r"CONTRADICTED:\s*(yes|no)", raw, re.IGNORECASE)
        m_rat = re.search(r"RATIONALE:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
        entailed = bool(m_ent and m_ent.group(1).lower() == "yes")
        contradicted = bool(m_con and m_con.group(1).lower() == "yes")
        rationale = m_rat.group(1).strip() if m_rat else raw.strip()
        return entailed, contradicted, rationale

    def label_claim(
        self,
        *,
        claim_id: str,
        image_id: str,
        rrg_model: str,
        claim_text: str,
        reference_phrases: list[str] | None = None,
        reference_report: str | None = None,
    ) -> RadFactLabel:
        """Label a single claim, decomposing the reference if phrases not cached."""
        t0 = time.time()
        if reference_phrases is None:
            if reference_report is None:
                raise ValueError("must provide reference_phrases or reference_report")
            reference_phrases = self.decompose(reference_report)
        entailed, contradicted, rationale = self.verify(
            reference_phrases=reference_phrases,
            candidate_claim=claim_text,
        )
        if contradicted:
            label = "CONTRADICTED"
        elif entailed:
            label = "SUPPORTED"
        else:
            label = "UNCERTAIN"
        return RadFactLabel(
            claim_id=claim_id,
            image_id=image_id,
            rrg_model=rrg_model,
            claim_text=claim_text,
            label=label,
            claim_entailed=entailed,
            claim_contradicted=contradicted,
            reference_phrases=reference_phrases,
            entailment_rationale=rationale,
            latency_s=time.time() - t0,
        )


def run_radfact_sweep(
    labeler: RadFactLabeler,
    claims: Iterable[dict],
    references: dict[str, str],
    out_jsonl: Path,
    *,
    phrase_cache: dict[str, list[str]] | None = None,
    log_every: int = 25,
) -> dict[str, int]:
    """Apply RadFact to a stream of claims, caching decompositions per image.

    Args:
        labeler: loaded RadFactLabeler.
        claims: iterable of dicts with keys ``claim_id``, ``image_id``, ``rrg_model``,
            ``claim_text``.
        references: map from ``image_id`` to raw reference report text.
        out_jsonl: append-mode JSONL output.
        phrase_cache: optional pre-populated phrase cache keyed by image_id.

    Returns:
        Dict mapping label -> count of rows emitted with that label.
    """
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if phrase_cache is None:
        phrase_cache = {}
    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNCERTAIN": 0, "ERROR": 0}
    claim_list = list(claims)
    with open(out_jsonl, "a") as fh:
        for i, c in enumerate(claim_list):
            image_id = str(c["image_id"])
            ref = references.get(image_id, "")
            if not ref:
                counts["ERROR"] += 1
                continue
            if image_id not in phrase_cache:
                try:
                    phrase_cache[image_id] = labeler.decompose(ref)
                except Exception as exc:
                    logger.warning("RadFact decompose failed for %s: %s", image_id, exc)
                    counts["ERROR"] += 1
                    continue
            try:
                label = labeler.label_claim(
                    claim_id=str(c["claim_id"]),
                    image_id=image_id,
                    rrg_model=str(c.get("rrg_model", "unknown")),
                    claim_text=str(c["claim_text"]),
                    reference_phrases=phrase_cache[image_id],
                )
            except Exception as exc:
                counts["ERROR"] += 1
                logger.warning("RadFact verify failed for claim %s: %s", c.get("claim_id"), exc)
                continue
            fh.write(json.dumps(asdict(label)) + "\n")
            fh.flush()
            counts[label.label] += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("radfact_sweep progress %d/%d counts=%s", i + 1, len(claim_list), counts)
    return counts
