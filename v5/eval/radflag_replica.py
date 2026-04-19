"""RadFlag replication: black-box temperature-sampling consistency detector.

Reimplements the detector from Zhang et al., "RadFlag: A Black-Box Hallucination
Detection Method for Medical Vision Language Models" (ML4H 2024,
arXiv:2411.00299). Their protocol:

1. Given an RRG model and a claim it generated, re-run the RRG on the same image
   at elevated temperature ``k`` times, producing ``k`` alternative generations.
2. For each alternative, decompose into atomic claims.
3. The target claim is "consistent" if it is entailed by at least some threshold
   fraction of alternatives; otherwise flagged as potential hallucination.
4. Entailment is assessed via an LLM (we use Claude Opus 4.7; the original used
   GPT-4).

Reported performance on MedVersa with 208 reports: 28% recall at 73% precision,
reducing 4.2 hallucinations/report to 1.9/report post-filter.

This implementation works against any generator exposing a ``generate`` method
with ``temperature`` support. We apply it to MAIRA-2 as the primary subject of
the prevalence study.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass
class RadFlagResult:
    claim_id: str
    image_id: str
    rrg_model: str
    target_claim: str
    n_resamples: int
    n_consistent: int
    consistency_rate: float
    flagged_hallucination: bool
    alternative_claims: list[list[str]]
    latency_s: float
    inference_failed: bool = False


_ENTAIL_SYSTEM = (
    "You are a strict logical entailment checker for radiology claims. "
    "Given a TARGET CLAIM and a list of ALTERNATIVE phrases from resampled "
    "report drafts, decide whether the target claim is logically consistent "
    "with (entailed by, or at least not contradicted by) the alternatives "
    "as a whole.\n"
    "Respond in exactly this format on two lines:\n"
    "CONSISTENT: yes|no\n"
    "RATIONALE: <one sentence>"
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


class RadFlagDetector:
    """Resample-and-check hallucination flagger.

    The generator interface expected: a callable
    ``resample(image, temperature, seed) -> str`` returning a fresh report at
    elevated temperature. The decomposer interface: ``decompose(report) -> list[str]``.
    We delegate decomposition to the same RadFact-style Haiku pipeline used by
    ``radfact_labeler`` to keep consistent atomic-phrase granularity.
    """

    entail_model = "claude-opus-4-7"

    def __init__(
        self,
        *,
        resample_fn: Callable[[Any, float, int], str],
        decompose_fn: Callable[[str], list[str]],
        n_resamples: int = 10,
        temperature: float = 1.0,
        consistency_threshold: float = 0.5,
    ):
        self.resample_fn = resample_fn
        self.decompose_fn = decompose_fn
        self.n_resamples = n_resamples
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold
        self._anthropic = _LazyAnthropic()

    def _check_entailment(self, target: str, alternatives: list[list[str]]) -> bool:
        flat = [p for alt in alternatives for p in alt][:60]
        alt_block = "\n".join(f"- {p}" for p in flat)
        user = f"TARGET CLAIM: {target}\n\nALTERNATIVE PHRASES:\n{alt_block}"
        for attempt in range(3):
            try:
                msg = self._anthropic.client.messages.create(
                    model=self.entail_model,
                    max_tokens=80,
                    system=_ENTAIL_SYSTEM,
                    messages=[{"role": "user", "content": user}],
                )
                raw = msg.content[0].text if msg.content else ""
                m = re.search(r"CONSISTENT:\s*(yes|no)", raw, re.IGNORECASE)
                return bool(m and m.group(1).lower() == "yes")
            except Exception as exc:
                if attempt == 2:
                    logger.warning("entailment check failed: %s", exc)
                    return False
                time.sleep(2 ** attempt)
        return False

    def flag_claim(
        self,
        *,
        claim_id: str,
        image_id: str,
        rrg_model: str,
        target_claim: str,
        image: Any,
    ) -> RadFlagResult:
        t0 = time.time()
        alternatives: list[list[str]] = []
        for seed in range(self.n_resamples):
            try:
                alt_report = self.resample_fn(image, self.temperature, seed)
            except Exception as exc:
                logger.warning("resample %d failed: %s", seed, exc)
                continue
            try:
                phrases = self.decompose_fn(alt_report)
            except Exception as exc:
                logger.warning("decompose on alt %d failed: %s", seed, exc)
                continue
            alternatives.append(phrases)

        if not alternatives:
            # All resample or decompose attempts failed — cannot assess
            # consistency. Flag the row with ``inference_failed=True`` so
            # downstream metric computation can exclude it, rather than
            # asserting hallucination on an infrastructure failure.
            return RadFlagResult(
                claim_id=claim_id,
                image_id=image_id,
                rrg_model=rrg_model,
                target_claim=target_claim,
                n_resamples=0,
                n_consistent=0,
                consistency_rate=0.0,
                flagged_hallucination=False,
                alternative_claims=[],
                latency_s=time.time() - t0,
                inference_failed=True,
            )

        n_consistent = 0
        for alt in alternatives:
            if self._check_entailment(target_claim, [alt]):
                n_consistent += 1
        rate = n_consistent / len(alternatives)
        flagged = rate < self.consistency_threshold
        return RadFlagResult(
            claim_id=claim_id,
            image_id=image_id,
            rrg_model=rrg_model,
            target_claim=target_claim,
            n_resamples=len(alternatives),
            n_consistent=n_consistent,
            consistency_rate=rate,
            flagged_hallucination=flagged,
            alternative_claims=alternatives,
            latency_s=time.time() - t0,
        )


def run_radflag_sweep(
    detector: RadFlagDetector,
    rows: Iterable[dict],
    image_loader: Callable[[str], Any],
    out_jsonl: Path,
    *,
    log_every: int = 10,
) -> dict[str, int]:
    """Apply RadFlag to a stream of (image, target_claim) rows.

    Args:
        detector: loaded RadFlagDetector.
        rows: iterable of dicts with keys ``claim_id``, ``image_id``, ``rrg_model``,
            ``claim_text``, ``image_path``.
        image_loader: callable that takes an image path and returns the argument
            expected by the detector's ``resample_fn``.
        out_jsonl: append-mode JSONL output.

    Returns:
        Dict with counts of ``flagged`` and ``not_flagged`` results.
    """
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"flagged": 0, "not_flagged": 0, "error": 0}
    row_list = list(rows)
    with open(out_jsonl, "a") as fh:
        for i, r in enumerate(row_list):
            try:
                image = image_loader(str(r["image_path"]))
            except Exception as exc:
                logger.warning("image load failed at row %d: %s", i, exc)
                counts["error"] += 1
                continue
            try:
                result = detector.flag_claim(
                    claim_id=str(r["claim_id"]),
                    image_id=str(r["image_id"]),
                    rrg_model=str(r.get("rrg_model", "unknown")),
                    target_claim=str(r["claim_text"]),
                    image=image,
                )
            except Exception as exc:
                logger.warning("flag_claim failed at row %d: %s", i, exc)
                counts["error"] += 1
                continue
            fh.write(json.dumps(asdict(result)) + "\n")
            fh.flush()
            counts["flagged" if result.flagged_hallucination else "not_flagged"] += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("radflag_sweep %d/%d counts=%s", i + 1, len(row_list), counts)
    return counts
