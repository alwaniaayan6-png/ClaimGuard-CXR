"""Baseline registry for ClaimGuard-Bench-Grounded.

Each baseline is a thin wrapper with a uniform ``score_claim()``
interface so the evaluation harness can iterate over baselines and
sites uniformly. Baselines that call external APIs (GPT-4o, Claude,
Llama-3.1-405B via OpenRouter) are expected to read credentials from
env vars at call time — never at import time.

Nothing here makes a network call at import. Concrete network-calling
subclasses must be instantiated explicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from claimguard_nmi.grounding.claim_schema import Claim


@dataclass
class BaselineScore:
    """Output of one baseline call on one (image, claim, evidence) triple."""
    contradicted_prob: float
    supported_prob: float
    raw_response: Optional[str] = None
    latency_ms: Optional[float] = None
    baseline_id: str = ""

    @property
    def pred_label(self) -> int:
        """0 = not contradicted, 1 = contradicted."""
        return int(self.contradicted_prob >= 0.5)


class Baseline(ABC):
    """Abstract claim-level verifier baseline."""
    baseline_id: str = ""

    @abstractmethod
    def score_claim(
        self,
        claim: Claim,
        evidence_text: str,
        image_path: Optional[str] = None,
    ) -> BaselineScore: ...


# ---------------------------------------------------------------------------
# Stub baselines (no external calls). Useful for CI / unit tests.
# ---------------------------------------------------------------------------
class MajorityClassBaseline(Baseline):
    """Always predicts 'not contradicted'. Demonstrates why accuracy alone is useless."""
    baseline_id = "majority_class"

    def score_claim(self, claim, evidence_text, image_path=None):
        return BaselineScore(
            contradicted_prob=0.0,
            supported_prob=1.0,
            raw_response="majority_class",
            baseline_id=self.baseline_id,
        )


class RuleBasedNegationBaseline(Baseline):
    """Simple keyword-overlap + negation proximity heuristic."""
    baseline_id = "rule_based_negation"

    NEG_CUES = frozenset({"no", "without", "negative", "absent", "rule out", "ruled out"})

    def score_claim(self, claim, evidence_text, image_path=None):
        finding = claim.finding.replace("_", " ").lower()
        ev = (evidence_text or "").lower()
        if finding not in ev:
            # Evidence does not mention the finding — treat as insufficient => not contradicted.
            return BaselineScore(
                contradicted_prob=0.2,
                supported_prob=0.8,
                raw_response="finding_not_in_evidence",
                baseline_id=self.baseline_id,
            )
        # Check if the finding mention is preceded by a negation cue.
        idx = ev.index(finding)
        window = ev[max(0, idx - 40): idx].split()
        has_negation_cue = any(cue in window for cue in self.NEG_CUES)
        claim_asserts_present = claim.certainty.value == "present"
        if claim_asserts_present and has_negation_cue:
            # Claim says present, evidence says absent.
            return BaselineScore(
                contradicted_prob=0.8,
                supported_prob=0.2,
                raw_response="neg_cue_before_finding",
                baseline_id=self.baseline_id,
            )
        return BaselineScore(
            contradicted_prob=0.1,
            supported_prob=0.9,
            raw_response="consistent",
            baseline_id=self.baseline_id,
        )


# ---------------------------------------------------------------------------
# LLM-backed baselines — contract only; concrete backends supplied by caller.
# ---------------------------------------------------------------------------
class LLMJudgeBaseline(Baseline):
    """Zero-shot LLM-as-judge baseline.

    Caller supplies a backend conforming to the ``LLMBackend`` contract
    (see extraction.claim_extractor). We parse the backend's JSON output
    into a ``{contradicted_prob, supported_prob}`` dict.
    """
    baseline_id = "llm_judge"

    SYSTEM_PROMPT = (
        "You are a medical fact-checker. Given a chest X-ray report claim and "
        "its retrieved evidence passage, decide whether the claim is "
        "CONTRADICTED by the evidence. Return JSON: "
        '{"contradicted_prob": 0.0-1.0, "rationale": "≤30 words"}'
    )

    def __init__(self, backend, baseline_id: Optional[str] = None):
        self._backend = backend
        if baseline_id:
            self.baseline_id = baseline_id

    @staticmethod
    def _strip_fences(raw: str) -> str:
        import re
        s = (raw or "").strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        return s

    def score_claim(self, claim, evidence_text, image_path=None):
        import json
        user = (
            f"Claim: {claim.raw_text}\n\n"
            f"Evidence: {evidence_text}\n\n"
            "Return JSON only."
        )
        raw = self._backend.generate_json(self.SYSTEM_PROMPT, user)
        # v2 review fix: LLMs frequently return ```json {…}``` fences. Previously
        # json.loads failed silently and produced 0.5 (a "floor UNCERTAIN"), which
        # is the exact pattern from the v3 sprint silver-grader failure.
        cleaned = self._strip_fences(raw)
        try:
            obj = json.loads(cleaned)
            contradicted_prob = float(obj.get("contradicted_prob", 0.5))
        except (json.JSONDecodeError, TypeError, ValueError):
            contradicted_prob = 0.5
        contradicted_prob = max(0.0, min(1.0, contradicted_prob))
        return BaselineScore(
            contradicted_prob=contradicted_prob,
            supported_prob=1.0 - contradicted_prob,
            raw_response=raw,
            baseline_id=self.baseline_id,
        )


class VLMClaimBaseline(Baseline):
    """Generic VLM-as-judge baseline. Takes a backend that accepts
    (image_path, prompt) -> str. Intended for CheXagent-8b, MedGemma-4B,
    LLaVA-Rad, Llama-3.2-Vision 11B.
    """
    baseline_id = "vlm_judge"

    SYSTEM_PROMPT = (
        "You are a radiologist. Given a chest X-ray and a textual claim about "
        "the image, decide if the claim is CORRECT (supported) or INCORRECT "
        "(contradicted) based on the image alone. Answer 'correct' or 'incorrect'."
    )

    def __init__(self, vlm_backend, baseline_id: Optional[str] = None):
        self._backend = vlm_backend
        if baseline_id:
            self.baseline_id = baseline_id

    def score_claim(self, claim, evidence_text, image_path=None):
        import re

        if image_path is None:
            return BaselineScore(
                contradicted_prob=0.5, supported_prob=0.5,
                raw_response="no_image", baseline_id=self.baseline_id,
            )
        prompt = f"{self.SYSTEM_PROMPT}\n\nClaim: {claim.raw_text}"
        raw = self._backend.generate(image_path=image_path, prompt=prompt)
        low = (raw or "").lower()
        # Word-boundary match so "incorrect" doesn't satisfy /correct/.
        has_incorrect = bool(re.search(r"\b(incorrect|contradict\w*|wrong)\b", low))
        has_correct = bool(re.search(r"\b(correct|supported|accurate)\b", low))
        if has_incorrect:
            contradicted_prob = 0.8
        elif has_correct:
            contradicted_prob = 0.2
        else:
            contradicted_prob = 0.5
        return BaselineScore(
            contradicted_prob=contradicted_prob,
            supported_prob=1.0 - contradicted_prob,
            raw_response=raw,
            baseline_id=self.baseline_id,
        )


class RadFlagBaseline(Baseline):
    """Self-consistency / ensemble-flag baseline (Chen et al. 2025).

    Given a single generator producing K samples at temperature T, the
    claim is flagged as hallucinated iff it appears in fewer than
    ``consistency_threshold`` of the K samples. Here we take already-
    sampled reports and compute overlap; the user supplies the K samples
    at score time.
    """
    baseline_id = "radflag"

    def __init__(self, consistency_threshold: float = 0.5):
        self._thr = consistency_threshold

    def score_claim(
        self,
        claim,
        evidence_text,
        image_path=None,
        k_samples: Optional[List[str]] = None,
    ):
        if not k_samples:
            return BaselineScore(
                contradicted_prob=0.5, supported_prob=0.5,
                raw_response="no_samples", baseline_id=self.baseline_id,
            )
        finding = claim.finding.replace("_", " ").lower()
        hits = sum(1 for s in k_samples if finding in (s or "").lower())
        fraction = hits / len(k_samples)
        contradicted_prob = 1.0 - fraction if claim.certainty.value == "present" else fraction
        contradicted_prob = max(0.0, min(1.0, contradicted_prob))
        return BaselineScore(
            contradicted_prob=contradicted_prob,
            supported_prob=1.0 - contradicted_prob,
            raw_response=f"{hits}/{len(k_samples)} consistency",
            baseline_id=self.baseline_id,
        )


_REGISTRY: Dict[str, type] = {
    "majority_class": MajorityClassBaseline,
    "rule_based_negation": RuleBasedNegationBaseline,
    "llm_judge": LLMJudgeBaseline,
    "vlm_judge": VLMClaimBaseline,
    "radflag": RadFlagBaseline,
}


def get_baseline(name: str, **kwargs) -> Baseline:
    """Instantiate a baseline by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline '{name}'; registered: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(**kwargs)


def list_baselines() -> List[str]:
    return sorted(_REGISTRY.keys())
