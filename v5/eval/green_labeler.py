"""GREEN-RadLlama-2-7B silver labeler for RRG-generated claims.

GREEN (Ostmeier et al., EMNLP Findings 2024, arXiv:2405.03595) is a fine-tuned
Llama-2-7B model that scores a (reference_report, candidate_report) pair along
six clinically-relevant error categories and emits a composite 0-1 score.
Reported Kendall tau against radiologists on ReXVal is 0.63.

We use GREEN in per-claim mode: for each atomic claim decomposed from an
RRG-generated report, we treat the claim as a minimal candidate "report"
and compare it against the full reference report. GREEN's explanation-list
output is parsed for any flagged significant errors -> CONTRADICTED; else
SUPPORTED.

The model ID ``StanfordAIMI/GREEN-RadLlama2-7b`` is released under the
Llama 2 Community License; our derivative silver labels inherit that license.
Inference does not touch MIMIC-CXR despite MIMIC appearing in the model's
training data (the known MIMIC-leakage confound is decoupled via VERT in
``vert_labeler.py``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)


@dataclass
class GreenLabel:
    """Per-claim GREEN silver label."""

    claim_id: str
    image_id: str
    rrg_model: str
    claim_text: str
    reference_excerpt: str
    label: str                  # SUPPORTED | CONTRADICTED | UNCERTAIN
    green_score: float          # 0-1 composite
    n_significant_errors: int
    error_categories_detected: list[str]
    raw_response: str
    latency_s: float


# GREEN was trained on full-report pairs. In per-claim mode (each claim becomes
# a minimal single-sentence candidate "report"), `missing_finding` trivially
# fires for every claim because the candidate omits everything else in the
# reference. We therefore treat it as non-significant in per-claim mode so it
# does not collapse all labels to CONTRADICTED.
_SIGNIFICANT_CATEGORIES = [
    "false_finding",
    "anatomic_location_error",
    "severity_error",
]

_NONSIG_CATEGORIES = [
    "missing_finding",
    "extraneous_comparison",
    "omitted_prior_comparison",
]


_GREEN_TEMPLATE = (
    "Original Report: {reference}\n\n"
    "Candidate Report: Findings: {candidate}\n\n"
    "Task: Compare the candidate report to the original report and identify any "
    "clinically significant errors. Respond with a structured analysis listing, "
    "for each of the six categories, the number of errors found:\n"
    "1. False finding (hallucinated findings)\n"
    "2. Missing finding\n"
    "3. Anatomic location error\n"
    "4. Severity error\n"
    "5. Extraneous comparison\n"
    "6. Omitted prior comparison\n"
    "Then provide a final GREEN score between 0 and 1."
)


class GreenLabeler:
    """Load GREEN-RadLlama-2-7B and label claims against a reference report."""

    model_id = "StanfordAIMI/GREEN-RadLlama2-7b"
    max_new_tokens = 512

    def __init__(self, device: torch.device | str = "cuda"):
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            if os.environ.get(k):
                kwargs["token"] = os.environ[k]
                break
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, **kwargs,
        ).to(self.device).eval()

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _parse_response(self, response: str) -> tuple[float, int, list[str]]:
        """Parse GREEN's actual output format.

        GREEN emits sections like:

            [Clinically Significant Errors]:
            (a) False report of a finding in the candidate: 2.
            (b) Missing a finding: 0.
            (c) Omitting prior comparison: 0.
            (d) Misassessment of severity: 1.
            (e) Mention of comparison: 0.
            (f) Omitting prior comparison: 0.

            [Clinically Insignificant Errors]:
            (a) ...

            [Matched Findings]:
            1. Calcified granuloma in the right upper lobe.
            2. ...

        For per-claim silver labeling:
          * (a) False report > 0 -> strong hallucination signal -> CONTRADICTED
          * (d) Misassessment of severity > 0 -> hallucination signal
          * Claim appears in [Matched Findings] (>= 1 numbered item in that
            section) and no (a)/(d) errors -> SUPPORTED
          * Neither -> UNCERTAIN

        Score is computed as the GREEN paper's precision-flavored ratio:
        matched / (matched + significant_errors) in per-claim mode.
        """
        # (a) False report count under [Clinically Significant Errors]
        sig_section = re.search(
            r"\[Clinically\s+Significant\s+Errors\]\s*:(.*?)(?=\[|$)",
            response, re.IGNORECASE | re.DOTALL,
        )
        n_false = 0
        n_severity = 0
        triggered: list[str] = []
        if sig_section:
            block = sig_section.group(1)
            m_a = re.search(r"\(a\)\s*[Ff]alse[^\n:]*:\s*(\d+)", block)
            if m_a:
                n_false = int(m_a.group(1))
                if n_false > 0:
                    triggered.append("false_finding")
            m_d = re.search(r"\(d\)\s*[Mm]isassessment[^\n:]*:\s*(\d+)", block)
            if m_d:
                n_severity = int(m_d.group(1))
                if n_severity > 0:
                    triggered.append("severity_error")

        # Count matched findings (numbered items under [Matched Findings])
        matched_section = re.search(
            r"\[Matched\s+Findings\]\s*:(.*?)(?=\[|</s>|$)",
            response, re.IGNORECASE | re.DOTALL,
        )
        n_matched = 0
        if matched_section:
            n_matched = len(re.findall(
                r"^\s*\d+\.\s+\S", matched_section.group(1), re.MULTILINE,
            ))

        n_sig = n_false + n_severity
        if n_matched + n_sig > 0:
            score = float(n_matched) / float(n_matched + n_sig)
        else:
            score = 0.5
        return score, n_sig, triggered

    def label_claim(
        self,
        *,
        claim_id: str,
        image_id: str,
        rrg_model: str,
        claim_text: str,
        reference_report: str,
        confidence_threshold: float = 0.7,
    ) -> GreenLabel:
        """Label a single claim against the reference report.

        Args:
            confidence_threshold: GREEN score above which SUPPORTED is returned when
                no significant errors are detected. Below it, UNCERTAIN.
        """
        t0 = time.time()
        reference_excerpt = reference_report[:1200]
        prompt = _GREEN_TEMPLATE.format(reference=reference_excerpt, candidate=claim_text)
        response = self._generate(prompt)
        score, n_sig, triggered = self._parse_response(response)
        if n_sig > 0:
            label = "CONTRADICTED"
        elif score >= confidence_threshold:
            label = "SUPPORTED"
        else:
            label = "UNCERTAIN"
        return GreenLabel(
            claim_id=claim_id,
            image_id=image_id,
            rrg_model=rrg_model,
            claim_text=claim_text,
            reference_excerpt=reference_excerpt,
            label=label,
            green_score=score,
            n_significant_errors=n_sig,
            error_categories_detected=triggered,
            raw_response=response,
            latency_s=time.time() - t0,
        )


def run_green_sweep(
    labeler: GreenLabeler,
    claims: Iterable[dict],
    references: dict[str, str],
    out_jsonl: Path,
    *,
    log_every: int = 50,
) -> dict[str, int]:
    """Apply GREEN to a stream of claims.

    Args:
        labeler: loaded GreenLabeler.
        claims: iterable of dicts with keys ``claim_id``, ``image_id``, ``rrg_model``,
            ``claim_text``.
        references: map from ``image_id`` to the ground-truth reference report text.
        out_jsonl: append-mode JSONL output.

    Returns:
        Dict mapping label -> count of rows emitted with that label.
    """
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNCERTAIN": 0, "ERROR": 0}
    claim_list = list(claims)
    with open(out_jsonl, "a") as fh:
        for i, c in enumerate(claim_list):
            ref = references.get(c["image_id"], "")
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
                logger.warning("GREEN label failed for claim %s: %s", c.get("claim_id"), exc)
                continue
            fh.write(json.dumps(asdict(label)) + "\n")
            fh.flush()
            counts[label.label] += 1
            if log_every and (i + 1) % log_every == 0:
                logger.info("green_sweep progress %d/%d counts=%s", i + 1, len(claim_list), counts)
    return counts
