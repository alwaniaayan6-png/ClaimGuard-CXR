"""LLM-based contextual claim extractor for ClaimGuard-CXR v2.

Replaces the naive rule-based sentence splitter from v1. Uses Phi-3-mini
(3.8B params, runs on CPU in ~2s/report) to extract contextually complete
atomic claims that preserve negation, temporal, and laterality context
across sentence boundaries.

Why this matters:
  Radiology reports are heavily context-dependent. "The previously noted
  moderate pleural effusion has resolved." — a rule-based splitter might
  extract "moderate pleural effusion" without the "resolved" context,
  triggering a false positive contradiction. Temporal references
  ("compared to prior", "interval decrease") and negation scope
  ("no evidence of pneumothorax or effusion") cross boundaries regularly.

Falls back to enhanced rule-based splitting when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt template for LLM claim extraction
EXTRACTION_PROMPT = """You are a radiology claim extractor. Given a radiology report, extract a JSON list of atomic, self-contained claims.

Rules for each claim:
1. Include ALL negation context ("no pleural effusion", not just "pleural effusion")
2. Include temporal qualifiers ("new", "unchanged", "resolved", "compared to prior")
3. Include laterality ("left", "right", "bilateral")
4. Include anatomical location ("left lower lobe", not just "lobe")
5. Each claim must be independently verifiable without reading the full report
6. If a sentence contains multiple findings, split into separate claims
7. Do NOT split a finding from its negation, temporal status, or location

Report:
{report_text}

Output ONLY a JSON array:
[{{"claim": "...", "pathology": "..."}}]

Where pathology is one of: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pneumonia, Pneumothorax, Support Devices, Other"""

# CheXpert pathology keywords for rule-based fallback
PATHOLOGY_KEYWORDS = {
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart"],
    "Consolidation": ["consolidation", "consolidated"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Pneumonia": ["pneumonia", "infectious process"],
    "Pneumothorax": ["pneumothorax"],
    "Lung Opacity": ["opacity", "opacification", "opacities"],
    "Lung Lesion": ["lesion", "mass", "nodule"],
    "Fracture": ["fracture", "fractured"],
    "Support Devices": ["tube", "catheter", "line", "pacemaker", "picc",
                        "endotracheal", "nasogastric", "tracheostomy"],
    "No Finding": ["normal", "unremarkable", "no acute", "clear lungs"],
}


class LLMClaimExtractor:
    """Extract contextually complete atomic claims from radiology reports.

    Uses Phi-3-mini for LLM-based extraction, with enhanced rule-based fallback.

    Args:
        use_llm: Whether to use LLM extraction (True) or rule-based fallback.
        model_name: HuggingFace model for LLM extraction.
        device: Device for LLM inference ("cpu" is fine for Phi-3-mini).
    """

    def __init__(
        self,
        use_llm: bool = True,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cpu",
    ):
        self.use_llm = use_llm
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_llm(self) -> bool:
        """Lazy-load LLM. Returns True if successful."""
        if self._model is not None:
            return True
        if not self.use_llm:
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading {self.model_name} for claim extraction...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # CPU inference
                device_map=self.device,
                trust_remote_code=True,
            )
            self._model.eval()
            logger.info(f"Loaded {self.model_name} on {self.device}")
            return True
        except Exception as e:
            logger.warning(f"LLM loading failed: {e}. Using rule-based fallback.")
            self.use_llm = False
            return False

    def extract_claims(
        self,
        report_text: str,
        max_claims: int = 20,
    ) -> list[dict]:
        """Extract atomic claims from a radiology report.

        Args:
            report_text: Full radiology report text.
            max_claims: Maximum number of claims to extract.

        Returns:
            List of dicts with 'claim' and 'pathology' keys.
        """
        if not report_text or len(report_text.strip()) < 10:
            return []

        if self.use_llm and self._load_llm():
            claims = self._extract_with_llm(report_text, max_claims)
            if claims:
                return claims
            # LLM failed to produce valid output, fall through to rule-based

        return self._extract_rule_based(report_text, max_claims)

    def _extract_with_llm(
        self,
        report_text: str,
        max_claims: int,
    ) -> list[dict]:
        """Extract claims using LLM."""
        import torch

        prompt = EXTRACTION_PROMPT.format(report_text=report_text)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse JSON from response
        claims = self._parse_json_claims(response)
        return claims[:max_claims]

    def _parse_json_claims(self, response: str) -> list[dict]:
        """Parse JSON claim list from LLM response."""
        # Try to find JSON array in the response
        # Handle cases where LLM wraps in ```json ... ```
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON array found in LLM response: {response[:200]}")
            return []

        try:
            claims_raw = json.loads(json_match.group())
            claims = []
            for item in claims_raw:
                if isinstance(item, dict) and "claim" in item:
                    claim_text = item["claim"].strip()
                    if len(claim_text) >= 10:
                        claims.append({
                            "claim": claim_text,
                            "pathology": item.get("pathology", "Other"),
                        })
            return claims
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

    def _extract_rule_based(
        self,
        report_text: str,
        max_claims: int,
    ) -> list[dict]:
        """Enhanced rule-based extraction with context-aware merging.

        Improvements over v1 naive sentence splitting:
        1. Merges sentences that share negation scope
        2. Preserves temporal context from preceding sentences
        3. Tags pathology category per claim
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', report_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return []

        # Context-aware merging pass
        merged = []
        i = 0
        while i < len(sentences):
            current = sentences[i]

            # Check if next sentence depends on context from current
            if i + 1 < len(sentences):
                next_sent = sentences[i + 1]

                # Merge if current has negation scope extending to next
                # e.g., "There is no pneumothorax." + "Or pleural effusion."
                if _is_continuation(current, next_sent):
                    current = current.rstrip('.') + ", " + next_sent[0].lower() + next_sent[1:]
                    i += 1

                # Merge if next references "it", "this", "the finding" etc.
                elif _is_anaphoric(next_sent):
                    current = current.rstrip('.') + "; " + next_sent
                    i += 1

            merged.append(current)
            i += 1

        # Tag pathology and return
        claims = []
        for sent in merged[:max_claims]:
            pathology = _classify_pathology(sent)
            claims.append({
                "claim": sent,
                "pathology": pathology,
            })

        return claims


def _is_continuation(current: str, next_sent: str) -> bool:
    """Check if next_sent is a continuation of current's negation/list scope."""
    continuation_starts = ["or ", "and ", "nor ", "also ", "additionally "]
    next_lower = next_sent.lower().strip()
    for prefix in continuation_starts:
        if next_lower.startswith(prefix):
            return True
    return False


def _is_anaphoric(sent: str) -> bool:
    """Check if sentence starts with anaphoric reference needing prior context."""
    anaphoric_starts = [
        "it ", "this ", "these ", "that ", "which ", "the finding ",
        "the above ", "the previously ", "compared to ", "interval ",
    ]
    sent_lower = sent.lower().strip()
    for prefix in anaphoric_starts:
        if sent_lower.startswith(prefix):
            return True
    return False


def _classify_pathology(claim_text: str) -> str:
    """Classify a claim into CheXpert pathology category."""
    claim_lower = claim_text.lower()
    for pathology, keywords in PATHOLOGY_KEYWORDS.items():
        for kw in keywords:
            if kw in claim_lower:
                return pathology
    return "Other"
