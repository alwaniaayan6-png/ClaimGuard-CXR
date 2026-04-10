"""Claim Decomposer for ClaimGuard-CXR.

Splits radiology reports into atomic verifiable claims, each mapped to a
pathology category. Supports both fine-tuned model inference and zero/few-shot
prompting as a lightweight alternative.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# CheXpert pathology categories for claim mapping
PATHOLOGY_CATEGORIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "Rare/Other",
]

# Keywords for pathology classification
_PATHOLOGY_KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart", "heart size"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "cephalization"],
    "Consolidation": ["consolidation", "airspace disease", "air bronchograms"],
    "Pneumonia": ["pneumonia", "infection", "infectious"],
    "Atelectasis": ["atelectasis", "volume loss", "collapse"],
    "Pneumothorax": ["pneumothorax", "air leak"],
    "Pleural Effusion": ["pleural effusion", "effusion", "fluid"],
    "Pleural Other": ["pleural thickening", "pleural plaque", "pleural abnormality"],
    "Fracture": ["fracture", "broken", "rib fracture"],
    "Support Devices": [
        "tube", "catheter", "line", "pacemaker", "device", "drain",
        "ett", "endotracheal", "picc", "port", "ng tube", "chest tube",
        "tracheostomy", "icd", "swan-ganz", "central venous",
    ],
    "Lung Opacity": ["opacity", "infiltrate", "haziness", "density"],
    "Lung Lesion": ["mass", "nodule", "lesion", "tumor"],
    "Enlarged Cardiomediastinum": ["mediastinum", "mediastinal", "widened"],
    "No Finding": ["normal", "clear", "unremarkable", "no acute", "within normal"],
}


@dataclass
class DecomposedClaim:
    """A single atomic claim extracted from a report."""
    text: str
    pathology_category: str
    confidence: float
    source_sentence: str = ""


def classify_claim_pathology(claim_text: str) -> str:
    """Map a claim to its most likely CheXpert pathology category.

    Args:
        claim_text: The atomic claim text.

    Returns:
        Best-matching pathology category string.
    """
    text_lower = claim_text.lower()

    best_category = "Rare/Other"
    best_score = 0

    for category, keywords in _PATHOLOGY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def sentence_split(report_text: str) -> list[str]:
    """Split a radiology report into sentences.

    Handles common radiology abbreviations and formatting quirks.

    Args:
        report_text: Full report text.

    Returns:
        List of sentence strings.
    """
    if not report_text or not report_text.strip():
        return []

    # Protect common abbreviations from splitting
    text = report_text.strip()
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|vs|etc|e\.g|i\.e)\.',
                  lambda m: m.group(0).replace('.', '<DOT>'), text)

    # Split on period + space, newline, or semicolon
    sentences = re.split(r'(?<=[.!?])\s+|\n+|;\s*', text)

    # Restore dots
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]

    # Filter empty and very short
    sentences = [s for s in sentences if len(s) > 5]

    return sentences


class ZeroShotDecomposer:
    """Zero/few-shot claim decomposer using a language model.

    Uses prompting to decompose reports without fine-tuning.
    Good for prototyping and when RadGraph-XL training data isn't available.

    Args:
        model_name: HuggingFace model ID for the LLM.
        device: Device to run on.
    """

    DECOMPOSE_PROMPT = """Break the following radiology report into individual atomic claims.
Each claim should be a single factual statement about one finding.
For each claim, also classify it into one of these categories: {categories}

Report: {report}

Output as JSON array: [{{"claim": "...", "category": "..."}}]
Claims:"""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cpu",
        max_new_tokens: int = 512,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Loaded decomposer model: {model_name}")

    def decompose(self, report_text: str) -> list[DecomposedClaim]:
        """Decompose a report into atomic claims using LLM prompting.

        Args:
            report_text: Full report text.

        Returns:
            List of DecomposedClaim objects.
        """
        if not report_text or not report_text.strip():
            return []

        prompt = self.DECOMPOSE_PROMPT.format(
            categories=", ".join(PATHOLOGY_CATEGORIES),
            report=report_text.strip(),
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse JSON output
        claims = self._parse_llm_output(generated, report_text)
        return claims

    def _parse_llm_output(self, text: str, source_report: str) -> list[DecomposedClaim]:
        """Parse LLM output into DecomposedClaim objects."""
        claims = []

        # Try JSON parsing first
        try:
            # Find JSON array in text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for item in parsed:
                    claim_text = item.get("claim", "")
                    category = item.get("category", "")
                    if claim_text:
                        # Validate category
                        if category not in PATHOLOGY_CATEGORIES:
                            category = classify_claim_pathology(claim_text)
                        claims.append(DecomposedClaim(
                            text=claim_text,
                            pathology_category=category,
                            confidence=0.8,
                            source_sentence=source_report,
                        ))
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: sentence splitting + keyword classification
        if not claims:
            logger.debug("LLM JSON parsing failed, falling back to sentence splitting")
            claims = self.fallback_decompose(source_report)

        return claims

    def fallback_decompose(self, report_text: str) -> list[DecomposedClaim]:
        """Fallback decomposition using sentence splitting + keyword classification.

        Args:
            report_text: Full report text.

        Returns:
            List of DecomposedClaim objects.
        """
        sentences = sentence_split(report_text)
        claims = []
        for sent in sentences:
            category = classify_claim_pathology(sent)
            claims.append(DecomposedClaim(
                text=sent,
                pathology_category=category,
                confidence=0.5,  # lower confidence for fallback
                source_sentence=sent,
            ))
        return claims


class SentenceDecomposer:
    """Simple sentence-level decomposer without an LLM.

    Splits reports into sentences and classifies each by pathology keywords.
    Use this when LLM inference is too expensive or for quick prototyping.
    """

    def decompose(self, report_text: str) -> list[DecomposedClaim]:
        """Decompose report by sentence splitting + keyword classification.

        Args:
            report_text: Full report text.

        Returns:
            List of DecomposedClaim objects.
        """
        if not report_text or not report_text.strip():
            return []

        sentences = sentence_split(report_text)
        claims = []

        for sent in sentences:
            category = classify_claim_pathology(sent)
            claims.append(DecomposedClaim(
                text=sent,
                pathology_category=category,
                confidence=0.6,
                source_sentence=sent,
            ))

        return claims


def merge_low_confidence_claims(
    claims: list[DecomposedClaim],
    threshold: float = 0.5,
) -> list[DecomposedClaim]:
    """Merge adjacent claims with low confidence into coarser units.

    When decomposer confidence is below threshold for a claim boundary,
    adjacent claims are merged rather than split incorrectly.

    Args:
        claims: List of claims from decomposer.
        threshold: Confidence threshold below which to merge.

    Returns:
        List of claims with low-confidence ones merged.
    """
    if len(claims) <= 1:
        return claims

    merged = [claims[0]]
    for claim in claims[1:]:
        if claim.confidence < threshold and merged[-1].confidence < threshold:
            # Merge with previous claim
            merged_text = merged[-1].text + " " + claim.text
            # Keep the category of the more specific (non-Rare/Other) claim
            category = (
                claim.pathology_category
                if merged[-1].pathology_category == "Rare/Other"
                else merged[-1].pathology_category
            )
            merged[-1] = DecomposedClaim(
                text=merged_text,
                pathology_category=category,
                confidence=max(merged[-1].confidence, claim.confidence),
                source_sentence=merged_text,
            )
        else:
            merged.append(claim)

    return merged


def decompose_report(
    report_text: str,
    decomposer: Optional[ZeroShotDecomposer | SentenceDecomposer] = None,
    merge_threshold: float = 0.5,
) -> list[DecomposedClaim]:
    """Main entry point for report decomposition.

    Args:
        report_text: Full report text.
        decomposer: Decomposer instance (creates SentenceDecomposer if None).
        merge_threshold: Confidence threshold for merging claims.

    Returns:
        List of DecomposedClaim objects.
    """
    if decomposer is None:
        decomposer = SentenceDecomposer()

    claims = decomposer.decompose(report_text)
    claims = merge_low_confidence_claims(claims, threshold=merge_threshold)

    logger.debug(f"Decomposed report into {len(claims)} claims")
    return claims
