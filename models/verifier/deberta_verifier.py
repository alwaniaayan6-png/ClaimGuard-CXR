"""DeBERTa-v3-large binary claim verifier for ClaimGuard-CXR v2.

Replaces RoBERTa-large from v1. Key improvements:
  - Disentangled attention (content-to-content, content-to-position, position-to-content)
  - Enhanced mask decoder uses absolute position in final layer
  - Consistently outperforms RoBERTa on NLI benchmarks by 2-5%
  - Clean 1024-dim CLS output (no unused heatmap branch)

Supports optional multimodal fusion via BiomedCLIP gating (see models/multimodal/).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class VerifierOutput:
    """Output from the DeBERTa claim verifier."""
    verdict_logits: torch.Tensor    # (batch, 2) — Not-Contradicted / Contradicted
    verdict_probs: torch.Tensor     # (batch, 2) — softmax probabilities
    not_contra_prob: torch.Tensor   # (batch,) — P(Not-Contradicted), conformal score
    cls_embedding: torch.Tensor     # (batch, hidden_dim) — for downstream fusion
    penultimate_hidden: torch.Tensor  # (batch, 256) — for CoFact density ratio estimation


class DeBERTaClaimVerifier(nn.Module):
    """Binary claim verifier using DeBERTa-v3-large as cross-encoder.

    Input:  [CLS] claim [SEP] evidence1 [SEP] evidence2 [SEP]
    Output: 2-class verdict (Not-Contradicted=0, Contradicted=1)

    Key design decisions (inherited from v1, validated):
      - Binary framing: 3-class confused Supported/Insufficient catastrophically
      - No label smoothing: 0.05 creates softmax ceiling that breaks conformal BH
      - No heatmap branch: v1's CNN received zeros; removed for v2 cleanliness

    Args:
        model_name: HuggingFace model ID for the text cross-encoder.
        num_classes: Number of verdict classes (default 2).
        head_hidden_dim: Hidden dimension for classification head.
        dropout: Dropout rate for head.
    """

    VERDICT_LABELS = ["Not-Contradicted", "Contradicted"]

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_classes: int = 2,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # DeBERTa-v3-large text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.text_encoder.config.hidden_size  # 1024 for large

        # Binary verdict head: CLS embedding -> 2-class
        self.verdict_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        # Learned temperature for post-hoc calibration
        self.temperature = nn.Parameter(torch.tensor(1.0))

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DeBERTaClaimVerifier: {total_params/1e6:.1f}M total, "
            f"{trainable_params/1e6:.1f}M trainable. Hidden: {self.hidden_dim}d"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_temperature: bool = False,
    ) -> VerifierOutput:
        """Forward pass through DeBERTa cross-encoder.

        Args:
            input_ids: Tokenized input (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            token_type_ids: Not used by DeBERTa v3 (type_vocab_size=0). Kept for API compat.
            apply_temperature: Whether to apply learned temperature scaling.

        Returns:
            VerifierOutput with verdict logits, probs, conformal score, CLS embedding.
        """
        # DeBERTa v3 has type_vocab_size=0 — do NOT pass token_type_ids
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 1024)

        # Extract penultimate hidden for CoFact density ratio estimation
        # verdict_head is Sequential: [Linear(1024,256), ReLU, Dropout, Linear(256,2)]
        # We want the 256-dim output after Linear+ReLU (before dropout and final linear)
        penultimate = self.verdict_head[0](cls_embedding)  # Linear(1024, 256)
        penultimate = self.verdict_head[1](penultimate)     # ReLU
        # Continue through rest of head for logits
        dropped = self.verdict_head[2](penultimate)         # Dropout
        verdict_logits = self.verdict_head[3](dropped)      # Linear(256, 2)

        if apply_temperature:
            scaled_logits = verdict_logits / self.temperature.clamp(min=0.01)
        else:
            scaled_logits = verdict_logits

        verdict_probs = F.softmax(scaled_logits, dim=-1)
        not_contra_prob = verdict_probs[:, 0]  # P(Not-Contradicted) = conformal score

        return VerifierOutput(
            verdict_logits=verdict_logits,
            verdict_probs=verdict_probs,
            not_contra_prob=not_contra_prob,
            cls_embedding=cls_embedding,
            penultimate_hidden=penultimate.detach(),  # detach: no grad needed for density est
        )

    def get_conformal_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get temperature-scaled P(Not-Contradicted) for conformal procedure.

        Returns:
            (batch,) tensor of scores in [0, 1].
        """
        output = self.forward(input_ids, attention_mask, apply_temperature=True)
        return output.not_contra_prob

    def encode_claim_evidence(
        self,
        claim: str,
        evidence_passages: list[str],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a claim with its evidence passages.

        Format: [CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]
        """
        evidence_text = " [SEP] ".join(evidence_passages[:2])
        encoding = self.tokenizer(
            claim,
            evidence_text,
            max_length=max_length,
            padding="max_length",
            truncation="only_second",  # H6 fix: preserve claim text
            return_tensors="pt",
        )
        return encoding

    def batch_encode(
        self,
        claims: list[str],
        evidence_list: list[list[str]],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Batch-encode multiple claim-evidence pairs."""
        texts_a = claims
        texts_b = [" [SEP] ".join(evs[:2]) for evs in evidence_list]
        encoding = self.tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
        )
        return encoding


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibrating verifier scores.

    Learns a single temperature parameter T on the calibration set to
    minimize NLL. Applied as: calibrated_logits = logits / T.
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Learn optimal temperature on calibration data."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        optimal_t = self.temperature.item()
        logger.info(f"Temperature scaling: optimal T = {optimal_t:.4f}")
        return optimal_t
