"""Evidence-Conditioned Multimodal Claim Verifier for ClaimGuard-CXR (Contribution 1).

DeBERTa-v3-large cross-encoder for text (claim + evidence) fused with a CNN
heatmap encoder for visual grounding. The text encoder processes
[claim; SEP; evidence_1; SEP; evidence_2] while the heatmap encoder processes
the 27x27 spatial attention map. Both are fused before the verdict head.

Trained with CE primary loss + InfoNCE auxiliary loss using 8 types of
clinically motivated hard negatives.
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
    """Output from the claim verifier."""
    verdict_logits: torch.Tensor    # (batch, 3) — Supported/Contradicted/Insufficient
    verdict_probs: torch.Tensor     # (batch, 3) — softmax probabilities
    faithfulness_score: torch.Tensor  # (batch,) — continuous score in [0, 1]
    cls_embedding: torch.Tensor     # (batch, fused_dim) — for contrastive loss


class HeatmapEncoder(nn.Module):
    """Small CNN that encodes a 27x27 grounding heatmap into a feature vector.

    Architecture: 3 conv layers with batch norm and ReLU, followed by
    adaptive average pooling and a linear projection.

    Input: (batch, 1, 27, 27) heatmap
    Output: (batch, output_dim) feature vector

    Total params: ~1.2M (with output_dim=768)
    """

    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 27x27 -> 13x13
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 13x13 -> 1x1
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Encode heatmap to feature vector.

        Args:
            heatmap: (batch, 27, 27) or (batch, 1, 27, 27) attention heatmap.

        Returns:
            (batch, output_dim) feature vector.
        """
        if heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(1)  # (B, 1, 27, 27)
        x = self.conv(heatmap)  # (B, 128, 1, 1)
        x = x.flatten(1)       # (B, 128)
        return self.proj(x)     # (B, output_dim)


class ClaimVerifier(nn.Module):
    """Multimodal claim verifier: DeBERTa text cross-encoder + CNN heatmap encoder.

    The text branch processes [CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]
    through DeBERTa-v3-large, producing a 1024-dim [CLS] embedding.

    The vision branch processes the 27x27 grounding heatmap through a 3-layer CNN,
    producing a 768-dim feature vector.

    Both are concatenated (1024+768=1792) and fed through the verdict + score heads.
    This gives the verifier direct access to spatial image information while keeping
    the text encoder well-calibrated (DeBERTa cross-encoders have better calibration
    than VLMs, which matters for conformal prediction).

    Args:
        model_name: HuggingFace model ID for the text cross-encoder.
        heatmap_dim: Output dimension of the heatmap encoder.
        num_classes: Number of verdict classes (default 3).
        head_hidden_dim: Hidden dimension for classification heads.
        dropout: Dropout rate for heads.
        temperature: Learned temperature for calibration (initialized to 1.0).
    """

    VERDICT_LABELS = ["Supported", "Contradicted", "Insufficient Evidence"]

    def __init__(
        self,
        model_name: str = "roberta-large",
        heatmap_dim: int = 768,
        num_classes: int = 3,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Text branch: DeBERTa cross-encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size  # 1024 for deberta-v3-large

        # Vision branch: CNN heatmap encoder
        self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)

        # Fused dimension
        fused_dim = text_dim + heatmap_dim  # 1024 + 768 = 1792

        # Verdict classification head: fused -> 3-class
        self.verdict_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        # Continuous faithfulness score head: fused -> scalar in [0, 1]
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        # Projection for contrastive loss (InfoNCE auxiliary)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 128),
        )

        # Learnable temperature for post-hoc calibration
        self.temperature = nn.Parameter(torch.tensor(temperature))

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"ClaimVerifier: {total_params/1e6:.1f}M total params, "
            f"{trainable_params/1e6:.1f}M trainable. "
            f"Text: {text_dim}d, Heatmap: {heatmap_dim}d, Fused: {fused_dim}d"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_temperature: bool = False,
    ) -> VerifierOutput:
        """Forward pass through text + vision branches.

        Args:
            input_ids: Tokenized text input (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            heatmap: Grounding heatmap (batch, 27, 27) or None. If None, uses zeros.
            token_type_ids: Optional token type IDs.
            apply_temperature: Whether to apply learned temperature scaling.

        Returns:
            VerifierOutput with verdict logits, probabilities, score, and fused embedding.
        """
        # Text branch
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_cls = text_outputs.last_hidden_state[:, 0, :]  # (batch, 1024)

        # Vision branch
        if heatmap is not None:
            heatmap_features = self.heatmap_encoder(heatmap)  # (batch, 768)
        else:
            # No heatmap available — use zeros (text-only fallback)
            batch_size = input_ids.shape[0]
            hmap_dim = self.heatmap_encoder.proj.out_features
            heatmap_features = torch.zeros(
                batch_size, hmap_dim, device=input_ids.device, dtype=text_cls.dtype
            )

        # Fuse text + vision
        fused = torch.cat([text_cls, heatmap_features], dim=-1)  # (batch, 1792)

        # Verdict logits
        verdict_logits = self.verdict_head(fused)  # (batch, 3)

        # Temperature scaling
        if apply_temperature:
            scaled_logits = verdict_logits / self.temperature.clamp(min=0.01)
        else:
            scaled_logits = verdict_logits

        verdict_probs = F.softmax(scaled_logits, dim=-1)  # (batch, 3)

        # Faithfulness score: P(Supported)
        faithfulness_score = verdict_probs[:, 0]

        return VerifierOutput(
            verdict_logits=verdict_logits,
            verdict_probs=verdict_probs,
            faithfulness_score=faithfulness_score,
            cls_embedding=fused,
        )

    def encode_claim_evidence(
        self,
        claim: str,
        evidence_passages: list[str],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a claim with its evidence passages.

        Format: [CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]

        Args:
            claim: The claim text.
            evidence_passages: List of evidence passage strings (max 2).
            max_length: Maximum token length.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        evidence_text = " [SEP] ".join(evidence_passages[:2])

        encoding = self.tokenizer(
            claim,
            evidence_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding

    def batch_encode(
        self,
        claims: list[str],
        evidence_list: list[list[str]],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Batch-encode multiple claim-evidence pairs.

        Args:
            claims: List of claim texts.
            evidence_list: List of evidence passage lists (one per claim).
            max_length: Maximum token length.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors (batch, seq_len).
        """
        texts_a = claims
        texts_b = [" [SEP] ".join(evs[:2]) for evs in evidence_list]

        encoding = self.tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding

    def predict(
        self,
        claim: str,
        evidence_passages: list[str],
        heatmap: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> dict:
        """Run single-sample inference.

        Args:
            claim: Claim text.
            evidence_passages: List of evidence passages.
            heatmap: Optional 27x27 grounding heatmap tensor.
            device: Device to run on.

        Returns:
            Dict with 'verdict', 'verdict_probs', 'faithfulness_score'.
        """
        self.eval()
        encoding = self.encode_claim_evidence(claim, evidence_passages)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        if heatmap is not None:
            heatmap = heatmap.unsqueeze(0).to(device) if heatmap.ndim == 2 else heatmap.to(device)

        with torch.no_grad():
            output = self.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                heatmap=heatmap,
                apply_temperature=True,
            )

        verdict_idx = output.verdict_probs[0].argmax().item()
        return {
            "verdict": self.VERDICT_LABELS[verdict_idx],
            "verdict_probs": {
                label: output.verdict_probs[0, i].item()
                for i, label in enumerate(self.VERDICT_LABELS)
            },
            "faithfulness_score": output.faithfulness_score[0].item(),
        }


class VerifierLoss(nn.Module):
    """Combined loss for verifier training: CE + InfoNCE.

    CE (cross-entropy) is the primary loss for verdict classification.
    InfoNCE (contrastive) is auxiliary for hard negative separation.

    Args:
        ce_weight: Weight for cross-entropy loss.
        infonce_weight: Weight for InfoNCE contrastive loss.
        infonce_temperature: Temperature for InfoNCE.
    """

    def __init__(
        self,
        ce_weight: float = 0.7,
        infonce_weight: float = 0.3,
        infonce_temperature: float = 0.07,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.infonce_weight = infonce_weight
        self.infonce_temperature = infonce_temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        verdict_logits: torch.Tensor,
        targets: torch.Tensor,
        fused_embeddings: torch.Tensor,
        contrastive_proj: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            verdict_logits: (batch, 3) raw logits from verdict head.
            targets: (batch,) integer class labels (0=Supported, 1=Contradicted, 2=Insufficient).
            fused_embeddings: (batch, fused_dim) fused text+vision embeddings for contrastive loss.
            contrastive_proj: Projection module for contrastive embeddings.

        Returns:
            Dict with 'total_loss', 'ce_loss', 'infonce_loss'.
        """
        # Cross-entropy loss (primary)
        ce = self.ce_loss(verdict_logits, targets)

        # InfoNCE contrastive loss (auxiliary)
        z = contrastive_proj(fused_embeddings)  # (batch, 128)
        z = F.normalize(z, dim=-1)

        sim = torch.mm(z, z.t()) / self.infonce_temperature  # (batch, batch)

        mask = torch.eye(len(targets), device=targets.device)
        labels_eq = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        labels_eq = labels_eq * (1 - mask)

        exp_sim = torch.exp(sim) * (1 - mask)
        pos_sim = exp_sim * labels_eq
        denom = exp_sim.sum(dim=1).clamp(min=1e-8)  # (batch,)
        pos_sum = pos_sim.sum(dim=1).clamp(min=1e-8)  # (batch,)

        has_positive = labels_eq.sum(dim=1) > 0
        if has_positive.any():
            infonce = -torch.log(pos_sum[has_positive] / denom[has_positive])
            infonce = infonce.mean()
        else:
            infonce = torch.tensor(0.0, device=targets.device)

        total = self.ce_weight * ce + self.infonce_weight * infonce

        return {
            "total_loss": total,
            "ce_loss": ce,
            "infonce_loss": infonce,
        }


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
