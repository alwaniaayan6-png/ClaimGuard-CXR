"""Canonical ``VerifierModel`` definition for ClaimGuard-CXR.

This module exists to prevent the D21 silent-load bug from spreading.
On 2026-04-15 we discovered that ``scripts/demo_provenance_gate_failure.py``,
``scripts/modal_train_dpo_refinement.py``, and
``data/augmentation/causal_term_identifier.py`` all had their own copies
of a verifier loader, and two of them (the trainer + causal-term ID)
were trying to load the v1/v3 checkpoint into a plain
``AutoModel + Linear(hidden, 2)`` layout.  The actual checkpoint is the
full ``VerifierModel`` (text_encoder + heatmap_encoder + verdict_head +
score_head + contrastive_proj), so ``strict=False`` silently dropped
all the head keys and the resulting ``Linear(1024, 2)`` head was random
init.  Task 9's sanity check caught the demo script's version of this
bug; the trainer + causal-term-ID copies were caught later by a
post-hoc Opus pre-flight reviewer before any GPU spend hit them.

This module is the SINGLE SOURCE OF TRUTH for:

* ``HeatmapEncoder``        — the 1-channel CNN with 768-dim projection
* ``VerifierModel``         — the full multimodal model used by every
                              ClaimGuard-CXR training and evaluation
                              script
* ``build_verifier_model``  — convenience constructor that pins
                              ``num_classes=2`` for binary verifier
                              checkpoints (v1/v3)
* ``load_verifier_checkpoint`` — load a state_dict into a fresh
                              ``VerifierModel``, with a hard-missing
                              key audit that REFUSES to silently
                              tolerate missing head weights

The architecture is byte-identical to the inline definition in
``scripts/modal_run_evaluation.py`` lines 105–162 — that file is the
historical reference and will be updated to import from here in a
follow-up commit (the demo script has already been verified to use a
matching definition; see ``scripts/demo_provenance_gate_failure.py``
``_build_verifier_model``).

Usage::

    from inference.verifier_model import (
        VerifierModel,
        build_verifier_model,
        load_verifier_checkpoint,
    )

    tokenizer, model = load_verifier_checkpoint(
        checkpoint_path="/data/checkpoints/verifier_binary_v3/best_verifier.pt",
        hf_backbone="roberta-large",
        device=torch.device("cuda"),
    )
    verdict_logits, sigmoid_score = model(input_ids, attention_mask)

The forward returns ``(verdict_logits, sigmoid_score)``.
``softmax(verdict_logits, dim=-1)[:, 0]`` is the canonical
"supported probability" used by every cfBH calibration and provenance-
gate decision in ClaimGuard.

Why not just import the loader from ``demo_provenance_gate_failure.py``:
that module is guarded by a top-level ``try: import modal as _modal``
which means importing it eagerly defines a Modal app, which we don't
want in non-Modal scripts (causal-term ID has its own Modal entry).
Keeping the architecture in a third file with no Modal dependency
breaks the dependency cycle.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture (matches scripts/modal_run_evaluation.py lines 105-162 exactly)
# ---------------------------------------------------------------------------


class HeatmapEncoder(nn.Module):
    """1-channel heatmap CNN with 768-dim projection.

    Used as the second input branch of ``VerifierModel``.  At Task 9
    inference time we feed an all-zero heatmap because text-only
    claim/evidence pairs don't have an attention map; the trained
    ``proj`` weights still contribute via the bias term.
    """

    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(1)
        return self.proj(self.conv(heatmap).flatten(1))


class VerifierModel(nn.Module):
    """Binary (or 3-class) ClaimGuard-CXR verifier.

    Forward signature::

        model(input_ids, attention_mask, heatmap=None) ->
            (verdict_logits, sigmoid_score)

    * ``verdict_logits`` shape ``(batch, num_classes)`` — softmax-
      friendly classification head over {not contradicted, contradicted}
      when ``num_classes=2``.
    * ``sigmoid_score`` shape ``(batch,)`` — separate regression head
      trained jointly.

    For Task 9 / cfBH calibration we read
    ``softmax(verdict_logits, dim=-1)[:, 0]`` as the supported-class
    probability.  The sigmoid_score is reserved for downstream
    threshold tuning and is not used by the conformal pipeline.
    """

    def __init__(
        self,
        model_name: str,
        heatmap_dim: int = 768,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)
        fused_dim = text_dim + heatmap_dim

        self.verdict_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.contrastive_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        text_cls = outputs.last_hidden_state[:, 0, :]
        if heatmap is not None:
            hmap_feat = self.heatmap_encoder(heatmap)
        else:
            hmap_feat = torch.zeros(
                text_cls.shape[0],
                self.heatmap_encoder.proj.out_features,
                device=text_cls.device,
                dtype=text_cls.dtype,
            )
        fused = torch.cat([text_cls, hmap_feat], dim=-1)
        verdict_logits = self.verdict_head(fused)
        score = torch.sigmoid(self.score_head(fused)).squeeze(-1)
        return verdict_logits, score


def build_verifier_model(
    hf_backbone: str = "roberta-large",
    num_classes: int = 2,
) -> VerifierModel:
    """Construct a fresh ``VerifierModel`` for inference or training.

    Args:
        hf_backbone: HF model name for the text encoder. Defaults to
            ``roberta-large`` to match v1/v3 training.
        num_classes: 2 for binary v1/v3/v4 checkpoints, 3 for legacy
            3-class. Defaults to 2.

    Returns:
        ``VerifierModel`` on CPU, in eval mode. Caller is responsible
        for ``.to(device)`` and ``.train()/.eval()``.
    """
    return VerifierModel(hf_backbone, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Checkpoint loader with hard-missing audit (D21 fix)
# ---------------------------------------------------------------------------


# Keys that may legitimately be missing on a healthy checkpoint:
#
# * ``text_encoder.pooler.*`` — HF strips the RoBERTa pooler when the
#   model is loaded via ``AutoModel.from_pretrained`` because we don't
#   use it (we read ``last_hidden_state[:, 0, :]`` directly).
# * ``text_encoder.embeddings.position_ids`` — registered buffer that
#   transformers >= 4.31 strips on save for RoBERTa models (they
#   moved it from a buffer to a runtime computation).  Adding to the
#   allow list per pre-flight reviewer Finding 2 (2026-04-15).
# * ``contrastive_proj.*`` — present in v3 training but not used at
#   inference time. Any downstream caller that needs the contrastive
#   head should load it explicitly.
#
# Any other missing key is a hard error: it means the checkpoint's
# architecture doesn't match VerifierModel and the loader is wrong.
_ALLOWED_MISSING_PREFIXES: tuple[str, ...] = (
    "text_encoder.pooler.",
    "text_encoder.embeddings.position_ids",
    "contrastive_proj.",
)


def load_verifier_checkpoint(
    checkpoint_path: str,
    hf_backbone: str,
    device: Any,
    num_classes: int = 2,
) -> tuple[Any, VerifierModel]:
    """Load a v1/v3/v4 binary VerifierModel checkpoint.

    Returns a ``(tokenizer, model)`` tuple. ``model`` is in
    ``.eval()`` mode on ``device``; switch to ``.train()`` if you're
    fine-tuning.

    Checkpoint layouts supported:

    * ``state == {"model_state_dict": {...VerifierModel keys...}, ...}``
      (v1, v3, v4 — the layout actually written by
      ``scripts/modal_train_verifier_binary.py``)
    * ``state == {...VerifierModel keys...}`` (flat dict)

    Legacy ``{"encoder": ..., "head": ...}`` layouts are NOT supported
    — that branch only worked for a pure ``AutoModel + Linear(hidden, 2)``
    architecture that no training script in this repo ever wrote, and
    its existence in the demo loader caused the D21 silent-load bug.

    Raises:
        ValueError: if the checkpoint is not a dict.
        RuntimeError: if any non-allowed key is missing after
            ``load_state_dict(strict=False)``. This is the explicit
            hard-error guard that catches the D21 bug pattern.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_backbone)
    model = build_verifier_model(hf_backbone, num_classes=num_classes).to(
        device,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is not a dict "
            f"(type={type(state).__name__}). Cannot load VerifierModel.",
        )
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state  # assume flat VerifierModel state

    # We use strict=False so we can categorize missing/unexpected keys
    # ourselves and raise on the unsafe categories. strict=True would
    # also raise but with a less informative message that lumps all
    # categories together.
    missing, unexpected = model.load_state_dict(sd, strict=False)

    hard_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in _ALLOWED_MISSING_PREFIXES)
    ]
    if hard_missing:
        raise RuntimeError(
            f"VerifierModel checkpoint load failed — "
            f"{len(hard_missing)} hard-missing keys after "
            f"load_state_dict. First 5: {hard_missing[:5]}. This "
            f"indicates the checkpoint's architecture does not match "
            f"the VerifierModel definition in inference/verifier_model.py. "
            f"Check scripts/modal_train_verifier_binary.py to see which "
            f"version of VerifierModel the checkpoint was trained against."
        )
    if unexpected:
        logger.warning(
            "VerifierModel load: %d unexpected keys (ignored). First 5: %s",
            len(unexpected), unexpected[:5],
        )
    if missing:
        logger.info(
            "VerifierModel load: %d allowed-missing keys "
            "(pooler/contrastive): %s",
            len(missing), missing[:5],
        )

    model.eval()
    return tokenizer, model


__all__ = [
    "HeatmapEncoder",
    "VerifierModel",
    "build_verifier_model",
    "load_verifier_checkpoint",
]
