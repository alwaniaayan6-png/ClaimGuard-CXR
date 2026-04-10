"""Verifier training script for ClaimGuard-CXR.

Local training wrapper — the actual GPU training runs via
scripts/modal_train_verifier.py on Modal.

This module provides helpers for data preparation, checkpoint loading,
and evaluation of the trained verifier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_trained_verifier(
    checkpoint_path: str | Path,
    model_name: str = "microsoft/deberta-v3-large",
    device: str = "cpu",
) -> "ClaimVerifier":
    """Load a trained verifier from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_name: Base model ID (must match training).
        device: Device to load onto.

    Returns:
        ClaimVerifier instance with loaded weights.
    """
    from .claim_verifier import ClaimVerifier

    model = ClaimVerifier(model_name=model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    logger.info(f"Loaded verifier from {checkpoint_path}")
    return model


def evaluate_verifier(
    model: "ClaimVerifier",
    eval_data_path: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Evaluate a trained verifier on held-out data.

    Args:
        model: Trained ClaimVerifier.
        eval_data_path: Path to evaluation data JSON.
        device: Device.
        batch_size: Evaluation batch size.

    Returns:
        Dict with accuracy, per-class metrics, calibration stats.
    """
    from sklearn.metrics import classification_report, accuracy_score

    with open(eval_data_path) as f:
        eval_data = json.load(f)

    all_preds = []
    all_labels = []
    all_scores = []

    model.eval()

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        claims = [ex["claim"] for ex in batch]
        evidence_list = [ex.get("evidence", []) for ex in batch]
        labels = [ex["label"] for ex in batch]

        encoding = model.batch_encode(claims, evidence_list)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            output = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                apply_temperature=True,
            )

        preds = output.verdict_logits.argmax(dim=-1).cpu().tolist()
        scores = output.faithfulness_score.cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_scores.extend(scores)

    report = classification_report(
        all_labels, all_preds,
        target_names=model.VERDICT_LABELS,
        output_dict=True,
    )

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "classification_report": report,
        "n_samples": len(all_labels),
        "mean_faithfulness_score": float(np.mean(all_scores)),
    }
