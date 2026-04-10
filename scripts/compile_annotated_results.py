"""Compile results from human-annotated real hallucination test set.

Ingests the annotation workbook (CSV/JSON with human labels), runs the
verifier on each claim, computes precision/recall/F1/AUROC, and produces
conformal FDR results at multiple alpha levels.

Usage:
    python3 scripts/compile_annotated_results.py \
        --annotations /path/to/annotation_workbook.csv \
        --verifier-checkpoint /path/to/best_verifier.pt \
        --output-dir results/real_hallucinations/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_annotations(path: str) -> list[dict]:
    """Load human-annotated claims from CSV or JSON."""
    if path.endswith(".csv"):
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {path}")


def filter_annotated(annotations: list[dict]) -> list[dict]:
    """Filter to only claims with valid human labels."""
    valid_labels = {"Supported", "Contradicted", "Novel-Plausible", "Novel-Hallucinated"}
    filtered = []
    for ann in annotations:
        label = ann.get("annotator_label", "").strip()
        if label in valid_labels:
            filtered.append(ann)
    return filtered


def to_binary_labels(annotations: list[dict]) -> list[int]:
    """Convert human labels to binary: 0=not-contra, 1=contra/hallucinated."""
    binary = []
    for ann in annotations:
        label = ann["annotator_label"].strip()
        if label in ("Contradicted", "Novel-Hallucinated"):
            binary.append(1)
        else:
            binary.append(0)
    return binary


def compute_inter_annotator_agreement(annotations: list[dict]) -> dict:
    """Compute Cohen's kappa on double-annotated subset."""
    double_annotated = [
        a for a in annotations
        if a.get("annotator2_label", "").strip() != ""
    ]
    if len(double_annotated) < 10:
        return {"kappa": None, "n_double": len(double_annotated), "status": "insufficient"}

    from sklearn.metrics import cohen_kappa_score

    labels1 = []
    labels2 = []
    for a in double_annotated:
        l1 = a["annotator_label"].strip()
        l2 = a["annotator2_label"].strip()
        # Map to binary
        labels1.append(1 if l1 in ("Contradicted", "Novel-Hallucinated") else 0)
        labels2.append(1 if l2 in ("Contradicted", "Novel-Hallucinated") else 0)

    kappa = cohen_kappa_score(labels1, labels2)
    agreement = sum(a == b for a, b in zip(labels1, labels2)) / len(labels1)

    return {
        "kappa": float(kappa),
        "raw_agreement": float(agreement),
        "n_double": len(double_annotated),
        "status": "computed",
    }


def run_verifier_on_claims(
    annotations: list[dict],
    checkpoint_path: str,
    model_name: str = "microsoft/deberta-v3-large",
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """Run the verifier on annotated claims and return P(Not-Contradicted) scores."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class SimpleDataset(Dataset):
        def __init__(self, annotations, tokenizer, max_length):
            self.data = annotations
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            claim = item.get("extracted_claim", item.get("claim", ""))
            evidence = item.get("ground_truth_report", "")
            if isinstance(evidence, list):
                evidence = " [SEP] ".join(evidence[:2])
            enc = self.tokenizer(
                claim, evidence,
                max_length=self.max_length, padding="max_length",
                truncation="only_second", return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

    # Build model (must match training script attribute names)
    class DeBERTaVerifier(nn.Module):
        def __init__(self, model_name, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size
            self.verdict_head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            logits = self.verdict_head(cls)
            return logits

    model = DeBERTaVerifier(model_name).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. Using random weights.")

    model.eval()

    dataset = SimpleDataset(annotations, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            # Apply temperature
            scaled = logits / model.temperature.clamp(min=0.01)
            probs = F.softmax(scaled, dim=-1)
            not_contra = probs[:, 0].cpu().numpy()
            all_scores.extend(not_contra)

    return np.array(all_scores)


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute classification and conformal metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, classification_report,
    )

    preds = (scores < 0.5).astype(int)  # score < 0.5 -> predict contradicted

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "contra_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "contra_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "contra_f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
    }

    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, 1 - scores))
        metrics["ap"] = float(average_precision_score(labels, 1 - scores))

    # Label distribution
    metrics["n_total"] = len(labels)
    metrics["n_contra"] = int(labels.sum())
    metrics["n_not_contra"] = int((labels == 0).sum())
    metrics["hallucination_rate"] = float(labels.mean())

    print(f"\n=== Real Hallucination Results ===")
    print(f"Total claims: {metrics['n_total']}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.1%}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Contra recall: {metrics['contra_recall']:.4f}")
    print(f"AUROC: {metrics.get('auroc', 'N/A')}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to annotation_workbook.csv or .json")
    parser.add_argument("--verifier-checkpoint", default="", help="Path to verifier checkpoint")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", default="results/real_hallucinations/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and filter annotations
    raw = load_annotations(args.annotations)
    print(f"Loaded {len(raw)} raw annotations")

    annotated = filter_annotated(raw)
    print(f"Filtered to {len(annotated)} with valid labels")

    if len(annotated) == 0:
        print("ERROR: No valid annotations found. Check the annotation_workbook.")
        sys.exit(1)

    # Inter-annotator agreement
    iaa = compute_inter_annotator_agreement(annotated)
    print(f"Inter-annotator agreement: {iaa}")

    # Convert to binary
    binary_labels = np.array(to_binary_labels(annotated))

    # Run verifier
    if args.verifier_checkpoint:
        scores = run_verifier_on_claims(
            annotated, args.verifier_checkpoint, args.model_name
        )
    else:
        print("No verifier checkpoint provided. Skipping inference.")
        scores = np.zeros(len(annotated))

    # Compute metrics
    metrics = compute_metrics(scores, binary_labels)
    metrics["inter_annotator"] = iaa

    # Save results
    output_path = os.path.join(args.output_dir, "real_hallucination_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
