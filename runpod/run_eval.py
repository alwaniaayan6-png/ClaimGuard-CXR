"""Full evaluation pipeline for ClaimGuard-CXR v2 on RunPod (standalone).

Supports both v1 (RoBERTa+heatmap) and v2 (DeBERTa clean) checkpoints.
Runs: verify -> temperature scale -> conformal FDR -> metrics.

Usage:
    python run_eval.py \
        --checkpoint /workspace/checkpoints/progressive_nli/final/best_verifier.pt \
        --cal-claims /workspace/data/eval_data/calibration_claims.json \
        --test-claims /workspace/data/eval_data/test_claims.json \
        --output-dir /workspace/results/eval_deberta_v2 \
        --model-type v2
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================
# Model definitions
# ============================================================
class V2Verifier(nn.Module):
    """DeBERTa-v3-large binary verifier (v2 architecture)."""
    def __init__(self, model_name="microsoft/deberta-v3-large", num_classes=2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        dim = self.text_encoder.config.hidden_size
        self.verdict_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.verdict_head(cls)

    def forward_with_hidden(self, input_ids, attention_mask):
        """Return logits AND penultimate hidden (for CoFact)."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        h = self.verdict_head[0](cls)  # Linear(dim, 256)
        h = self.verdict_head[1](h)    # ReLU
        logits = self.verdict_head[3](self.verdict_head[2](h))  # Dropout -> Linear(256, 2)
        return logits, h.detach()


class V1Verifier(nn.Module):
    """RoBERTa-large + heatmap binary verifier (v1 architecture)."""
    def __init__(self, model_name="roberta-large", heatmap_dim=768, num_classes=2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        dim = self.text_encoder.config.hidden_size

        # Heatmap encoder (receives zeros in practice)
        self.heatmap_encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.heatmap_encoder_proj = nn.Linear(128, heatmap_dim)
        fused_dim = dim + heatmap_dim

        self.verdict_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes),
        )
        self.score_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1),
        )
        self.contrastive_proj = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 128),
        )

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        hmap = torch.zeros(cls.shape[0], self.heatmap_encoder_proj.out_features,
                           device=cls.device, dtype=cls.dtype)
        fused = torch.cat([cls, hmap], dim=-1)
        return self.verdict_head(fused)


class TemperatureScaling(nn.Module):
    def __init__(self, init_t=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_t))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.temperature.item())


# ============================================================
# Eval dataset
# ============================================================
class EvalDataset(Dataset):
    def __init__(self, claims, tokenizer, max_length=512):
        self.claims = claims
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        item = self.claims[idx]
        claim = item["claim"]
        evidence = " [SEP] ".join(item.get("evidence", [])[:2])
        enc = self.tokenizer(claim, evidence, max_length=self.max_length,
                             padding="max_length", truncation="only_second",
                             return_tensors="pt")
        label = 1 if item["label"] == 1 else 0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label,
            "patient_id": str(item.get("patient_id", "")),
        }


# ============================================================
# Conformal FDR
# ============================================================
def inverted_cfbh(cal_contra_scores, test_scores, alpha):
    """Inverted conformal BH: calibrate on contradicted claims."""
    n_cal = len(cal_contra_scores)
    n_test = len(test_scores)

    p_values = np.array([(np.sum(cal_contra_scores >= s) + 1) / (n_cal + 1) for s in test_scores])

    sorted_pvals = np.sort(p_values)
    k_star = 0
    for k in range(1, n_test + 1):
        if sorted_pvals[k - 1] <= k * alpha / n_test:
            k_star = k

    bh_threshold = sorted_pvals[k_star - 1] if k_star > 0 else 0.0
    green_mask = p_values <= bh_threshold if k_star > 0 else np.zeros(n_test, dtype=bool)
    return green_mask, p_values, k_star


# ============================================================
# Main eval
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cal-claims", required=True)
    parser.add_argument("--test-claims", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="v2", choices=["v1", "v2"])
    parser.add_argument("--model-name", default=None,
                        help="HF model name. Default: deberta-v3-large for v2, roberta-large for v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_name is None:
        args.model_name = "microsoft/deberta-v3-large" if args.model_type == "v2" else "roberta-large"

    print(f"Model: {args.model_name} ({args.model_type})")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model
    if args.model_type == "v2":
        model = V2Verifier(args.model_name).to(device)
    else:
        model = V1Verifier(args.model_name).to(device)

    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"WARNING: No checkpoint at {args.checkpoint}")

    model.eval()

    # Load data
    with open(args.cal_claims) as f:
        cal_data = json.load(f)
    with open(args.test_claims) as f:
        test_data = json.load(f)
    print(f"Calibration: {len(cal_data)}, Test: {len(test_data)}")

    # Run inference
    def get_scores(data, desc):
        ds = EvalDataset(data, tokenizer)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                all_logits.append(logits.cpu())
                all_labels.extend([l for l in batch["label"]])
        return torch.cat(all_logits, dim=0), np.array(all_labels)

    print("\nRunning calibration inference...")
    cal_logits, cal_labels = get_scores(cal_data, "Cal")
    print("Running test inference...")
    test_logits, test_labels = get_scores(test_data, "Test")

    # Temperature scaling on calibration set
    print("\nFitting temperature scaling...")
    temp_scaler = TemperatureScaling()
    opt_T = temp_scaler.fit(cal_logits, torch.tensor(cal_labels, dtype=torch.long))
    print(f"Optimal temperature: {opt_T:.4f}")

    # Apply temperature to get calibrated scores
    with torch.no_grad():
        cal_probs = F.softmax(temp_scaler(cal_logits), dim=-1).numpy()
        test_probs = F.softmax(temp_scaler(test_logits), dim=-1).numpy()

    cal_scores = cal_probs[:, 0]  # P(Not-Contradicted)
    test_scores = test_probs[:, 0]

    # Classification metrics
    test_preds = (test_scores < 0.5).astype(int)  # < 0.5 -> contradicted
    acc = float(accuracy_score(test_labels, test_preds))
    macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    contra_prec = float(precision_score(test_labels, test_preds, pos_label=1, zero_division=0))
    contra_rec = float(recall_score(test_labels, test_preds, pos_label=1, zero_division=0))
    contra_f1 = float(f1_score(test_labels, test_preds, pos_label=1, zero_division=0))

    try:
        auroc = float(roc_auc_score(test_labels, 1 - test_scores))
        ap = float(average_precision_score(test_labels, 1 - test_scores))
    except ValueError:
        auroc, ap = 0.5, 0.5

    # ECE
    ece = _compute_ece(test_scores, test_labels, n_bins=10)

    print(f"\n=== Classification Results ===")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Contra Recall:  {contra_rec:.4f}")
    print(f"AUROC:          {auroc:.4f}")
    print(f"ECE:            {ece:.4f}")

    # Conformal FDR at multiple alpha levels
    cal_contra_scores = cal_scores[cal_labels == 1]
    print(f"\nCalibration contradicted claims: {len(cal_contra_scores)}")

    conformal_results = {}
    for alpha in [0.01, 0.05, 0.10, 0.15, 0.20]:
        green_mask, p_values, k_star = inverted_cfbh(cal_contra_scores, test_scores, alpha)
        n_green = int(green_mask.sum())
        fdr = float(test_labels[green_mask].mean()) if n_green > 0 else 0.0
        power = float(green_mask[test_labels == 0].mean()) if (test_labels == 0).sum() > 0 else 0.0

        conformal_results[str(alpha)] = {
            "n_green": n_green, "fdr": fdr, "power": power,
            "coverage": n_green / len(test_labels),
        }
        print(f"alpha={alpha}: n_green={n_green}, FDR={fdr:.4f}, power={power:.4f}")

    # Save results
    results = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "classification": {
            "accuracy": acc, "macro_f1": macro_f1,
            "contra_precision": contra_prec, "contra_recall": contra_rec,
            "contra_f1": contra_f1, "auroc": auroc, "ap": ap, "ece": ece,
        },
        "conformal": conformal_results,
        "temperature": opt_T,
        "n_cal": len(cal_data), "n_test": len(test_data),
        "n_cal_contra": int(len(cal_contra_scores)),
    }

    output_path = os.path.join(args.output_dir, "full_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    np.savez(
        os.path.join(args.output_dir, "predictions.npz"),
        test_scores=test_scores, test_labels=test_labels,
        test_preds=test_preds, cal_scores=cal_scores, cal_labels=cal_labels,
    )

    print(f"\nResults saved to {output_path}")
    return results


def _compute_ece(scores, labels, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = (labels[mask] == 0).mean()  # P(not-contra) should match score
        bin_conf = scores[mask].mean()
        ece += mask.sum() / len(scores) * abs(bin_acc - bin_conf)
    return float(ece)


if __name__ == "__main__":
    main()
