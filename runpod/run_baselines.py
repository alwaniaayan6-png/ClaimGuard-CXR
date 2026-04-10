"""Run all text-only baselines on RunPod (standalone, no Modal).

Baselines:
  1. Hypothesis-only DeBERTa (tests for surface shortcuts)
  2. DeBERTa-v3-large zero-shot NLI (off-the-shelf, no fine-tuning)
  3. RadFlag-style self-consistency (CPU, keyword overlap)

Usage:
    python run_baselines.py \
        --test-claims /workspace/data/eval_data/test_claims.json \
        --output-dir /workspace/results/baselines
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BASELINE 1: Hypothesis-only (no evidence)
# ============================================================
def run_hypothesis_only(
    training_data_path: str,
    test_claims_path: str,
    output_path: str,
    model_name: str = "microsoft/deberta-v3-large",
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum: int = 4,
    lr: float = 1e-5,
    seed: int = 42,
):
    """Train and evaluate hypothesis-only baseline."""
    print("\n" + "=" * 60)
    print("BASELINE 1: Hypothesis-only (no evidence)")
    print("=" * 60)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class HypOnlyDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(
                item["claim"], max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            label = 1 if item["label"] == 1 else 0
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    class HypModel(nn.Module):
        def __init__(self, model_name, num_classes=2):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            dim = self.encoder.config.hidden_size
            self.head = nn.Sequential(
                nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.head(out.last_hidden_state[:, 0, :])

    # Load training data
    with open(training_data_path) as f:
        all_data = json.load(f)

    # Patient-stratified split
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}")) for i, item in enumerate(all_data)})
    val_rng = random.Random(seed + 1)
    n_val = max(1, int(len(pid_set) * 0.10))
    val_patients = set(val_rng.sample(pid_set, n_val))
    train_ex = [item for i, item in enumerate(all_data) if str(item.get("patient_id", f"_idx{i}")) not in val_patients]
    val_ex = [item for i, item in enumerate(all_data) if str(item.get("patient_id", f"_idx{i}")) in val_patients]

    train_ds = HypOnlyDataset(train_ex, tokenizer)
    val_ds = HypOnlyDataset(val_ex, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    model = HypModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), max(total_steps, 1))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"HypOnly E{epoch+1}")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(model(ids, mask), labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                preds.extend(logits.argmax(-1).cpu().tolist())
                labs.extend(batch["label"].tolist())
        val_acc = accuracy_score(labs, preds)
        print(f"  E{epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")
        best_val_acc = max(best_val_acc, val_acc)

    # Evaluate on test set
    with open(test_claims_path) as f:
        test_data = json.load(f)
    test_ds = HypOnlyDataset(test_data, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    model.eval()
    preds, labs, probs_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="HypOnly test"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(batch["label"].tolist())
            probs_list.extend(p)

    labs = np.array(labs)
    preds = np.array(preds)
    probs_arr = np.array(probs_list)

    test_acc = float(accuracy_score(labs, preds))
    contra_recall = float((preds[labs == 1] == 1).mean()) if labs.sum() > 0 else 0.0
    try:
        auroc = float(roc_auc_score(labs, probs_arr))
    except ValueError:
        auroc = 0.5

    result = {
        "method": "Hypothesis-only DeBERTa",
        "accuracy": test_acc,
        "contra_recall": contra_recall,
        "auroc": auroc,
        "best_val_acc": best_val_acc,
        "interpretation": _interpret(test_acc),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nHypothesis-only: acc={test_acc:.4f}, contra_recall={contra_recall:.4f}")
    print(f"Interpretation: {result['interpretation']}")
    return result


def _interpret(acc):
    if acc < 0.70: return "STRONG: verifier uses evidence, not shortcuts"
    elif acc < 0.80: return "MILD artifacts: some perturbations detectable from claim alone (expected)"
    elif acc < 0.90: return "MODERATE artifacts: consider adversarial perturbations"
    else: return "SEVERE artifacts: task too easy without evidence"


# ============================================================
# BASELINE 2: DeBERTa zero-shot NLI
# ============================================================
def run_deberta_nli_zeroshot(
    test_claims_path: str,
    output_path: str,
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    batch_size: int = 32,
):
    """Run off-the-shelf DeBERTa NLI without any fine-tuning."""
    print("\n" + "=" * 60)
    print("BASELINE 2: DeBERTa-v3-large zero-shot NLI")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    with open(test_claims_path) as f:
        claims = json.load(f)

    class NLIDataset(Dataset):
        def __init__(self, claims, tokenizer):
            self.claims = claims
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.claims)

        def __getitem__(self, idx):
            item = self.claims[idx]
            evidence = " ".join(item.get("evidence", [])[:2])
            enc = self.tokenizer(evidence, item["claim"], max_length=512,
                                 padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": 1 if item["label"] == 1 else 0,
            }

    ds = NLIDataset(claims, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DeBERTa NLI"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).logits
            probs = F.softmax(logits, dim=-1)
            contra_prob = probs[:, 0].cpu().numpy()  # label 0 = contradiction in NLI
            preds = (contra_prob > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch["label"])
            all_probs.extend(contra_prob)

    labs = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    acc = float(accuracy_score(labs, preds))
    contra_recall = float((preds[labs == 1] == 1).mean()) if labs.sum() > 0 else 0.0
    try:
        auroc = float(roc_auc_score(labs, probs))
    except ValueError:
        auroc = 0.5

    result = {"method": "DeBERTa-v3-large zero-shot NLI", "accuracy": acc,
              "contra_recall": contra_recall, "auroc": auroc, "n_claims": len(claims)}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDeBERTa NLI zero-shot: acc={acc:.4f}, contra_recall={contra_recall:.4f}, auroc={auroc:.4f}")
    return result


# ============================================================
# BASELINE 3: RadFlag-style self-consistency (CPU)
# ============================================================
def run_radflag_consistency(test_claims_path: str, output_path: str):
    """Self-consistency baseline via keyword overlap variance."""
    print("\n" + "=" * 60)
    print("BASELINE 3: RadFlag-style self-consistency")
    print("=" * 60)

    with open(test_claims_path) as f:
        claims = json.load(f)

    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                 "had", "her", "was", "one", "our", "out", "has", "with", "this",
                 "that", "from", "have", "been", "were", "they", "than", "each"}
    scores, labels = [], []

    for item in tqdm(claims, desc="RadFlag"):
        claim_words = set(re.findall(r'\b[a-z]{3,}\b', item["claim"].lower())) - stopwords
        evidence_words = set(" ".join(item.get("evidence", [])).lower().split())
        if not claim_words:
            scores.append(0.5)
        else:
            base = len(claim_words & evidence_words) / len(claim_words)
            rng = random.Random(42 + hash(item["claim"]) % 10000)
            variants = []
            for _ in range(5):
                n_drop = rng.randint(1, min(2, len(claim_words)))
                dropped = set(rng.sample(list(claim_words), n_drop))
                vw = claim_words - dropped or claim_words
                variants.append(len(vw & evidence_words) / len(vw))
            var = float(np.var(variants)) if len(variants) > 1 else 0
            scores.append(0.5 * base + 0.5 * (1.0 - min(1.0, var * 10)))
        labels.append(1 if item["label"] == 1 else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    threshold = np.median(scores)
    preds = (scores < threshold).astype(int)

    acc = float(accuracy_score(labels, preds))
    contra_recall = float((preds[labels == 1] == 1).mean()) if labels.sum() > 0 else 0.0
    try:
        auroc = float(roc_auc_score(labels, -scores))
    except ValueError:
        auroc = 0.5

    result = {"method": "RadFlag self-consistency", "accuracy": acc,
              "contra_recall": contra_recall, "auroc": auroc, "n_claims": len(labels)}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRadFlag: acc={acc:.4f}, contra_recall={contra_recall:.4f}")
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", default="/workspace/data/verifier_training_data_v2.json")
    parser.add_argument("--test-claims", default="/workspace/data/eval_data/test_claims.json")
    parser.add_argument("--output-dir", default="/workspace/results/baselines")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Hypothesis-only
    run_hypothesis_only(
        args.training_data, args.test_claims,
        f"{args.output_dir}/hypothesis_only.json",
    )

    # 2. DeBERTa NLI zero-shot
    run_deberta_nli_zeroshot(
        args.test_claims,
        f"{args.output_dir}/deberta_nli_zeroshot.json",
    )

    # 3. RadFlag consistency
    run_radflag_consistency(
        args.test_claims,
        f"{args.output_dir}/radflag_consistency.json",
    )

    print("\n" + "=" * 60)
    print("ALL BASELINES COMPLETE")
    print("=" * 60)
    for f_name in os.listdir(args.output_dir):
        if f_name.endswith(".json"):
            with open(os.path.join(args.output_dir, f_name)) as f:
                r = json.load(f)
            print(f"  {r['method']}: acc={r['accuracy']:.4f}, contra_recall={r['contra_recall']:.4f}")


if __name__ == "__main__":
    main()
