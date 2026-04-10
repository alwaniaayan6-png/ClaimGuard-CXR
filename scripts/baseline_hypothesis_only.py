"""Hypothesis-only baseline for ClaimGuard-CXR v2 artifact analysis.

Tests whether the verifier's 98% accuracy comes from genuine evidence
reasoning or from surface shortcuts in synthetic perturbations.

Method: Train a DeBERTa-v3-large verifier on the SAME 30K training data,
but with evidence passages replaced by padding. If this "claim-only"
model achieves significantly above 66% (majority class), the perturbations
contain exploitable lexical cues.

Expected outcome:
  - 70-80%: Some perturbations (negation, laterality) ARE detectable from
    the claim alone — this is VALID signal, not an artifact
  - >90%: Task too easy, perturbations have severe surface shortcuts
  - <75%: Strong evidence that the verifier genuinely uses evidence

Reference: Herlihy & Rudinger (ACL 2021) — hypothesis-only baselines for MedNLI

Usage:
    modal run --detach scripts/baseline_hypothesis_only.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-hypothesis-only")

hyp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=hyp_image,
    gpu="H100",
    timeout=60 * 60 * 3,
    volumes={"/data": vol},
)
def train_hypothesis_only(
    data_path: str = "/data/verifier_training_data.json",
    output_dir: str = "/data/checkpoints/hypothesis_only",
    model_name: str = "microsoft/deberta-v3-large",
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    gradient_accumulation: int = 4,
    num_epochs: int = 3,
    max_length: int = 128,  # Claims only — much shorter than 512
    seed: int = 42,
) -> dict:
    """Train claim-only verifier (no evidence) to test for surface shortcuts."""
    import json
    import os
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    with open(data_path) as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class HypothesisOnlyDataset(Dataset):
        """Dataset that provides ONLY the claim text, no evidence."""
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            claim = item["claim"]
            # KEY DIFFERENCE: No evidence! Just the claim.
            orig_label = item["label"]
            label = 1 if orig_label == 1 else 0

            encoding = self.tokenizer(
                claim,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Patient-stratified split (same as v1)
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}"))
                      for i, item in enumerate(training_data)})
    val_rng = random.Random(seed + 1)
    n_val = max(1, int(len(pid_set) * 0.10))
    val_patients = set(val_rng.sample(pid_set, n_val))
    train_ex = [item for i, item in enumerate(training_data)
                if str(item.get("patient_id", f"_idx{i}")) not in val_patients]
    val_ex = [item for i, item in enumerate(training_data)
              if str(item.get("patient_id", f"_idx{i}")) in val_patients]
    print(f"Split: {len(train_ex)} train, {len(val_ex)} val")

    train_ds = HypothesisOnlyDataset(train_ex, tokenizer, max_length)
    val_ds = HypothesisOnlyDataset(val_ex, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Simple classification head on DeBERTa CLS
    class HypothesisOnlyModel(nn.Module):
        def __init__(self, model_name, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.encoder.config.hidden_size
            self.head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.head(cls)

    model = HypothesisOnlyModel(model_name).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.10), total_steps
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    results = {"epoch_metrics": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels) / gradient_accumulation
            loss.backward()
            total_loss += loss.item() * gradient_accumulation

            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids, mask)
                val_loss_total += criterion(logits, labels).item() * ids.shape[0]
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_total / len(val_ex)
        val_acc = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=["Not-Contra", "Contradicted"],
            output_dict=True,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_contra_recall": report["Contradicted"]["recall"],
            "val_contra_f1": report["Contradicted"]["f1-score"],
        }
        results["epoch_metrics"].append(epoch_result)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"contra_recall={report['Contradicted']['recall']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{output_dir}/best_hypothesis_only.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # Artifact analysis
    results["final_val_accuracy"] = results["epoch_metrics"][-1]["val_accuracy"]
    results["interpretation"] = _interpret_accuracy(results["final_val_accuracy"])

    with open(f"{output_dir}/hypothesis_only_results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print(f"\n=== ARTIFACT ANALYSIS ===")
    print(f"Hypothesis-only accuracy: {results['final_val_accuracy']:.4f}")
    print(f"Interpretation: {results['interpretation']}")
    return results


def _interpret_accuracy(acc: float) -> str:
    """Interpret hypothesis-only accuracy for artifact analysis."""
    if acc < 0.70:
        return ("STRONG EVIDENCE against artifacts. Claim-only model barely "
                "beats majority class. Verifier genuinely uses evidence.")
    elif acc < 0.75:
        return ("MILD artifacts detected. Some perturbations (negation, laterality) "
                "are detectable from claim alone — this is expected and valid.")
    elif acc < 0.85:
        return ("MODERATE artifacts. Perturbations contain lexical cues. "
                "Consider adding paraphrased contradictions.")
    elif acc < 0.90:
        return ("SIGNIFICANT artifacts. Many perturbations have surface shortcuts. "
                "Need adversarial perturbations that preserve surface features.")
    else:
        return ("SEVERE artifacts. Task is nearly solvable without evidence. "
                "Synthetic perturbations are too easy. Redesign hard negatives.")


@app.local_entrypoint()
def main():
    result = train_hypothesis_only.remote()
    print(f"\nFinal result: {result}")
