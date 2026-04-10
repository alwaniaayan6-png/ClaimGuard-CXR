"""DeBERTa-v3-large binary verifier training for ClaimGuard-CXR v2.

Replaces modal_train_verifier_binary.py (RoBERTa-large) with DeBERTa-v3-large.

Key changes from v1:
  - DeBERTa-v3-large backbone (disentangled attention, +2-5% on NLI)
  - Lower learning rate (1e-5 vs 2e-5) — DeBERTa is more sensitive
  - 5 epochs (vs 3) — DeBERTa benefits from more training at lower LR
  - Clean 1024-dim head (no heatmap zeros)
  - Optional progressive NLI pre-fine-tuning (MedNLI -> RadNLI -> ClaimGuard)

Usage:
    modal run --detach scripts/modal_train_verifier_deberta.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-verifier-deberta")

verifier_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=verifier_image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/data": vol},
)
def train_deberta_verifier(
    data_path: str = "/data/verifier_training_data.json",
    output_dir: str = "/data/checkpoints/verifier_deberta_v2",
    model_name: str = "microsoft/deberta-v3-large",
    pretrained_checkpoint: str = "",  # Path to progressive NLI checkpoint (optional)
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    gradient_accumulation: int = 4,
    num_epochs: int = 5,
    warmup_fraction: float = 0.10,
    max_length: int = 512,
    seed: int = 42,
) -> dict:
    """Train DeBERTa-v3-large binary verifier on Modal H100."""
    import json
    import os
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm

    # Determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"WARN: deterministic algorithms: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class VerifierDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            claim = item["claim"]
            evidence = " [SEP] ".join(item.get("evidence", [])[:2])
            # Binary remap: {0=Supported, 2=Insufficient} -> 0; {1=Contradicted} -> 1
            label = 1 if item["label"] == 1 else 0

            encoding = self.tokenizer(
                claim,
                evidence,
                max_length=self.max_length,
                padding="max_length",
                truncation="only_second",  # Preserve claim text
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Patient-stratified 10% validation split (same as v1)
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}"))
                      for i, item in enumerate(training_data)})
    val_rng = random.Random(seed + 1)
    n_val = max(1, int(len(pid_set) * 0.10))
    val_patients = set(val_rng.sample(pid_set, n_val))

    train_ex, val_ex = [], []
    for i, item in enumerate(training_data):
        pid = str(item.get("patient_id", f"_idx{i}"))
        (val_ex if pid in val_patients else train_ex).append(item)
    print(f"Split: {len(train_ex)} train, {len(val_ex)} val")

    train_ds = VerifierDataset(train_ex, tokenizer, max_length)
    val_ds = VerifierDataset(val_ex, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model: clean DeBERTa binary verifier (no heatmap branch)
    # IMPORTANT: attribute names MUST match deberta_verifier.py and demo/app.py
    # for checkpoint compatibility: text_encoder, verdict_head, temperature
    class DeBERTaBinaryVerifier(nn.Module):
        def __init__(self, model_name, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size  # 1024
            self.verdict_head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.verdict_head(cls)

    model = DeBERTaBinaryVerifier(model_name).to(device)

    # Load progressive NLI checkpoint if provided
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        print(f"Loading progressive NLI checkpoint from {pretrained_checkpoint}")
        state_dict = torch.load(pretrained_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected if head was reinitialized): {missing[:5]}")
    elif pretrained_checkpoint:
        print(f"WARNING: Checkpoint not found at {pretrained_checkpoint}. Training from scratch.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss: no label smoothing (critical for conformal)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 2  # More patience for DeBERTa (slower convergence)
    metrics_history = []

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * ids.shape[0]
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += ids.shape[0]
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        model.train()
        acc = total_correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else float("inf")
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0
        return avg_loss, acc, auroc

    print(f"\nStarting training: {num_epochs} epochs, effective batch={batch_size*gradient_accumulation}")
    print(f"LR: {learning_rate}, warmup: {warmup_steps}/{total_steps} steps")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
        val_loss, val_acc, val_auroc = evaluate()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_auroc": val_auroc,
        }
        metrics_history.append(epoch_metrics)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"val_auroc={val_auroc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "best_verifier.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save training log
    results = {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "best_epoch": max(metrics_history, key=lambda x: x["val_accuracy"])["epoch"],
        "final_val_accuracy": metrics_history[-1]["val_accuracy"],
        "final_val_auroc": metrics_history[-1]["val_auroc"],
        "epochs_trained": len(metrics_history),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "max_length": max_length,
            "warmup_fraction": warmup_fraction,
        },
        "epoch_metrics": metrics_history,
    }
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print(f"\nTraining complete. Best val_acc: {max(m['val_accuracy'] for m in metrics_history):.4f}")
    return results


@app.local_entrypoint()
def main():
    result = train_deberta_verifier.remote()
    print(f"\nResult: {result}")
