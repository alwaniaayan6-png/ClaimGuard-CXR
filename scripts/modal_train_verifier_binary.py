"""Binary (contra-vs-not-contra) claim verifier training for ClaimGuard-CXR.

Usage (from local Mac):
    modal run --detach scripts/modal_train_verifier_binary.py

Trains a RoBERTa-large binary cross-encoder verifier:
  - Label 0: NOT contradicted (formerly Supported OR Insufficient)
  - Label 1: Contradicted (hallucinated)

Motivation (2026-04-05): The 3-class verifier (val_acc=72.84%, test_acc=72.51%)
had catastrophic Insufficient-vs-Supported confusion (3786/5000 insuf -> supp).
Its supported_prob scores collapsed to a ~0.001-wide band around 0.53, which
made conformal BH p-values uninformative (n_green=0 at all alpha levels).

The binary framing:
  - Model outputs 2 logits instead of 3
  - Scores spread naturally across [0, 1]
  - Binary AUC on the frozen 3-class model was already 0.9933

Checkpoint paths:
  - v1 (original): /data/checkpoints/verifier_binary/best_verifier.pt
  - v2 (taxonomy fix): /data/checkpoints/verifier_binary_v2/best_verifier.pt
  - v3 (12-type taxonomy, default): /data/checkpoints/verifier_binary_v3/best_verifier.pt

Reproducibility note (2026-04-14 pre-flight fix):
  The pip_install list pins EXACT versions via `==` for the reproducibility-
  critical libraries.  A prior version used `>=` which let Modal resolve to
  whatever PyPI served at container build time, meaning v1 and v3 runs
  could disagree at the 3rd decimal purely from transformers API drift.
"""

from __future__ import annotations

import modal

# Define the Modal app and container image
app = modal.App("claimguard-verifier-binary")

# Container image with all dependencies.  Exact pins below lock the
# v3 training run to the same library versions as the v1 image, so
# the v1-vs-v3 comparison is clean (any decimal-place drift is the
# taxonomy change, not torch/transformers upgrades).  Bump these
# deliberately when the next verifier version needs a newer stack.
verifier_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.40.0",
        "accelerate==0.30.1",
        "datasets==2.19.1",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.0",
        "scipy==1.13.1",
        "tqdm==4.66.4",
        "pyyaml==6.0.1",
        "tiktoken==0.7.0",
        "sentencepiece==0.2.0",
    )
)

# Persistent volume for storing checkpoints and data
vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=verifier_image,
    gpu="H100",  # 80GB VRAM — blazing fast for RoBERTa-large
    timeout=60 * 60 * 3,  # 3 hour timeout (should finish in ~30 min)
    volumes={"/data": vol},
)
def train_verifier(
    # v3 defaults — v3 training data = v2 taxonomy + 4 new fabrication /
    # compound perturbation types (fabricate_measurement, fabricate_prior,
    # fabricate_temporal, compound_2err, compound_3err).  Pass --data-path
    # and --output-dir explicitly to retrain the v2 checkpoint.
    data_path: str = "/data/verifier_training_data_v3.json",
    output_dir: str = "/data/checkpoints/verifier_binary_v3",
    model_name: str = "roberta-large",
    learning_rate: float = 2e-5,
    batch_size: int = 32,  # H100 80GB easily fits batch=32
    gradient_accumulation: int = 2,  # effective batch = 64 (matches prior runs)
    num_epochs: int = 3,  # 3 epochs first, scale up if needed
    ce_weight: float = 0.7,
    infonce_weight: float = 0.3,
    warmup_fraction: float = 0.10,
    max_length: int = 512,
    seed: int = 42,
) -> dict:
    """Train the claim verifier on Modal GPU.

    Returns dict with training metrics.
    """
    import json
    import os
    import random
    from pathlib import Path

    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm

    # Set seeds (M11: full determinism enforcement)
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
        print(f"WARN: could not enable deterministic algorithms: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = getattr(torch.cuda.get_device_properties(0), 'total_memory',
                           getattr(torch.cuda.get_device_properties(0), 'total_mem', 0))
        print(f"Memory: {total_mem / 1e9:.1f} GB")

    # Load training data
    print(f"Loading training data from {data_path}...")
    with open(data_path, "r") as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} training examples")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build dataset
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
            # BINARY REMAP: {0 Supported, 2 Insufficient} -> 0 (not-contra); {1 Contradicted} -> 1
            orig_label = item["label"]
            label = 1 if orig_label == 1 else 0

            encoding = self.tokenizer(
                claim,
                evidence,
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

    # C12 FIX: patient-stratified 10% validation split.
    # Holds out 10% of patients (seed=42). Best checkpoint selected on val loss
    # (not train loss). Early stopping on val loss, patience=1.
    val_frac = 0.10
    # Stratify by patient_id (fall back to random if missing)
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}"))
                      for i, item in enumerate(training_data)})
    val_rng = random.Random(seed + 1)
    n_val_patients = max(1, int(len(pid_set) * val_frac))
    val_patients = set(val_rng.sample(pid_set, n_val_patients))
    train_examples, val_examples = [], []
    for i, item in enumerate(training_data):
        pid = str(item.get("patient_id", f"_idx{i}"))
        if pid in val_patients:
            val_examples.append(item)
        else:
            train_examples.append(item)
    print(f"Patient-stratified split: {len(train_examples)} train, "
          f"{len(val_examples)} val (val_patients={n_val_patients})")

    dataset = VerifierDataset(train_examples, tokenizer, max_length)
    val_dataset = VerifierDataset(val_examples, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    print(f"Loading {model_name}...")
    from transformers import AutoModel
    import torch.nn as nn
    import torch.nn.functional as F

    class HeatmapEncoder(nn.Module):
        """CNN that encodes 27x27 heatmap -> 768-dim vector (~1.2M params)."""
        def __init__(self, output_dim=768):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, output_dim)

        def forward(self, heatmap):
            if heatmap.ndim == 3:
                heatmap = heatmap.unsqueeze(1)
            return self.proj(self.conv(heatmap).flatten(1))

    class VerifierModel(nn.Module):
        def __init__(self, model_name, heatmap_dim=768, num_classes=2, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size  # 768 for base, 1024 for large
            self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)
            fused_dim = text_dim + heatmap_dim  # 1792

            self.verdict_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
            self.score_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.contrastive_proj = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 128),
            )

        def forward(self, input_ids, attention_mask, heatmap=None):
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_cls = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
            if heatmap is not None:
                hmap_feat = self.heatmap_encoder(heatmap)  # (B, 768)
            else:
                hmap_feat = torch.zeros(text_cls.shape[0], self.heatmap_encoder.proj.out_features, device=text_cls.device, dtype=text_cls.dtype)
            fused = torch.cat([text_cls, hmap_feat], dim=-1)  # (B, 1792)
            verdict_logits = self.verdict_head(fused)
            score = torch.sigmoid(self.score_head(fused)).squeeze(-1)
            proj = self.contrastive_proj(fused)
            return verdict_logits, score, proj, fused

    model = VerifierModel(model_name).to(device)
    # No gradient checkpointing on H100 — 80GB VRAM has plenty of memory
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Text encoder: {sum(p.numel() for p in model.text_encoder.parameters()):,}")
    print(f"  Heatmap encoder: {sum(p.numel() for p in model.heatmap_encoder.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss — NO label smoothing for binary verifier (2026-04-05)
    # Label smoothing 0.05 creates a hard ceiling at P=0.975 / logit_margin=3.66
    # which collapses the cal score distribution and breaks conformal BH (n_green=0).
    # Binary has enough regularization from dropout + early stopping.
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float("inf")         # train loss (kept for backward compat)
    best_val_loss = float("inf")     # C12 FIX: primary selection criterion
    epochs_since_best_val = 0
    early_stop_patience = 1
    metrics_history = []

    @torch.no_grad()
    def _evaluate_val() -> tuple[float, float]:
        """Return (val_loss, val_acc) on held-out validation set."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for b in val_loader:
            ii = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            lab = b["label"].to(device)
            hm = torch.zeros(ii.shape[0], 27, 27, device=device)
            vl, _, _, _ = model(ii, am, heatmap=hm)
            loss = ce_criterion(vl, lab)
            total_loss += float(loss.item()) * ii.shape[0]
            total_correct += int((vl.argmax(dim=-1) == lab).sum().item())
            total += ii.shape[0]
        model.train()
        if total == 0:
            return float("inf"), 0.0
        return total_loss / total, total_correct / total

    print(f"Starting training: {num_epochs} epochs, {len(dataloader)} batches/epoch")
    print(f"Effective batch size: {batch_size * gradient_accumulation}")
    print(f"FP32 training (stable on H100), gradient checkpointing: disabled (80GB VRAM)")

    for epoch in range(num_epochs):
        model.train()
        epoch_ce_loss = 0.0
        epoch_infonce_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Heatmap: random for now (will use real grounding heatmaps once grounding module is trained)
            # This trains the text branch; heatmap branch learns from zeros initially
            heatmap = torch.zeros(input_ids.shape[0], 27, 27, device=device)

            # FP32 training — DeBERTa-v3 produces NaN with fp16/bf16 autocast
            verdict_logits, score, proj, fused = model(input_ids, attention_mask, heatmap=heatmap)

            # CE loss only — skip InfoNCE (unstable with small batch sizes,
            # causes NaN when too few positive pairs exist in a batch of 8)
            ce_loss = ce_criterion(verdict_logits, labels)

            # NaN guard (M4 FIX: do NOT count skipped batches in n_batches)
            if torch.isnan(ce_loss):
                print(f"  WARNING: NaN CE loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            loss = ce_loss / gradient_accumulation
            loss.backward()

            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # M4 FIX: only accumulate metrics for batches that succeeded
            epoch_ce_loss += ce_loss.item()
            epoch_total_loss += loss.item() * gradient_accumulation
            n_batches += 1

            progress.set_postfix({
                "ce": f"{ce_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Epoch metrics
        avg_ce = epoch_ce_loss / max(n_batches, 1)
        avg_total = epoch_total_loss / max(n_batches, 1)

        # C12 FIX: evaluate on held-out val set for best-checkpoint selection
        val_loss, val_acc = _evaluate_val()
        train_val_gap = avg_ce - val_loss

        # SAVE CHECKPOINTS FIRST — before any print that could crash
        checkpoint_path = os.path.join(output_dir, f"verifier_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_total,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, checkpoint_path)

        # Save best checkpoint ON VAL LOSS (C12 FIX — was on train loss)
        improved_val = val_loss < best_val_loss
        if improved_val:
            best_val_loss = val_loss
            epochs_since_best_val = 0
            best_path = os.path.join(output_dir, "best_verifier.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
                "val_loss": best_val_loss,
                "val_acc": val_acc,
            }, best_path)
        else:
            epochs_since_best_val += 1

        if avg_total < best_loss:
            best_loss = avg_total

        # Commit volume BEFORE logging (so we don't lose progress on crash)
        vol.commit()

        # Now safe to log
        metrics = {
            "epoch": epoch + 1,
            "train_ce_loss": avg_ce,
            "train_total_loss": avg_total,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_val_gap": train_val_gap,
            "is_best_val": improved_val,
        }
        metrics_history.append(metrics)
        print(f"Epoch {epoch+1}: train_CE={avg_ce:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} gap={train_val_gap:+.4f}"
              + (" [BEST]" if improved_val else ""))
        print(f"  Saved epoch {epoch+1} checkpoint and committed volume")

        # Early stopping on val loss (patience=1)
        if epochs_since_best_val > early_stop_patience:
            print(f"  Early stopping — val loss has not improved for "
                  f"{epochs_since_best_val} epochs (patience={early_stop_patience})")
            break

    # Save final
    final_path = os.path.join(output_dir, "verifier_final.pt")
    torch.save(model.state_dict(), final_path)

    # Save metrics
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Commit volume changes
    vol.commit()

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")

    return {
        "best_loss": best_loss,
        "final_epoch": num_epochs,
        "metrics": metrics_history,
    }


@app.local_entrypoint()
def main():
    """Entry point when running `modal run scripts/modal_train_verifier.py`."""
    print("Launching verifier training on Modal...")
    result = train_verifier.remote()
    print(f"\nTraining complete!")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Epochs: {result['final_epoch']}")
