"""Progressive NLI fine-tuning: MNLI -> MedNLI -> ClaimGuard (RunPod standalone).

Stripped of Modal decorators. Runs directly on RunPod H100.

Usage:
    python run_progressive_nli.py \
        --data-path /workspace/data/verifier_training_data_v2.json \
        --output-dir /workspace/checkpoints/progressive_nli
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NLIModel(nn.Module):
    def __init__(self, model_name_or_path, num_classes=3, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        text_dim = self.encoder.config.hidden_size
        self.nli_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.nli_head(cls)


class PairDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx], self.hypotheses[idx],
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ClaimGuardDataset(Dataset):
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
        label = 1 if item["label"] == 1 else 0
        enc = self.tokenizer(
            claim, evidence, max_length=self.max_length,
            padding="max_length", truncation="only_second", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_stage(model, train_loader, val_loader, epochs, lr, stage_name,
                save_dir, device, grad_accum=4):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.10), max(total_steps, 1)
    )
    criterion = nn.CrossEntropyLoss()
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    metrics = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"{stage_name} E{epoch+1}")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())
        val_acc = accuracy_score(all_labels, all_preds)
        metrics.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc})
        print(f"  {stage_name} E{epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.encoder.state_dict(), f"{save_dir}/best_encoder.pt")
            torch.save(model.state_dict(), f"{save_dir}/best_full.pt")
            print(f"  -> Saved best (val_acc={val_acc:.4f})")

    return {"stage": stage_name, "best_val_acc": best_val_acc, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/workspace/data/verifier_training_data_v2.json")
    parser.add_argument("--output-dir", default="/workspace/checkpoints/progressive_nli")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--mnli-epochs", type=int, default=1)
    parser.add_argument("--mednli-epochs", type=int, default=3)
    parser.add_argument("--claimguard-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--mnli-subsample", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = {"stages": []}

    model = NLIModel(args.model_name, num_classes=3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Enable gradient checkpointing to save VRAM
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # ===== STAGE 1: MNLI =====
    mnli_ckpt = f"{args.output_dir}/stage1_mnli/best_encoder.pt"
    if os.path.exists(mnli_ckpt):
        print(f"\n=== STAGE 1: MNLI (SKIPPED — checkpoint exists at {mnli_ckpt}) ===")
        model.encoder.load_state_dict(torch.load(mnli_ckpt, map_location=device, weights_only=True))
        results["stages"].append({"stage": "MNLI", "status": "skipped_checkpoint_exists"})
    else:
        print("\n=== STAGE 1: MNLI ===")
        try:
            mnli = load_dataset("glue", "mnli", split="train", trust_remote_code=True)
            mnli = mnli.filter(lambda x: x["label"] in [0, 1, 2])
            if len(mnli) > args.mnli_subsample:
                mnli = mnli.shuffle(seed=args.seed).select(range(args.mnli_subsample))
            split = mnli.train_test_split(test_size=0.05, seed=args.seed)
            train_ds = PairDataset(
                split["train"]["premise"], split["train"]["hypothesis"],
                split["train"]["label"], tokenizer, 256,
            )
            val_ds = PairDataset(
                split["test"]["premise"], split["test"]["hypothesis"],
                split["test"]["label"], tokenizer, 256,
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=4, pin_memory=True)
            r = train_stage(model, train_loader, val_loader, args.mnli_epochs, 2e-5,
                            "MNLI", f"{args.output_dir}/stage1_mnli", device, args.grad_accum)
            results["stages"].append(r)
        except Exception as e:
            print(f"MNLI failed: {e}")
            results["stages"].append({"stage": "MNLI", "status": f"failed: {e}"})

    # ===== STAGE 2: MedNLI =====
    mednli_ckpt = f"{args.output_dir}/stage2_mednli/best_encoder.pt"
    if os.path.exists(mednli_ckpt):
        print(f"\n=== STAGE 2: MedNLI (SKIPPED — checkpoint exists) ===")
        model.encoder.load_state_dict(torch.load(mednli_ckpt, map_location=device, weights_only=True))
        results["stages"].append({"stage": "MedNLI", "status": "skipped_checkpoint_exists"})
    else:
        print("\n=== STAGE 2: MedNLI ===")
        try:
            mednli = load_dataset("bigbio/mednli", name="mednli_bigbio_te", split="train",
                                  trust_remote_code=True)
            mednli_val = load_dataset("bigbio/mednli", name="mednli_bigbio_te", split="validation",
                                      trust_remote_code=True)
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

            train_ds = PairDataset(
                mednli["premise"], mednli["hypothesis"],
                [label_map.get(str(l), 1) for l in mednli["label"]], tokenizer, 256,
            )
            val_ds = PairDataset(
                mednli_val["premise"], mednli_val["hypothesis"],
                [label_map.get(str(l), 1) for l in mednli_val["label"]], tokenizer, 256,
            )

            model.nli_head = nn.Sequential(
                nn.Linear(model.encoder.config.hidden_size, 256), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(256, 3),
            ).to(device)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=4, pin_memory=True)
            r = train_stage(model, train_loader, val_loader, args.mednli_epochs, 1e-5,
                            "MedNLI", f"{args.output_dir}/stage2_mednli", device, args.grad_accum)
            results["stages"].append(r)
        except Exception as e:
            print(f"MedNLI failed: {e}")
            results["stages"].append({"stage": "MedNLI", "status": f"failed: {e}"})

    # ===== STAGE 3: ClaimGuard (Binary) =====
    print(f"\n=== STAGE 3: ClaimGuard ({args.data_path}) ===")
    with open(args.data_path) as f:
        cg_data = json.load(f)
    print(f"Loaded {len(cg_data)} training examples")

    # Patient-stratified split
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}"))
                      for i, item in enumerate(cg_data)})
    val_rng = random.Random(args.seed + 1)
    n_val = max(1, int(len(pid_set) * 0.10))
    val_patients = set(val_rng.sample(pid_set, n_val))
    train_ex = [item for i, item in enumerate(cg_data)
                if str(item.get("patient_id", f"_idx{i}")) not in val_patients]
    val_ex = [item for i, item in enumerate(cg_data)
              if str(item.get("patient_id", f"_idx{i}")) in val_patients]
    print(f"Split: {len(train_ex)} train, {len(val_ex)} val")

    # Swap to binary head
    model.nli_head = nn.Sequential(
        nn.Linear(model.encoder.config.hidden_size, 256), nn.ReLU(),
        nn.Dropout(0.1), nn.Linear(256, 2),
    ).to(device)
    model.num_classes = 2

    train_ds = ClaimGuardDataset(train_ex, tokenizer, 512)
    val_ds = ClaimGuardDataset(val_ex, tokenizer, 512)
    # Reduce batch size for 512 seq_len (OOM on 45GB GPUs at batch_size=16)
    cg_batch = max(4, args.batch_size // 4)  # 4 for L40S, 16 for H100
    cg_accum = max(1, args.grad_accum * (args.batch_size // cg_batch))  # keep effective batch ~64
    print(f"ClaimGuard batch_size={cg_batch}, grad_accum={cg_accum}, effective={cg_batch * cg_accum}")
    train_loader = DataLoader(train_ds, batch_size=cg_batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cg_batch * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    r = train_stage(model, train_loader, val_loader, args.claimguard_epochs, 1e-5,
                    "ClaimGuard", f"{args.output_dir}/stage3_claimguard", device, cg_accum)
    results["stages"].append(r)

    # Save final checkpoint compatible with deberta_verifier.py
    final_dir = f"{args.output_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    flat_state = {}
    for k, v in model.encoder.state_dict().items():
        flat_state[f"text_encoder.{k}"] = v
    for k, v in model.nli_head.state_dict().items():
        flat_state[f"verdict_head.{k}"] = v
    flat_state["temperature"] = torch.tensor(1.0)
    torch.save(flat_state, f"{final_dir}/best_verifier.pt")
    print(f"\nFinal checkpoint: {final_dir}/best_verifier.pt")

    results["final_checkpoint"] = f"{final_dir}/best_verifier.pt"
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Progressive NLI Complete ===")
    for s in results["stages"]:
        print(f"  {s['stage']}: {s.get('best_val_acc', s.get('status', '?'))}")


if __name__ == "__main__":
    main()
