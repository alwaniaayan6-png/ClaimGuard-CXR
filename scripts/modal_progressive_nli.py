"""Progressive NLI fine-tuning for DeBERTa-v3-large: MNLI -> MedNLI -> RadNLI -> ClaimGuard.

Each stage checkpoints; the next stage loads from the previous. This progressive
domain adaptation chain is consistently best in the literature for domain-specific NLI.

Stages:
  1. MNLI (433K pairs) — general NLI competence
  2. MedNLI (11K pairs) — medical domain adaptation
  3. RadNLI (480 pairs, PhysioNet) — radiology-specific NLI
  4. ClaimGuard hard negatives (30K pairs) — task-specific fine-tuning

All datasets loaded from HuggingFace Datasets except RadNLI (not on HF — skip
if unavailable and go straight from MedNLI to ClaimGuard).

Usage:
    modal run --detach scripts/modal_progressive_nli.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-progressive-nli")

nli_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=nli_image,
    gpu="H100",
    timeout=60 * 60 * 6,  # 6 hours — MNLI stage is long
    volumes={"/data": vol},
)
def run_progressive_nli(
    claimguard_data_path: str = "/data/verifier_training_data.json",
    output_base_dir: str = "/data/checkpoints/progressive_nli",
    model_name: str = "microsoft/deberta-v3-large",
    mnli_epochs: int = 1,
    mednli_epochs: int = 3,
    claimguard_epochs: int = 5,
    mnli_lr: float = 2e-5,
    mednli_lr: float = 1e-5,
    claimguard_lr: float = 1e-5,
    batch_size: int = 16,
    gradient_accumulation: int = 4,
    max_length: int = 256,
    claimguard_max_length: int = 512,
    seed: int = 42,
) -> dict:
    """Run the full progressive NLI pipeline."""
    import json
    import os
    import random

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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    os.makedirs(output_base_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = {"stages": []}

    # ============================================================
    # Model: DeBERTa + 3-class NLI head (for MNLI/MedNLI stages)
    # then swapped to 2-class for ClaimGuard stage
    # ============================================================
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

    # ============================================================
    # Generic NLI dataset
    # ============================================================
    class NLIDataset(Dataset):
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

    def train_stage(model, train_loader, val_loader, epochs, lr, stage_name, save_dir):
        """Train one NLI stage and return metrics."""
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs // gradient_accumulation
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * 0.10), total_steps
        )
        criterion = nn.CrossEntropyLoss()
        os.makedirs(save_dir, exist_ok=True)

        best_val_acc = 0.0
        stage_metrics = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"{stage_name} E{epoch+1}")):
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

            avg_loss = total_loss / len(train_loader)

            # Validate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(ids, mask)
                    all_preds.extend(logits.argmax(-1).cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            val_acc = accuracy_score(all_labels, all_preds)
            stage_metrics.append({"epoch": epoch+1, "loss": avg_loss, "val_acc": val_acc})
            print(f"  {stage_name} E{epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.encoder.state_dict(), f"{save_dir}/best_encoder.pt")
                torch.save(model.state_dict(), f"{save_dir}/best_full.pt")

        vol.commit()
        return {"stage": stage_name, "best_val_acc": best_val_acc, "metrics": stage_metrics}

    # ============================================================
    # STAGE 1: MNLI
    # ============================================================
    print("\n=== STAGE 1: MNLI (433K pairs) ===")
    try:
        mnli = load_dataset("nli_for_simcse", "mnli", split="train")
        # MNLI labels: 0=entailment, 1=neutral, 2=contradiction
        # Filter out -1 labels
        mnli = mnli.filter(lambda x: x["label"] in [0, 1, 2])

        # Subsample for speed — full MNLI is 393K, use 100K
        if len(mnli) > 100000:
            mnli = mnli.shuffle(seed=seed).select(range(100000))

        # Split 95/5
        split = mnli.train_test_split(test_size=0.05, seed=seed)
        train_ds = NLIDataset(
            split["train"]["premise"], split["train"]["hypothesis"],
            split["train"]["label"], tokenizer, max_length,
        )
        val_ds = NLIDataset(
            split["test"]["premise"], split["test"]["hypothesis"],
            split["test"]["label"], tokenizer, max_length,
        )
    except Exception as e:
        print(f"MNLI loading failed: {e}")
        print("Trying alternate MNLI source...")
        try:
            mnli = load_dataset("glue", "mnli", split="train")
            mnli = mnli.filter(lambda x: x["label"] in [0, 1, 2])
            if len(mnli) > 100000:
                mnli = mnli.shuffle(seed=seed).select(range(100000))
            split = mnli.train_test_split(test_size=0.05, seed=seed)
            train_ds = NLIDataset(
                split["train"]["premise"], split["train"]["hypothesis"],
                split["train"]["label"], tokenizer, max_length,
            )
            val_ds = NLIDataset(
                split["test"]["premise"], split["test"]["hypothesis"],
                split["test"]["label"], tokenizer, max_length,
            )
        except Exception as e2:
            print(f"MNLI alternate also failed: {e2}. Skipping MNLI stage.")
            train_ds = None

    model = NLIModel(model_name, num_classes=3).to(device)

    if train_ds is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                                num_workers=2, pin_memory=True)
        mnli_result = train_stage(
            model, train_loader, val_loader, mnli_epochs, mnli_lr,
            "MNLI", f"{output_base_dir}/stage1_mnli"
        )
        results["stages"].append(mnli_result)
        print(f"MNLI best val_acc: {mnli_result['best_val_acc']:.4f}")
    else:
        results["stages"].append({"stage": "MNLI", "status": "skipped"})

    # ============================================================
    # STAGE 2: MedNLI
    # ============================================================
    print("\n=== STAGE 2: MedNLI (11K pairs) ===")
    try:
        mednli = load_dataset("bigbio/mednli", name="mednli_bigbio_te", split="train",
                              trust_remote_code=True)
        mednli_val = load_dataset("bigbio/mednli", name="mednli_bigbio_te", split="validation",
                                  trust_remote_code=True)

        # BigBio format: premise, hypothesis, label (entailment/neutral/contradiction)
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        train_ds = NLIDataset(
            mednli["premise"], mednli["hypothesis"],
            [label_map.get(str(l), 1) for l in mednli["label"]], tokenizer, max_length,
        )
        val_ds = NLIDataset(
            mednli_val["premise"], mednli_val["hypothesis"],
            [label_map.get(str(l), 1) for l in mednli_val["label"]], tokenizer, max_length,
        )
    except Exception as e:
        print(f"MedNLI loading failed: {e}")
        print("Trying alternate MedNLI source...")
        try:
            mednli = load_dataset("jgc128/mednli", split="train")
            mednli_val = load_dataset("jgc128/mednli", split="dev")
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
            train_ds = NLIDataset(
                mednli["sentence1"], mednli["sentence2"],
                [label_map.get(l, 1) for l in mednli["gold_label"]], tokenizer, max_length,
            )
            val_ds = NLIDataset(
                mednli_val["sentence1"], mednli_val["sentence2"],
                [label_map.get(l, 1) for l in mednli_val["gold_label"]], tokenizer, max_length,
            )
        except Exception as e2:
            print(f"MedNLI alternate failed: {e2}. Skipping MedNLI stage.")
            train_ds = None

    if train_ds is not None:
        # Reinitialize NLI head but keep encoder from MNLI
        new_head = nn.Sequential(
            nn.Linear(model.encoder.config.hidden_size, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, 3),
        ).to(device)
        model.nli_head = new_head

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                                num_workers=2, pin_memory=True)
        mednli_result = train_stage(
            model, train_loader, val_loader, mednli_epochs, mednli_lr,
            "MedNLI", f"{output_base_dir}/stage2_mednli"
        )
        results["stages"].append(mednli_result)
        print(f"MedNLI best val_acc: {mednli_result['best_val_acc']:.4f}")
    else:
        results["stages"].append({"stage": "MedNLI", "status": "skipped"})

    # ============================================================
    # STAGE 3: RadNLI (skip if not available — requires PhysioNet)
    # ============================================================
    print("\n=== STAGE 3: RadNLI (skipped — requires PhysioNet credentialing) ===")
    results["stages"].append({"stage": "RadNLI", "status": "skipped_physionet_required"})

    # ============================================================
    # STAGE 4: ClaimGuard hard negatives (binary: 2-class)
    # ============================================================
    print(f"\n=== STAGE 4: ClaimGuard ({claimguard_data_path}) ===")
    with open(claimguard_data_path) as f:
        cg_data = json.load(f)
    print(f"Loaded {len(cg_data)} ClaimGuard training examples")

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
                padding="max_length", truncation="only_second",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Patient-stratified split
    pid_set = sorted({str(item.get("patient_id", f"_idx{i}"))
                      for i, item in enumerate(cg_data)})
    val_rng = random.Random(seed + 1)
    n_val = max(1, int(len(pid_set) * 0.10))
    val_patients = set(val_rng.sample(pid_set, n_val))
    train_ex = [item for i, item in enumerate(cg_data)
                if str(item.get("patient_id", f"_idx{i}")) not in val_patients]
    val_ex = [item for i, item in enumerate(cg_data)
              if str(item.get("patient_id", f"_idx{i}")) in val_patients]

    train_ds = ClaimGuardDataset(train_ex, tokenizer, claimguard_max_length)
    val_ds = ClaimGuardDataset(val_ex, tokenizer, claimguard_max_length)

    # Swap head to 2-class binary for ClaimGuard
    binary_head = nn.Sequential(
        nn.Linear(model.encoder.config.hidden_size, 256), nn.ReLU(),
        nn.Dropout(0.1), nn.Linear(256, 2),
    ).to(device)
    model.nli_head = binary_head
    model.num_classes = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                            num_workers=2, pin_memory=True)

    cg_result = train_stage(
        model, train_loader, val_loader, claimguard_epochs, claimguard_lr,
        "ClaimGuard", f"{output_base_dir}/stage4_claimguard"
    )
    results["stages"].append(cg_result)
    print(f"ClaimGuard best val_acc: {cg_result['best_val_acc']:.4f}")

    # Save final checkpoint in format compatible with deberta_verifier.py
    # Attribute names: text_encoder, verdict_head, temperature
    final_state = {
        "text_encoder": model.encoder.state_dict(),
        "verdict_head": model.nli_head.state_dict(),
    }
    final_dir = f"{output_base_dir}/final"
    os.makedirs(final_dir, exist_ok=True)

    # Save as a flat state dict matching DeBERTaBinaryVerifier
    flat_state = {}
    for k, v in model.encoder.state_dict().items():
        flat_state[f"text_encoder.{k}"] = v
    for k, v in model.nli_head.state_dict().items():
        flat_state[f"verdict_head.{k}"] = v
    flat_state["temperature"] = torch.tensor(1.0)
    torch.save(flat_state, f"{final_dir}/best_verifier.pt")

    # Also save a convenience copy at the standard checkpoint location
    torch.save(flat_state, f"{output_base_dir}/../verifier_deberta_progressive/best_verifier.pt")

    results["final_checkpoint"] = f"{final_dir}/best_verifier.pt"

    with open(f"{output_base_dir}/progressive_nli_results.json", "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print(f"\n=== Progressive NLI complete ===")
    for stage in results["stages"]:
        status = stage.get("best_val_acc", stage.get("status", "unknown"))
        print(f"  {stage['stage']}: {status}")
    return results


@app.local_entrypoint()
def main():
    result = run_progressive_nli.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
