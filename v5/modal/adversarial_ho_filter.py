"""Adversarial hypothesis-only (HO) filter — Modal entrypoint.

Trains a text-only RoBERTa model on GroundBench training claims.
Any training example where the HO model already predicts the correct
label (i.e. solvable without image evidence) is flagged and removed
from the v5 training manifest.

Usage:
    modal run --detach v5/modal/adversarial_ho_filter.py

Outputs (written to Modal volume /data/v5):
    groundbench/ho_filter_manifest.json  — kept rows (image-evidence required)
    groundbench/ho_scores.json           — per-row HO softmax scores
    groundbench/ho_filter_summary.txt    — stats: total / dropped / kept
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------
app = modal.App("claimguard-v5-ho-filter")

V5_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "scikit-learn==1.4.2",
        "numpy",
        "tqdm",
    )
)

volume = modal.Volume.from_name("claimguard-v5-data", create_if_missing=True)
DATA_ROOT = Path("/data/v5")
GROUNDBENCH_DIR = DATA_ROOT / "groundbench"


# ---------------------------------------------------------------------------
# Core logic (runs inside the container)
# ---------------------------------------------------------------------------
def _load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _train_ho_model(rows: list[dict], device: str = "cuda") -> "transformers.RobertaForSequenceClassification":  # noqa: F821
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, RobertaForSequenceClassification
    from torch.optim import AdamW

    label2id = {"SUPPORTED": 0, "CONTRADICTED": 1}
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("label") in label2id]

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    class _DS(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict:
            r = self.rows[i]
            enc = tokenizer(
                r["claim_text"],
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label2id[r["label"]], dtype=torch.long),
            }

    ds = _DS(train_rows)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
    model.to(device)
    opt = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
        print(f"  HO epoch {epoch+1}: avg loss {total_loss/len(loader):.4f}")

    model.eval()
    return model, tokenizer


def _score_rows(
    model: "torch.nn.Module",  # noqa: F821
    tokenizer,
    rows: list[dict],
    device: str = "cuda",
) -> list[dict]:
    import torch
    import torch.nn.functional as F

    label2id = {"SUPPORTED": 0, "CONTRADICTED": 1}
    scored = []
    for r in rows:
        if r.get("label") not in label2id:
            scored.append({**r, "ho_score": None, "ho_correct": False})
            continue
        enc = tokenizer(
            r["claim_text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
        true = label2id[r["label"]]
        scored.append(
            {
                **r,
                "ho_score_supported": float(probs[0]),
                "ho_score_contradicted": float(probs[1]),
                "ho_correct": pred == true,
            }
        )
    return scored


@app.function(
    image=V5_IMAGE,
    gpu="H100",
    timeout=3600 * 4,
    volumes={str(DATA_ROOT): volume},
    secrets=[modal.Secret.from_name("huggingface", required=False)],
)
def run_ho_filter() -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest_path = GROUNDBENCH_DIR / "groundbench_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"GroundBench manifest not found at {manifest_path}. "
            "Run build_groundbench.py first."
        )

    rows = _load_manifest(manifest_path)
    print(f"Loaded {len(rows)} rows from manifest.")

    print("Training HO model (RoBERTa-large, text-only, 3 epochs)...")
    model, tokenizer = _train_ho_model(rows, device=device)

    print("Scoring all rows...")
    scored = _score_rows(model, tokenizer, rows, device=device)

    # Save scores
    scores_path = GROUNDBENCH_DIR / "ho_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scored, f, indent=2)

    # Filter: keep only rows where HO model is WRONG (image evidence required)
    train_scored = [r for r in scored if r.get("split") == "train"]
    kept = [r for r in train_scored if not r["ho_correct"]]
    dropped = [r for r in train_scored if r["ho_correct"]]
    non_train = [r for r in scored if r.get("split") != "train"]

    # Rebuild manifest: all non-train rows + filtered train rows
    filtered_manifest = kept + non_train
    kept_path = GROUNDBENCH_DIR / "ho_filter_manifest.json"
    with open(kept_path, "w") as f:
        json.dump(filtered_manifest, f, indent=2)

    # Summary
    summary_lines = [
        f"HO Filter Summary",
        f"=================",
        f"Train rows total : {len(train_scored)}",
        f"Dropped (HO correct) : {len(dropped)} ({100*len(dropped)/max(len(train_scored),1):.1f}%)",
        f"Kept (image required): {len(kept)} ({100*len(kept)/max(len(train_scored),1):.1f}%)",
        f"Non-train rows   : {len(non_train)}",
        f"Output manifest  : {kept_path}",
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(GROUNDBENCH_DIR / "ho_filter_summary.txt", "w") as f:
        f.write(summary_text + "\n")

    volume.commit()
    return {"kept": len(kept), "dropped": len(dropped), "total_train": len(train_scored)}


@app.local_entrypoint()
def main() -> None:
    result = run_ho_filter.remote()
    print(f"\nHO filter complete: {result}")
