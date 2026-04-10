"""DeBERTa-v3-large zero-shot NLI baseline for ClaimGuard-CXR v2.

Uses off-the-shelf DeBERTa-v3-large-mnli (cross-encoder/nli-deberta-v3-large)
without ANY fine-tuning on radiology data. Tests how much domain-specific
training contributes over general NLI competence.

Usage:
    modal run --detach scripts/baseline_deberta_zeroshot_nli.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-baseline-deberta-nli")

baseline_image = (
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
    image=baseline_image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def run_deberta_nli_baseline(
    test_claims_path: str = "/data/eval_data/test_claims.json",
    output_path: str = "/data/eval_results_deberta_zeroshot_nli/results.json",
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    batch_size: int = 32,
    max_length: int = 512,
    seed: int = 42,
) -> dict:
    """Run zero-shot DeBERTa NLI on test claims."""
    import json
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained NLI model (not fine-tuned on radiology)
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # NLI labels: 0=contradiction, 1=neutral, 2=entailment

    # Load test claims
    with open(test_claims_path) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} test claims")

    class NLIDataset(Dataset):
        def __init__(self, claims, tokenizer, max_length):
            self.claims = claims
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.claims)

        def __getitem__(self, idx):
            item = self.claims[idx]
            claim = item["claim"]
            evidence = " ".join(item.get("evidence", [])[:2])
            # NLI format: premise=evidence, hypothesis=claim
            enc = self.tokenizer(
                evidence, claim,
                max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            label = 1 if item["label"] == 1 else 0
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": label,
            }

    dataset = NLIDataset(claims, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_preds = []
    all_labels = []
    all_contra_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="DeBERTa NLI zero-shot"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits  # (B, 3): contradiction, neutral, entailment
            probs = F.softmax(logits, dim=-1)

            # Contradiction probability -> our "contra" prediction
            contra_prob = probs[:, 0].cpu().numpy()
            # Binary prediction: if P(contradiction) > 0.5, predict contradicted
            preds = (contra_prob > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_contra_probs.extend(contra_prob)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_contra_probs = np.array(all_contra_probs)

    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    contra_recall = float((all_preds[all_labels == 1] == 1).mean()) if all_labels.sum() > 0 else 0.0

    try:
        auroc = float(roc_auc_score(all_labels, all_contra_probs))
    except ValueError:
        auroc = 0.5

    results = {
        "method": "DeBERTa-v3-large zero-shot NLI",
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "contra_recall": contra_recall,
        "auroc": auroc,
        "n_claims": len(claims),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print(f"\nDeBERTa NLI zero-shot: acc={accuracy:.4f}, contra_recall={contra_recall:.4f}, auroc={auroc:.4f}")
    return results


@app.local_entrypoint()
def main():
    result = run_deberta_nli_baseline.remote()
    print(f"\nResult: {result}")
