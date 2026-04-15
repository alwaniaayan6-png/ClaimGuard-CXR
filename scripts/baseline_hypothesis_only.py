"""Hypothesis-only baseline for ClaimGuard-CXR artifact analysis.

Tests whether the verifier's accuracy comes from genuine evidence
reasoning or from surface shortcuts in synthetic perturbations.

Method: Train a RoBERTa-large verifier on the SAME training data as
v1 / v3 but with evidence passages REPLACED by a single pad token. If
this "claim-with-masked-evidence" model achieves significantly above
66% (majority class), the perturbations contain exploitable lexical
cues.

**Critical methodology note (2026-04-14 pre-flight reviewer fix).**
A prior version of this script tokenized the claim alone with
``max_length=128``, while the v1/v3 verifier tokenizes
``(claim, evidence)`` as a pair with ``max_length=512``.  That is
NOT a hypothesis-only baseline — it's a different architecture and
different input shape, and the resulting number is not directly
comparable to the v1 verifier's accuracy.  The fix: this script now
tokenizes ``(claim, "[PAD]")`` as a pair with ``max_length=512``,
matching the v1/v3 input shape exactly.  The "[PAD]" evidence string
is a single whitespace-only token that the tokenizer expands into
attention-mask-zero positions — the model literally cannot read the
evidence, but the input shape is identical to the real verifier.

Expected outcome (calibrate against v1's 97.71% pre-sprint number):
  - 70-80%: Some perturbations (negation, laterality) ARE detectable
    from the claim alone — this is VALID signal, not an artifact
  - >90%: Task too easy, perturbations have severe surface shortcuts
  - <75%: Strong evidence that the verifier genuinely uses evidence

v3 retrain target: the HO gap should be the difference between v3
verifier accuracy and this HO accuracy.  The pre-sprint v1 gap was
0.60 pp (98.31% - 97.71%).  Task 2 + Task 3 aim to grow this to ≥ 5 pp.

Reference: Herlihy & Rudinger (ACL 2021) — hypothesis-only baselines for MedNLI

Usage (v3 retrain):
    modal run --detach scripts/baseline_hypothesis_only.py \\
        --data-path /data/verifier_training_data_v3.json \\
        --output-dir /data/checkpoints/hypothesis_only_v3
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
    data_path: str = "/data/verifier_training_data_v3.json",
    output_dir: str = "/data/checkpoints/hypothesis_only_v3",
    model_name: str = "roberta-large",
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    gradient_accumulation: int = 2,
    num_epochs: int = 3,
    max_length: int = 512,  # Match v1/v3 verifier input shape exactly
    seed: int = 42,
) -> dict:
    """Train masked-evidence verifier to test for surface shortcuts.

    Architectural constraint (2026-04-14 pre-flight fix): this baseline
    MUST use the same model architecture, same tokenization shape, and
    same max_length as the v1/v3 verifier it is being compared to.
    The prior version used DeBERTa-v3-large with claim-only tokenization
    at max_length=128 — which is a DIFFERENT architecture and a
    DIFFERENT input format, so the resulting "97.71%" number was never
    a true hypothesis-only baseline.

    Defaults updated:
      * model_name: roberta-large (matches v1/v3)
      * max_length: 512 (matches v1/v3)
      * learning_rate: 2e-5 (matches v1/v3 trainer)
      * batch_size: 32 × grad_accum 2 (matches v1/v3 effective batch 64)
      * data_path: /data/verifier_training_data_v3.json (v3 taxonomy)
      * output_dir: /data/checkpoints/hypothesis_only_v3 (no v1 overwrite)
    """
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

    # Masked evidence sentinel.  This is a single non-trivial token
    # that is passed as the second sequence to the tokenizer so the
    # (claim, evidence) pair shape matches v1/v3 exactly, but carries
    # no information the model could exploit.  We deliberately avoid
    # passing the empty string because some HF tokenizers collapse
    # empty second sequences to single-sequence layout, which would
    # DEFEAT the purpose of this baseline.
    MASKED_EVIDENCE = tokenizer.pad_token or "[PAD]"

    class HypothesisOnlyDataset(Dataset):
        """Dataset that masks evidence to a single pad token.

        The claim is preserved verbatim; the evidence position is
        replaced by a single pad token.  Tokenization is
        ``(claim, "[PAD]")`` as a PAIR — same shape as v1/v3 —
        truncated to max_length=512.  The model sees the full claim
        structure and a single-token evidence field; any accuracy
        above majority-class (~66%) must come from claim-side lexical
        shortcuts."""

        def __init__(self, data, tokenizer, max_length, masked_evidence):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.masked_evidence = masked_evidence

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            claim = item["claim"]
            orig_label = item["label"]
            label = 1 if orig_label == 1 else 0

            encoding = self.tokenizer(
                claim,
                self.masked_evidence,  # PAIR shape, matches v1/v3 verifier
                max_length=self.max_length,
                padding="max_length",
                truncation="only_first",
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

    train_ds = HypothesisOnlyDataset(
        train_ex, tokenizer, max_length, MASKED_EVIDENCE,
    )
    val_ds = HypothesisOnlyDataset(
        val_ex, tokenizer, max_length, MASKED_EVIDENCE,
    )
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
def main(
    data_path: str = "/data/verifier_training_data_v3.json",
    output_dir: str = "/data/checkpoints/hypothesis_only_v3",
    model_name: str = "roberta-large",
    num_epochs: int = 3,
    max_length: int = 512,
    seed: int = 42,
):
    """Run the masked-evidence baseline with CLI-overridable paths.

    Default targets the v3 training data + v3 output dir so an
    accidental invocation does NOT overwrite the v1 HO checkpoint.
    Override with ``--data-path /data/verifier_training_data.json``
    to re-measure the v1 HO baseline under the fixed methodology.
    """
    result = train_hypothesis_only.remote(
        data_path=data_path,
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=num_epochs,
        max_length=max_length,
        seed=seed,
    )
    print(f"\nFinal result: {result}")
