"""Adversarial hypothesis-only (HO) filter for ClaimGuard-CXR v5.

The HO filter is the paper's core training-time mitigation for evidence-blindness.
It operates in three steps:

1. Train a text-only RoBERTa-large classifier on `(claim, evidence) -> label`.
   This baseline has no access to the image; any correct prediction it makes is
   evidence of a textual shortcut in the claim or evidence.
2. Score every training row with this baseline. Rows that the HO model solves
   with high confidence at the true label are flagged as shortcut-solvable.
3. Write per-row weights to a JSONL: shortcut-solvable rows are downweighted
   (default 0.2), the rest stay at 1.0. The v5 trainer reads this file and
   multiplies per-example losses accordingly.

See `ARCHITECTURE_V5_0_EVIDENCE_BLINDNESS.md` §6.3 for the full rationale and
`tests/test_ho_filter.py` for unit tests of the scoring + weighting logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class _HODataset(Dataset):
    """Text-only dataset: tokenized (claim, evidence) pairs -> label."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Any,
        max_text_tokens: int,
    ):
        self.rows: list[dict] = []
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}:
                    self.rows.append(r)
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        claim = row["claim_text"]
        evidence = row.get("evidence_text") or ""
        if evidence.strip():
            enc = self.tokenizer(
                claim,
                evidence,
                max_length=self.max_text_tokens,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                claim,
                max_length=self.max_text_tokens,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        label = 1 if row["gt_label"] == "CONTRADICTED" else 0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


def run_ho_filter(
    *,
    train_jsonl: Path,
    output_weights_path: Path,
    tokenizer: Any,
    device: torch.device | str = "cuda",
    max_text_tokens: int = 256,
    confidence_threshold: float = 0.7,
    downweight: float = 0.2,
    n_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    seed: int = 17,
    backbone: str = "roberta-large",
) -> dict:
    """Train an HO baseline on (claim, evidence) -> label and write per-row weights.

    Args:
        train_jsonl: v5 GroundBench training JSONL.
        output_weights_path: destination for per-row {row_idx, weight} JSONL.
        tokenizer: tokenizer instance compatible with the HO backbone (defaults
            to RoBERTa-large). For BPE compatibility we re-use the v5 tokenizer.
        device: "cuda" or "cpu" or torch.device.
        max_text_tokens: max tokens for the claim+evidence pair.
        confidence_threshold: rows where the HO baseline assigns the TRUE label
            a probability above this threshold are considered shortcut-solvable.
        downweight: training weight applied to shortcut-solvable rows.
        n_epochs: training epochs for the HO baseline.
        batch_size: batch size for HO training/scoring.
        lr: learning rate.
        seed: random seed.
        backbone: HF model id for the text backbone.

    Returns:
        Summary dict with counts and distributions.
    """
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)

    device = torch.device(device) if not isinstance(device, torch.device) else device
    ds = _HODataset(train_jsonl, tokenizer, max_text_tokens)
    if len(ds) == 0:
        raise RuntimeError(f"no resolved-GT rows in {train_jsonl}")
    loader_train = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_score = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(n_epochs):
        running = 0.0
        n_batches = 0
        for batch in loader_train:
            ii = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(input_ids=ii, attention_mask=am, labels=y)
            out.loss.backward()
            optimizer.step()
            running += float(out.loss.detach())
            n_batches += 1
        logger.info("HO epoch %d mean_loss=%.4f", epoch, running / max(1, n_batches))

    # Score
    model.eval()
    weights = [1.0] * len(ds)
    n_downweighted = 0
    n_supported_down = 0
    n_contradicted_down = 0
    with torch.no_grad():
        for batch in loader_score:
            ii = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            idx = batch["row_idx"]
            logits = model(input_ids=ii, attention_mask=am).logits
            probs = F.softmax(logits, dim=-1)
            true_label_prob = probs.gather(1, y.unsqueeze(-1)).squeeze(-1)
            for j in range(ii.size(0)):
                if float(true_label_prob[j]) > confidence_threshold:
                    weights[int(idx[j])] = downweight
                    n_downweighted += 1
                    if int(y[j]) == 0:
                        n_supported_down += 1
                    else:
                        n_contradicted_down += 1

    # Write
    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    with output_weights_path.open("w") as f:
        for i, w in enumerate(weights):
            f.write(json.dumps({"row_idx": i, "weight": w}) + "\n")

    summary = {
        "n_rows": len(ds),
        "n_downweighted": n_downweighted,
        "fraction_downweighted": n_downweighted / max(1, len(ds)),
        "n_supported_downweighted": n_supported_down,
        "n_contradicted_downweighted": n_contradicted_down,
        "confidence_threshold": confidence_threshold,
        "downweight": downweight,
        "backbone": backbone,
        "weights_path": str(output_weights_path),
    }
    logger.info("HO filter summary: %s", summary)
    return summary
