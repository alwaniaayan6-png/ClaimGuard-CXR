"""Gururangan-style text-only artifact audit of the ClaimGuard training data.

Trains a text-only RoBERTa-base classifier on ``(claim, evidence) -> label`` with
**no image input**, measuring how much of the verification signal is solvable
from text alone. Reported as ``text_only_acc - majority_class_acc``: the delta
above the trivial class-prior baseline.

Per pre-flight reviewer M4, the M4-corrected target is:

* On the **pre-HO-filter** training distribution: text-only delta is expected
  to be large (shortcut-solvable claims are plentiful).
* On the **post-HO-filter** training distribution (examples downweighted by
  ``adversarial_ho_filter``): text-only delta should drop to <= 2pp.

If the delta is >5pp on post-HO data, the HO filter is not effectively
removing text-shortcut claims and the v6.0 mitigation story needs reframing.

Also measures HO-filter activation on the real-RRG test claims (M6): what
fraction of naturally-occurring paraphrastic hallucinations are flagged by
the HO filter as text-solvable? This checks whether the mitigation, trained
on synthetic negation-flipped claims, transfers to real hallucinations.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ArtifactAuditResult:
    config: str
    n_train: int
    n_test: int
    test_accuracy: float
    majority_class_acc: float
    delta_vs_majority_pp: float
    per_class_f1: dict[str, float]


@dataclass
class HoActivationResult:
    n_claims: int
    n_text_solvable: int
    rate_text_solvable: float
    mean_ho_score: float
    p50_ho_score: float
    p95_ho_score: float


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def train_text_only_classifier(
    train_jsonl: Path,
    test_jsonl: Path,
    *,
    weights_jsonl: Path | None = None,
    config_name: str = "default",
    model_id: str = "roberta-base",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
    device: torch.device | str = "cuda",
) -> ArtifactAuditResult:
    """Train a RoBERTa-base text-only classifier on (claim, evidence) -> label.

    If ``weights_jsonl`` is provided, applies per-example sample weights as
    produced by ``v5/modal/ho_filter.py`` (post-HO-filter training).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device(device) if not isinstance(device, torch.device) else device
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device).train()

    raw_train_rows = _load_jsonl(train_jsonl)
    test_rows = _load_jsonl(test_jsonl)
    # HO-filter writes weights positionally over the resolved-GT rows only.
    # Mirror GroundBenchDataset's filter (v5/ho_filter.py:50) to align indices.
    train_rows = [r for r in raw_train_rows if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}]

    weights_by_idx: dict[int, float] = {}
    if weights_jsonl is not None:
        for r in _load_jsonl(weights_jsonl):
            if "row_idx" not in r:
                continue
            weights_by_idx[int(r["row_idx"])] = float(r.get("weight", 1.0))

    def _prepare(rows: list[dict], include_weights: bool) -> tuple[list[str], list[str], list[int], list[float]]:
        claims = [str(r.get("claim_text", "")) for r in rows]
        evids = [str(r.get("evidence_text", "")) for r in rows]
        labels = [1 if r.get("gt_label") == "CONTRADICTED" else 0 for r in rows]
        if include_weights and weights_by_idx:
            ws = [weights_by_idx.get(i, 1.0) for i in range(len(rows))]
        else:
            ws = [1.0] * len(rows)
        return claims, evids, labels, ws

    tr_claims, tr_evids, tr_labels, tr_weights = _prepare(train_rows, include_weights=True)
    te_claims, te_evids, te_labels, _ = _prepare(test_rows, include_weights=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(tr_claims)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            enc = tok(
                [tr_claims[i] for i in idx],
                [tr_evids[i] for i in idx],
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            y = torch.tensor([tr_labels[i] for i in idx], device=device)
            w = torch.tensor([tr_weights[i] for i in idx], dtype=torch.float32, device=device)
            out = model(**enc, labels=y)
            losses = nn.functional.cross_entropy(out.logits, y, reduction="none")
            loss = (losses * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        logger.info("artifact_audit epoch %d loss=%.4f", epoch + 1, total_loss / max(1, n))

    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for s in range(0, len(te_claims), batch_size):
            enc = tok(
                te_claims[s:s + batch_size],
                te_evids[s:s + batch_size],
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

    te_labels_arr = np.asarray(te_labels)
    preds_arr = np.asarray(preds)
    acc = float((preds_arr == te_labels_arr).mean())
    label_counts = Counter(te_labels_arr.tolist())
    majority_class = max(label_counts, key=label_counts.get)
    majority_acc = float(label_counts[majority_class] / max(1, len(te_labels_arr)))

    per_class_f1: dict[str, float] = {}
    for cls, name in [(0, "SUPPORTED"), (1, "CONTRADICTED")]:
        tp = int(((preds_arr == cls) & (te_labels_arr == cls)).sum())
        fp = int(((preds_arr == cls) & (te_labels_arr != cls)).sum())
        fn = int(((preds_arr != cls) & (te_labels_arr == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[name] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return ArtifactAuditResult(
        config=config_name,
        n_train=n,
        n_test=len(te_labels_arr),
        test_accuracy=acc,
        majority_class_acc=majority_acc,
        delta_vs_majority_pp=(acc - majority_acc) * 100.0,
        per_class_f1=per_class_f1,
    )


def measure_ho_activation_on_rrg_claims(
    claims_jsonl: Path,
    ho_scores_jsonl: Path,
    *,
    threshold: float = 0.7,
) -> HoActivationResult:
    """Compute HO-filter activation rate on real-RRG claim set (M6 check).

    Args:
        claims_jsonl: real-RRG claims from ``rrg_generate`` + claim extraction.
        ho_scores_jsonl: scores from ``ho_filter.score_external_claims`` or
            equivalent, one row per claim_id with key ``ho_score``.
        threshold: score above which a claim is deemed text-solvable.

    Returns:
        ``HoActivationResult`` with activation rate and score distribution.
    """
    claims = _load_jsonl(claims_jsonl)
    by_id: dict[str, float] = {}
    for r in _load_jsonl(ho_scores_jsonl):
        by_id[str(r.get("claim_id", ""))] = float(r.get("ho_score", 0.5))
    scores: list[float] = []
    n_solvable = 0
    for c in claims:
        cid = str(c.get("claim_id", ""))
        s = by_id.get(cid)
        if s is None:
            continue
        scores.append(s)
        if s >= threshold:
            n_solvable += 1
    if not scores:
        return HoActivationResult(
            n_claims=0, n_text_solvable=0, rate_text_solvable=0.0,
            mean_ho_score=0.0, p50_ho_score=0.0, p95_ho_score=0.0,
        )
    arr = np.asarray(scores)
    return HoActivationResult(
        n_claims=len(arr),
        n_text_solvable=n_solvable,
        rate_text_solvable=float(n_solvable / len(arr)),
        mean_ho_score=float(arr.mean()),
        p50_ho_score=float(np.median(arr)),
        p95_ho_score=float(np.quantile(arr, 0.95)),
    )
