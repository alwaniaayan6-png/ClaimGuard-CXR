"""White-box hidden-state probe for hallucination detection.

Reimplemented from the description in Hardy et al. 2025, "ReXTrust: A White-Box
Approach to Fine-Grained Hallucination Detection in Radiology Reports"
(arXiv:2412.15264; PMLR v281 hardy25a). The original ReXTrust code and weights
are not publicly released as of April 2026, so this module is an honest
reimplementation — not a replication — and the paper labels the corresponding
results row accordingly ("White-box hidden-state probe, inspired by ReXTrust").

Protocol:

1. For each (image, generated_claim) pair, run the underlying LVLM in teacher-
   forcing mode with the claim tokens appended to the visual prompt.
2. Extract the per-token last-layer hidden states for the claim tokens.
3. Aggregate across tokens (mean pool by default; ReXTrust uses last-token
   hidden state - we provide both options).
4. Train a small logistic classifier or MLP on (hidden_state -> silver_label).

We apply to two LVLMs for an architecture-independence signal (per
pre-flight B3 resolution): MedVersa (``hyzhou/MedVersa``) and MAIRA-2
(``microsoft/maira-2``). Results are reported per-backbone in Table 1.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ProbeTrainConfig:
    arch: str = "logistic"          # logistic | mlp
    hidden_dim: int = 256           # MLP hidden dim
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 64
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    pool: str = "mean"              # mean | last | cls


@dataclass
class ProbeEvalResult:
    backbone: str
    n_train: int
    n_val: int
    n_test: int
    train_auroc: float
    val_auroc: float
    test_auroc: float
    test_precision_at_halluc: float
    test_recall_at_halluc: float


class HiddenStateExtractor:
    """Extracts last-layer hidden states from an LVLM for a (image, claim) pair.

    Subclass per backbone. The base interface ``extract(pil, claim) -> np.ndarray``
    returns a 1-D embedding.
    """

    name: str

    def extract(self, pil, claim: str) -> np.ndarray:
        raise NotImplementedError


class _BaseHFExtractor(HiddenStateExtractor):
    model_id: str
    pool: str = "mean"

    def __init__(self, device: torch.device | str = "cuda", pool: str = "mean"):
        import os
        from transformers import AutoProcessor, AutoModelForCausalLM
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.pool = pool
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, **kwargs,
        ).to(self.device).eval()

    _CLAIM_PREFIX = "Review the following claim about the chest radiograph and continue the "
    _CLAIM_LEAD = "radiology assessment:\nCLAIM: "
    _CLAIM_TAIL = "\nASSESSMENT:"

    def _inputs(self, pil, claim: str) -> tuple[dict, int, int]:
        """Return (inputs, claim_start_idx, claim_end_idx) for claim-token pooling.

        Tokenizes the prompt with and without the claim so we can compute the
        exact token-position range that contains the claim. ReXTrust's protocol
        pools over claim-token hidden states, not over the trailing
        "ASSESSMENT:" prompt token.
        """
        prompt_before = self._CLAIM_PREFIX + self._CLAIM_LEAD
        prompt_full = prompt_before + claim + self._CLAIM_TAIL
        inputs = self.processor(images=pil, text=prompt_full, return_tensors="pt").to(self.device)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids_before = tokenizer(prompt_before, add_special_tokens=False)["input_ids"]
        ids_claim = tokenizer(claim, add_special_tokens=False)["input_ids"]
        full_ids = inputs.get("input_ids")
        offset = 0
        if full_ids is not None:
            try:
                offset = max(0, full_ids.shape[-1] - (len(ids_before) + len(ids_claim) + 2))
            except Exception:
                offset = 0
        claim_start = len(ids_before) + offset
        claim_end = claim_start + len(ids_claim)
        return inputs, claim_start, claim_end

    @torch.no_grad()
    def extract(self, pil, claim: str) -> np.ndarray:
        inputs, claim_start, claim_end = self._inputs(pil, claim)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, d)
        h = last_hidden[0]                       # (seq, d)
        seq_len = h.shape[0]
        claim_start = max(0, min(seq_len - 1, claim_start))
        claim_end = max(claim_start + 1, min(seq_len, claim_end))
        claim_slice = h[claim_start:claim_end]
        if claim_slice.shape[0] == 0:
            claim_slice = h
        if self.pool == "last":
            emb = claim_slice[-1]
        elif self.pool == "cls":
            emb = h[0]
        else:
            emb = claim_slice.mean(dim=0)
        return emb.float().cpu().numpy()


class MAIRA2Extractor(_BaseHFExtractor):
    name = "maira-2"
    model_id = "microsoft/maira-2"


class MedVersaExtractor(HiddenStateExtractor):
    """MedVersa uses a custom ``registry.get_model_class('medomni')`` loader.

    We try to import the rajpurkarlab medomni registry. If the package is not
    installed in the current environment, the extractor raises at init time so
    callers can fall back to a MAIRA-2-only run.
    """

    name = "medversa"
    model_id = "hyzhou/MedVersa"

    def __init__(self, device: torch.device | str = "cuda", pool: str = "mean"):
        try:
            from medomni.common.registry import registry  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "medomni package not installed. Install from "
                "https://github.com/rajpurkarlab/MedVersa to enable MedVersa extractor."
            ) from exc
        model_cls = registry.get_model_class("medomni")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.pool = pool
        self.model = model_cls.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def extract(self, pil, claim: str) -> np.ndarray:
        if not hasattr(self.model, "get_hidden_states"):
            raise RuntimeError("MedVersa model does not expose get_hidden_states; update medomni or subclass")
        h = self.model.get_hidden_states(pil, claim)  # (seq, d) expected
        if isinstance(h, torch.Tensor):
            arr = h
            if arr.ndim == 3:
                arr = arr[0]
            if self.pool == "last":
                vec = arr[-1]
            elif self.pool == "cls":
                vec = arr[0]
            else:
                vec = arr.mean(dim=0)
            return vec.float().cpu().numpy()
        return np.asarray(h, dtype=np.float32)


class _Probe(nn.Module):
    def __init__(self, d_in: int, arch: str = "logistic", hidden: int = 256):
        super().__init__()
        if arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = len(pos) * len(neg)
    wins = 0
    ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total


def train_probe(
    embeddings_train: np.ndarray,
    labels_train: np.ndarray,
    embeddings_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    cfg: ProbeTrainConfig | None = None,
    backbone_name: str = "unknown",
    device: torch.device | str = "cpu",
) -> tuple[_Probe, ProbeEvalResult]:
    """Train a probe and return (model, eval metrics).

    Labels follow the convention: 1 = hallucinated (CONTRADICTED), 0 = SUPPORTED.
    """
    cfg = cfg or ProbeTrainConfig()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    n_total = len(embeddings_train)
    if n_total == 0:
        raise ValueError("empty training set")
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * cfg.val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    Xtr = torch.tensor(embeddings_train[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(labels_train[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.tensor(embeddings_train[val_idx], dtype=torch.float32, device=device)
    yva_np = labels_train[val_idx]
    Xte = torch.tensor(embeddings_test, dtype=torch.float32, device=device)
    yte_np = labels_test

    probe = _Probe(Xtr.shape[1], arch=cfg.arch, hidden=cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_pos = max(1, int(ytr.sum().item()))
    n_neg = max(1, len(ytr) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(cfg.epochs):
        probe.train()
        order = torch.randperm(Xtr.shape[0], device=device)
        total_loss = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            logits = probe(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            logger.info("probe epoch %d loss=%.4f", epoch + 1, total_loss / len(ytr))

    probe.eval()
    with torch.no_grad():
        tr_score = torch.sigmoid(probe(Xtr)).cpu().numpy()
        va_score = torch.sigmoid(probe(Xva)).cpu().numpy()
        te_score = torch.sigmoid(probe(Xte)).cpu().numpy()
    ytr_np = ytr.cpu().numpy()

    # Precision/recall at argmax-F1 threshold selected on validation.
    thresholds = np.linspace(0.05, 0.95, 37)
    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        yhat = (va_score >= thr).astype(int)
        tp = int(((yhat == 1) & (yva_np == 1)).sum())
        fp = int(((yhat == 1) & (yva_np == 0)).sum())
        fn = int(((yhat == 0) & (yva_np == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    yhat_te = (te_score >= best_thr).astype(int)
    tp = int(((yhat_te == 1) & (yte_np == 1)).sum())
    fp = int(((yhat_te == 1) & (yte_np == 0)).sum())
    fn = int(((yhat_te == 0) & (yte_np == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = ProbeEvalResult(
        backbone=backbone_name,
        n_train=len(tr_idx),
        n_val=len(val_idx),
        n_test=len(te_np := labels_test),
        train_auroc=_auroc(ytr_np, tr_score),
        val_auroc=_auroc(yva_np, va_score),
        test_auroc=_auroc(yte_np, te_score),
        test_precision_at_halluc=prec,
        test_recall_at_halluc=rec,
    )
    return probe, metrics


def extract_embeddings(
    extractor: HiddenStateExtractor,
    pairs: Iterable[dict],
    image_root: Path,
    out_npz: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract embeddings for a stream of (image, claim, label) rows.

    Args:
        extractor: HiddenStateExtractor.
        pairs: iterable of dicts with keys ``image_path``, ``claim_text``, ``label``
            (label in {SUPPORTED, CONTRADICTED}).
        image_root: base dir for relative image paths.
        out_npz: path for saving ``embeddings`` + ``labels`` + ``claim_ids``.

    Returns:
        (embeddings, labels, claim_ids).
    """
    from PIL import Image
    rows = list(pairs)
    embs: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    t0 = time.time()
    for i, r in enumerate(rows):
        ipath = Path(r["image_path"])
        if not ipath.is_absolute():
            ipath = image_root / ipath
        try:
            pil = Image.open(ipath).convert("RGB")
        except Exception as exc:
            logger.warning("image load failed for %s: %s", ipath, exc)
            continue
        try:
            emb = extractor.extract(pil, str(r.get("claim_text", "")))
        except Exception as exc:
            logger.warning("extract failed at row %d: %s", i, exc)
            continue
        embs.append(emb)
        labels.append(1 if r["label"] == "CONTRADICTED" else 0)
        ids.append(str(r.get("claim_id", f"row{i}")))
        if (i + 1) % 100 == 0:
            logger.info("extract %d/%d (%.1fs/row)", i + 1, len(rows),
                        (time.time() - t0) / (i + 1))
    if not embs:
        raise RuntimeError("no embeddings extracted")
    X = np.stack(embs, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, embeddings=X, labels=y, claim_ids=np.asarray(ids))
    return X, y, ids
