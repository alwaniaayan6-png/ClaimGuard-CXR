"""Primary evaluation: image-grounded ground truth.

Inputs: (1) a checkpoint path for ImageGroundedVerifier, (2) a GroundBench JSONL
with radiologist-anchored GT, (3) a config specifying site name and alpha
levels. Outputs: per-site metric JSON with 95% bootstrap CIs + conformal FDR
numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..conformal.doubly_robust_cfbh import doubly_robust_cfbh
from ..conformal.inverted_cfbh import inverted_cfbh
from ..conformal.utils import compute_empirical_fdr
from ..conformal.weighted_cfbh import weighted_cfbh
from ..model import ImageGroundedVerifier, V5Config, build_v5_model, build_v5_tokenizer
from ..provenance import TrustTier, apply_provenance_gate
from ..train import GroundBenchDataset, V5TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    ckpt_path: Path
    test_jsonl: Path
    cal_jsonl: Path
    image_root: Path
    out_json: Path
    site_name: str
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    include_no_gt: bool = False
    bootstrap_replicates: int = 10_000
    conformal_variants: tuple[str, ...] = ("inverted",)  # {"inverted","weighted","doubly_robust"}
    feature_columns: tuple[str, ...] = ("claim_length", "age")
    apply_provenance_gate: bool = True
    device: str = "cuda"


def _load_checkpoint(cfg: EvalConfig) -> tuple[ImageGroundedVerifier, object]:
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model_cfg = V5Config(**{k: v for k, v in ckpt["model_cfg"].items() if k in V5Config.__dataclass_fields__})
    model = build_v5_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device).eval()
    tok = build_v5_tokenizer(model_cfg)
    return model, tok


def _score_dataset(model: ImageGroundedVerifier, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            out = model(pv, ii, am)
            probs = torch.softmax(out["verdict_logits"], dim=-1)[:, 0]  # P(not-contradicted)
            scores.extend(probs.cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    return np.asarray(scores), np.asarray(labels, dtype=int)


def _features_from_rows(rows: list[dict], columns: Iterable[str]) -> np.ndarray:
    feats: list[list[float]] = []
    for r in rows:
        row_feats: list[float] = []
        for c in columns:
            if c == "claim_length":
                row_feats.append(float(len(r.get("claim_text", ""))))
            elif c == "age":
                a = r.get("age")
                row_feats.append(float(a) if a is not None else 0.0)
            else:
                v = r.get(c)
                try:
                    row_feats.append(float(v))
                except (TypeError, ValueError):
                    row_feats.append(0.0)
        feats.append(row_feats)
    return np.asarray(feats, dtype=float)


def _compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Basic classification metrics with 95% bootstrap CI on accuracy + AUROC."""
    from sklearn.metrics import roc_auc_score

    preds = (scores < 0.5).astype(int)  # P(not-contradicted) < 0.5 → contradicted prediction
    acc = float((preds == labels).mean())
    try:
        auroc = float(roc_auc_score(labels, 1 - scores))  # 1 - scores so that positive class = contradicted
    except ValueError:
        auroc = float("nan")
    # Per-class recall
    pos = labels == 1
    neg = labels == 0
    recall_contra = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    recall_supp = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return {
        "accuracy": acc,
        "auroc": auroc,
        "contra_recall": recall_contra,
        "supp_recall": recall_supp,
    }


def _bootstrap_ci(metric_fn, scores: np.ndarray, labels: np.ndarray, n: int = 10_000, seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(labels)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        vals[i] = metric_fn(scores[idx], labels[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


@dataclass
class EvalResult:
    site: str
    n_test: int
    n_cal: int
    metrics: dict[str, float]
    metrics_ci: dict[str, tuple[float, float]]
    conformal: list[dict]
    provenance_gate_applied: bool
    per_class_n: dict[str, int]
    coverage_stats: dict[str, int]


def evaluate(cfg: EvalConfig) -> EvalResult:
    model, tok = _load_checkpoint(cfg)

    # Load raw rows for features + provenance
    test_rows: list[dict] = []
    cal_rows: list[dict] = []
    with cfg.test_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_rows.append(json.loads(line))
    with cfg.cal_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cal_rows.append(json.loads(line))

    # Filter to resolved GT for primary metrics
    if not cfg.include_no_gt:
        test_rows = [r for r in test_rows if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}]
        cal_rows = [r for r in cal_rows if r.get("gt_label") in {"SUPPORTED", "CONTRADICTED"}]

    # Build loader-style batches by re-using the training dataset
    train_like_cfg = V5TrainConfig(
        train_jsonl=cfg.test_jsonl,
        val_jsonl=cfg.cal_jsonl,
        out_dir=cfg.out_json.parent,
        image_root=cfg.image_root,
    )
    test_ds = GroundBenchDataset(cfg.test_jsonl, cfg.image_root, tok, train_like_cfg)
    cal_ds = GroundBenchDataset(cfg.cal_jsonl, cfg.image_root, tok, train_like_cfg)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False, num_workers=2)

    test_scores, test_labels = _score_dataset(model, test_loader, cfg.device)
    cal_scores, cal_labels = _score_dataset(model, cal_loader, cfg.device)

    metrics = _compute_metrics(test_scores, test_labels)

    # Bootstrap CIs for accuracy + auroc
    metrics_ci: dict[str, tuple[float, float]] = {}
    metrics_ci["accuracy"] = _bootstrap_ci(
        lambda s, l: float(((s < 0.5).astype(int) == l).mean()),
        test_scores,
        test_labels,
        n=cfg.bootstrap_replicates,
    )
    try:
        from sklearn.metrics import roc_auc_score

        metrics_ci["auroc"] = _bootstrap_ci(
            lambda s, l: float(roc_auc_score(l, 1 - s)),
            test_scores,
            test_labels,
            n=cfg.bootstrap_replicates,
        )
    except Exception:
        metrics_ci["auroc"] = (float("nan"), float("nan"))

    # Conformal layer
    cal_contra = cal_scores[cal_labels == 1]
    conformal_results: list[dict] = []
    for alpha in cfg.alphas:
        for variant in cfg.conformal_variants:
            if variant == "inverted":
                r = inverted_cfbh(test_scores, cal_contra, alpha=alpha)
            elif variant == "weighted":
                cal_feats = _features_from_rows(cal_rows, cfg.feature_columns)
                test_feats = _features_from_rows(test_rows, cfg.feature_columns)
                # restrict cal rows to contradicted for density-ratio fit
                cal_contra_mask = cal_labels == 1
                r = weighted_cfbh(
                    test_scores=test_scores,
                    cal_contra_scores=cal_contra,
                    cal_features=cal_feats[cal_contra_mask],
                    test_features=test_feats,
                    alpha=alpha,
                )
            elif variant == "doubly_robust":
                cal_feats = _features_from_rows(cal_rows, cfg.feature_columns)
                test_feats = _features_from_rows(test_rows, cfg.feature_columns)
                cal_contra_mask = cal_labels == 1
                r = doubly_robust_cfbh(
                    test_scores=test_scores,
                    cal_contra_scores=cal_contra,
                    cal_features=cal_feats[cal_contra_mask],
                    test_features=test_feats,
                    alpha=alpha,
                )
            else:
                raise ValueError(f"unknown conformal variant {variant}")
            green = r.green_mask
            # Apply provenance gate after conformal
            if cfg.apply_provenance_gate:
                for i, row in enumerate(test_rows):
                    if not green[i]:
                        continue
                    tier_raw = row.get("evidence_trust_tier", "unknown")
                    try:
                        tier = TrustTier(tier_raw)
                    except ValueError:
                        tier = TrustTier.UNKNOWN if hasattr(TrustTier, "UNKNOWN") else None
                    # downgrade: only trusted / independent stay in green
                    if tier_raw not in {"trusted", "independent"}:
                        green[i] = False
            diag = compute_empirical_fdr(
                green, test_labels, alpha=alpha, n_bootstrap=cfg.bootstrap_replicates
            )
            conformal_results.append(
                {
                    "variant": variant,
                    **asdict(diag),
                    "bh_threshold": float(r.bh_threshold),
                }
            )

    result = EvalResult(
        site=cfg.site_name,
        n_test=len(test_labels),
        n_cal=len(cal_labels),
        metrics=metrics,
        metrics_ci={k: (float(a), float(b)) for k, (a, b) in metrics_ci.items()},
        conformal=conformal_results,
        provenance_gate_applied=cfg.apply_provenance_gate,
        per_class_n={"supported": int((test_labels == 0).sum()), "contradicted": int((test_labels == 1).sum())},
        coverage_stats={
            "n_resolved": len(test_labels),
        },
    )
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(asdict(result), indent=2, default=str))
    return result
