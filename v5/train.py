"""Training loop for ClaimGuard-CXR v5.

Responsibilities:
- Build a PyTorch Dataset from GroundBench JSONL.
- Build image tensors (resized to 224x224, normalized with BiomedCLIP stats).
- Build tokenized text inputs (claim + evidence).
- Multi-objective loss (see v5/losses.py).
- Adversarial HO filter: optionally drop examples solved by the evidence-blind
  baseline with ≥ 0.9 confidence.
- Image-masked degradation sanity check after each epoch.
- Per-epoch validation on image-grounded val set.
- Checkpoint the best-on-val with a manifest.

Modal entrypoint is `verifact/v5/modal/train_v5.py`; this file is import-only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .losses import LossOutput, LossWeights, total_loss
from .model import ImageGroundedVerifier, V5Config, build_v5_model, build_v5_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class V5TrainConfig:
    train_jsonl: Path
    val_jsonl: Path
    out_dir: Path
    image_root: Path
    max_text_tokens: int = 256
    image_size: int = 224
    batch_size: int = 32
    grad_accum: int = 2
    epochs: int = 4
    lr_encoders: float = 1e-5
    lr_heads: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    loss_weights: LossWeights = field(default_factory=LossWeights)
    seed: int = 17
    mixed_precision: str = "bf16"  # {"fp32", "bf16", "fp16"}
    use_wandb: bool = True
    wandb_project: str = "claimguard-v5"
    wandb_run_name: str | None = None
    image_masked_prob: float = 0.2  # probability of applying image-masked consistency step per batch
    contrast_prob: float = 0.5       # probability of computing contrastive loss per batch
    uncertainty_prob: float = 0.2    # probability of running MC-dropout sampling per batch (cost: uncertainty_n_samples extra forwards)
    uncertainty_n_samples: int = 5   # number of MC forward passes when the uncertainty loss fires
    adversarial_ho_filter: bool = False  # downweight HO-solvable examples during training
    ho_filter_weights_path: Path | None = None  # path to per-example weights written by the HO filter
    freeze_image_layers: int = 8
    freeze_text_layers: int = 16
    grounding_enabled: bool = True


class GroundBenchDataset(Dataset):
    """Load rows from a GroundBench JSONL, build (image, claim, evidence, label) samples.

    If `weights_path` is provided, loads per-example training weights from a
    JSONL mapping {"row_idx": int, "weight": float} produced by the adversarial
    HO filter. Missing rows default to weight 1.0.
    """

    def __init__(
        self,
        jsonl_path: Path,
        image_root: Path,
        tokenizer: Any,
        cfg: V5TrainConfig,
        *,
        weights_path: Path | None = None,
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
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.weights: list[float] = [1.0] * len(self.rows)
        if weights_path is not None and weights_path.exists():
            with weights_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    w = json.loads(line)
                    idx = w["row_idx"]
                    if 0 <= idx < len(self.weights):
                        self.weights[idx] = float(w["weight"])
            logger.info(
                "loaded HO filter weights from %s (mean=%.3f)",
                weights_path,
                sum(self.weights) / max(1, len(self.weights)),
            )
        logger.info("Loaded %d resolved-GT rows from %s", len(self.rows), jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_or_abs: str) -> torch.Tensor:
        from PIL import Image
        from torchvision import transforms

        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.image_root / p
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open image %s (%s); substituting zeros", p, exc)
            img = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), 0)
        transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=BIOMEDCLIP_MEAN, std=BIOMEDCLIP_STD),
            ]
        )
        return transform(img)

    def _encode_text(self, claim: str, evidence: str | None) -> dict[str, torch.Tensor]:
        # Use the tokenizer's text_pair API so the RoBERTa </s></s> separator is
        # injected as real special tokens, not as three characters (< / s >).
        # Truncate only the evidence side to preserve the claim intact.
        if evidence is None or not evidence.strip():
            enc = self.tokenizer(
                claim,
                max_length=self.cfg.max_text_tokens,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                claim,
                evidence,
                max_length=self.cfg.max_text_tokens,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = self._load_image(row["image_path"])
        enc = self._encode_text(row["claim_text"], row.get("evidence_text"))
        label = 1 if row["gt_label"] == "CONTRADICTED" else 0
        out = {
            "pixel_values": image,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }
        # Optionally attach grounding target if bbox/mask is carried in the row.
        # (handled at collate stage via a sidecar channel; elided here for brevity)
        return out


def _build_optimizer(model: ImageGroundedVerifier, cfg: V5TrainConfig) -> AdamW:
    head_params = []
    encoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(h) for h in ("verdict_head", "score_head", "grounding_head", "fusion_blocks", "fusion_norm", "image_proj", "text_proj", "image_domain_adapter", "verdict_token")):
            head_params.append(p)
        else:
            encoder_params.append(p)
    return AdamW(
        [
            {"params": encoder_params, "lr": cfg.lr_encoders},
            {"params": head_params, "lr": cfg.lr_heads},
        ],
        weight_decay=cfg.weight_decay,
    )


def _lr_schedule(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# (Adversarial HO filter moved to v5/ho_filter.py. Import via:
#   from .ho_filter import run_ho_filter
# to avoid circular imports with this module's train_v5.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sanity check: image-masked degradation
# ---------------------------------------------------------------------------


def image_masked_degradation(
    model: ImageGroundedVerifier,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Return (full_acc, image_masked_acc). Gap should be ≥ 15 pp by epoch 3."""
    model.eval()
    n = 0
    full_correct = 0
    masked_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out_full = model(pv, ii, am)
            out_mask = model(pv, ii, am, image_masked=True)
            p_full = out_full["verdict_logits"].argmax(dim=-1)
            p_mask = out_mask["verdict_logits"].argmax(dim=-1)
            full_correct += int((p_full == y).sum())
            masked_correct += int((p_mask == y).sum())
            n += y.numel()
    return full_correct / max(1, n), masked_correct / max(1, n)


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_acc: float
    val_full_acc: float
    val_masked_acc: float
    image_masked_gap: float
    cls: float
    ground: float
    consist: float
    contrast: float
    uncert: float
    wall_time_sec: float


def train_v5(cfg: V5TrainConfig, model_cfg: V5Config | None = None) -> list[EpochStats]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model_cfg = model_cfg or V5Config(
        freeze_image_layers=cfg.freeze_image_layers,
        freeze_text_layers=cfg.freeze_text_layers,
        grounding_enabled=cfg.grounding_enabled,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_v5_tokenizer(model_cfg)
    model = build_v5_model(model_cfg).to(device)

    # If HO filter is enabled, ensure weights exist on disk before loading.
    ho_weights_path: Path | None = cfg.ho_filter_weights_path
    if cfg.adversarial_ho_filter:
        if ho_weights_path is None:
            ho_weights_path = cfg.out_dir / "ho_filter_weights.jsonl"
        if not ho_weights_path.exists():
            logger.info("HO filter enabled but weights not found at %s; running filter now", ho_weights_path)
            from .ho_filter import run_ho_filter

            run_ho_filter(
                train_jsonl=cfg.train_jsonl,
                output_weights_path=ho_weights_path,
                tokenizer=tokenizer,
                device=device,
                max_text_tokens=cfg.max_text_tokens,
                seed=cfg.seed,
            )

    train_ds = GroundBenchDataset(
        cfg.train_jsonl, cfg.image_root, tokenizer, cfg,
        weights_path=ho_weights_path if cfg.adversarial_ho_filter else None,
    )
    val_ds = GroundBenchDataset(cfg.val_jsonl, cfg.image_root, tokenizer, cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    optimizer = _build_optimizer(model, cfg)
    total_steps = (len(train_loader) // max(1, cfg.grad_accum)) * cfg.epochs
    scheduler = _lr_schedule(optimizer, total_steps=total_steps, warmup_steps=cfg.warmup_steps)

    if cfg.use_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={**asdict(cfg), "model_cfg": asdict(model_cfg)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb init failed: %s", exc)
            cfg.use_wandb = False

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    stats: list[EpochStats] = []
    step_global = 0

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.mixed_precision, torch.float32)
    use_autocast = cfg.mixed_precision in {"bf16", "fp16"}

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_cls = running_ground = running_consist = running_contrast = running_uncert = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            ii = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast and device.type == "cuda"):
                out_full = model(pv, ii, am, return_grounding=cfg.loss_weights.ground > 0)
                if cfg.loss_weights.consist > 0 and torch.rand(1).item() < cfg.image_masked_prob:
                    out_mask = model(pv, ii, am, image_masked=True)
                    logits_masked = out_mask["verdict_logits"]
                else:
                    logits_masked = None

                # Contrastive evidence pair: shuffle batch evidence to produce mismatches.
                # CRITICAL: applies ONLY to SUPPORTED anchors (label==0). Applying it to
                # CONTRADICTED anchors inverts the objective (for a CONTRADICTED claim,
                # shuffled evidence may be MORE correct than the original, so rewarding
                # score_matched > score_mismatched is wrong-direction).
                # Also use identity-rejecting permutation so i->i pairs (which would be
                # "matched" masquerading as "mismatched") cannot occur.
                support_score_matched_for_loss = out_full["support_score"]
                score_mis = None
                if cfg.loss_weights.contrast > 0 and torch.rand(1).item() < cfg.contrast_prob:
                    supported_mask = (y == 0)
                    n_sup = int(supported_mask.sum().item())
                    if n_sup >= 2:
                        sup_idx = supported_mask.nonzero(as_tuple=True)[0]
                        perm = torch.randperm(n_sup, device=device)
                        identity = (perm == torch.arange(n_sup, device=device))
                        if identity.any():
                            perm = (perm + 1) % n_sup
                        out_mis = model(pv[sup_idx], ii[sup_idx][perm], am[sup_idx][perm])
                        support_score_matched_for_loss = out_full["support_score"][sup_idx]
                        score_mis = out_mis["support_score"]

                # MC-dropout samples for uncertainty loss (v5.4 only; no-op
                # elsewhere because cfg.loss_weights.uncert == 0).
                uncertainty_mc_probs = None
                if cfg.loss_weights.uncert > 0 and torch.rand(1).item() < cfg.uncertainty_prob:
                    mc_samples = []
                    for _ in range(cfg.uncertainty_n_samples):
                        out_mc = model(pv, ii, am)
                        mc_samples.append(torch.softmax(out_mc["verdict_logits"], dim=-1))
                    uncertainty_mc_probs = torch.stack(mc_samples, dim=0)

                # Per-example weights from adversarial HO filter (identity
                # weights if the filter is disabled or weights not loaded).
                example_weights = batch.get("weight")
                if example_weights is not None:
                    example_weights = example_weights.to(device, non_blocking=True)

                loss_out: LossOutput = total_loss(
                    weights=cfg.loss_weights,
                    verdict_logits_full=out_full["verdict_logits"],
                    verdict_logits_masked=logits_masked,
                    support_score_matched=support_score_matched_for_loss,
                    support_score_mismatched=score_mis,
                    labels=y,
                    grounding_logits=out_full.get("grounding_logits"),
                    grounding_target=None,  # populated when MS-CXR rows are in-batch; elided here
                    grounding_mask=None,
                    uncertainty_mc_probs=uncertainty_mc_probs,
                    example_weights=example_weights,
                )
                loss = loss_out.total / cfg.grad_accum

            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_global += 1

            running_loss += float(loss_out.total.detach())
            running_cls += float(loss_out.cls.detach())
            running_ground += float(loss_out.ground.detach())
            running_consist += float(loss_out.consist.detach())
            running_contrast += float(loss_out.contrast.detach())
            running_uncert += float(loss_out.uncert.detach())
            n_batches += 1

            if cfg.use_wandb and step_global % 10 == 0:
                try:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": float(loss_out.total.detach()),
                            "train/cls": float(loss_out.cls.detach()),
                            "train/ground": float(loss_out.ground.detach()),
                            "train/consist": float(loss_out.consist.detach()),
                            "train/contrast": float(loss_out.contrast.detach()),
                            "train/uncert": float(loss_out.uncert.detach()),
                            "train/lr_encoders": optimizer.param_groups[0]["lr"],
                            "train/lr_heads": optimizer.param_groups[1]["lr"],
                            "step": step_global,
                        }
                    )
                except Exception:
                    pass

        # Validation
        full_acc, masked_acc = image_masked_degradation(model, val_loader, device)
        gap = full_acc - masked_acc

        es = EpochStats(
            epoch=epoch,
            train_loss=running_loss / max(1, n_batches),
            val_acc=full_acc,
            val_full_acc=full_acc,
            val_masked_acc=masked_acc,
            image_masked_gap=gap,
            cls=running_cls / max(1, n_batches),
            ground=running_ground / max(1, n_batches),
            consist=running_consist / max(1, n_batches),
            contrast=running_contrast / max(1, n_batches),
            uncert=running_uncert / max(1, n_batches),
            wall_time_sec=time.time() - t0,
        )
        stats.append(es)
        logger.info(
            "epoch %d | train_loss=%.4f val_full=%.4f val_masked=%.4f gap=%.4f (%.1fs)",
            epoch, es.train_loss, full_acc, masked_acc, gap, es.wall_time_sec,
        )
        if cfg.use_wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "val/full_acc": full_acc,
                        "val/masked_acc": masked_acc,
                        "val/gap": gap,
                        "val/train_loss": es.train_loss,
                    }
                )
            except Exception:
                pass

        # Early-abort heuristic per arch doc §6.3: abort if gap < 10 pp at epoch 3.
        if epoch >= 2 and gap < 0.10:
            logger.error(
                "Image-masked gap %.4f < 0.10 at epoch %d — aborting per §6.3 contingency",
                gap, epoch,
            )
            break

        if full_acc > best_acc:
            best_acc = full_acc
            ckpt = cfg.out_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(cfg),
                    "epoch": epoch,
                    "val_full_acc": full_acc,
                    "val_masked_acc": masked_acc,
                    "image_masked_gap": gap,
                },
                ckpt,
            )
            logger.info("saved best checkpoint to %s (val_acc=%.4f)", ckpt, full_acc)

    # Manifest
    manifest = {
        "version": "5.0.0-dev",
        "best_val_acc": best_acc,
        "epochs": [asdict(s) for s in stats],
        "config": asdict(cfg),
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats
