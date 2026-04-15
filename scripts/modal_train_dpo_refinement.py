"""Task 3c — Modal H100 DPO-style refinement trainer for ClaimGuard v4.

This script implements the third piece of the ACL 2025 "Dually
Self-Improved" counterfactual pipeline.  Upstream, Task 3a identifies
causal token spans (``data/augmentation/causal_term_identifier.py``),
Task 3b generates counterfactual paraphrases (``counterfactual_generator.py``).
This script takes a JSON file of ``{claim, evidence, counterfactual}``
preference pairs and fine-tunes the v3 verifier into v4.

**KNOWN ISSUE (2026-04-14 pre-flight, blocks launch).**
The DPO loss below uses ``chosen = counterfactual, rejected = original``.
For a binary classifier where BOTH sides carry the same true label
(contradicted), the naive DPO margin-maximization objective will pull
the ORIGINAL claim's contradicted score DOWN to expand the margin
between chosen and rejected, which is the opposite of what the plan
wants — the goal is to make the verifier SAME-SCORE on original and
counterfactual (consistency), not to prefer one over the other.

The correct fix is one of:

    (A) Replace the DPO loss with a consistency regularizer:
        ``L = ||logit(y|C) - logit(y|C')||²``
        This pulls the two scores toward each other without biasing
        the absolute direction. Matches the Dually Self-Improved
        paper's core intuition.

    (B) Regenerate counterfactuals so that "rejected" is a TRUE
        causal-breaking variant (one where the causal token IS
        changed, producing the OPPOSITE verdict). Then the DPO pair
        has different true labels and the margin-maximization is
        correct. Requires rewriting Task 3b.

Resolution required before launching Task 3.  Do not run
``modal run --detach scripts/modal_train_dpo_refinement.py`` until
one of (A) or (B) is implemented.  Until then, ship v3 (Task 2
checkpoint) as the paper's v-current and document the Task 3 abort
in the Limitations section.

Why a custom DPO loss (not ``trl.DPOTrainer``)?
-----------------------------------------------
``trl.DPOTrainer`` (0.9.x) is hardwired to ``AutoModelForCausalLM`` — it
computes log-probs over token-level logits.  Our v3 verifier is a
RoBERTa-large cross-encoder classifier, so we reimplement the Rafailov
et al. 2023 objective for a 2-class classifier:

    chosen   = (counterfactual, evidence)  ← preserves causal tokens
    rejected = (original claim, evidence)  ← may carry surface shortcuts

    L_DPO = -E[log σ(β · (log π_θ(y|chosen) - log π_θ(y|rejected)
                          - log π_ref(y|chosen) + log π_ref(y|rejected)))]

where y=1 is the contradicted class, π_θ is the trainable policy, and
π_ref is a frozen copy of v3.  **See KNOWN ISSUE above — this is the
signature semantic concern that blocks launch.**

Hyperparameters (from plan + Rafailov 2023)
-------------------------------------------
* β = 0.1                       — standard DPO temperature
* lr = 5e-6                     — same as v3 fine-tune
* single epoch                  — the counterfactual pool is small
* batch_size = 8                — ~24 GB VRAM at max_length=256
* gradient_clip = 1.0           — standard
* freeze first 8 RoBERTa layers — trains only layers 9-24 + head
* early-stop: KL > 5 OR reward margin < 0 for 50 consecutive steps
* log reward stats every 100 steps

Outputs
-------
``/data/checkpoints/verifier_binary_v4_dpo/best_verifier.pt`` with the
same ``{"encoder": ..., "head": ...}`` format as v3, plus
``training_history.json`` containing per-step loss/margin/KL/reward.

The pure helpers (``DPOEarlyStopTracker``, ``load_preference_pairs``,
``DPOTrainingConfig``) are testable without torch/Modal; the heavy
training loop lives inside ``_run_training`` and is only imported
inside the Modal container.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

VOLUME_NAME = "claimguard-data"
APP_NAME = "claimguard-dpo-refinement"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class DPOTrainingConfig:
    """Hyperparameters for the DPO refinement run.

    Defaults match the Task 3 plan.  All fields are JSON-serializable.
    """

    base_checkpoint: str = (
        "/data/checkpoints/verifier_binary_v3/best_verifier.pt"
    )
    output_dir: str = "/data/checkpoints/verifier_binary_v4_dpo"
    preference_data: str = "/data/counterfactual_preference_pairs_v3.json"
    hf_backbone: str = "roberta-large"
    max_length: int = 256
    batch_size: int = 8
    lr: float = 5e-6
    beta: float = 0.1
    num_epochs: int = 1
    gradient_clip: float = 1.0
    kl_max: float = 5.0
    margin_min: float = 0.0
    patience: int = 50
    log_every: int = 100
    freeze_first_n_layers: int = 8
    seed: int = 42
    target_label: int = 1  # contradicted class

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "DPOTrainingConfig":
        return cls(**json.loads(s))


# ---------------------------------------------------------------------------
# Pure helpers — no torch / modal dependency.
# ---------------------------------------------------------------------------


def load_preference_pairs(
    path: "os.PathLike[str] | str",
) -> list[dict[str, str]]:
    """Load counterfactual preference pairs from a JSON file.

    Expected schema is a list of dicts, each with at least:

        {
            "claim": "original contradicted claim",
            "evidence": "associated evidence text",
            "counterfactual": "paraphrased variant"
        }

    Also accepted:

        {"claim": ..., "evidence": ..., "counterfactuals": [str, str, ...]}

    which is flattened into one row per counterfactual.  Rows missing
    ``claim`` or with no valid counterfactuals are skipped silently.
    Rows where ``counterfactual == claim`` are also skipped (nothing to
    learn from).

    Args:
        path: Path to the preference-pairs JSON file.

    Returns:
        A list of ``{"claim", "evidence", "counterfactual"}`` dicts,
        all fields guaranteed to be non-empty strings.

    Raises:
        ValueError: If the top-level JSON is not a list.
        FileNotFoundError: If the path does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"expected JSON list at {path}, got {type(raw).__name__}"
        )

    pairs: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            continue
        evidence = row.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = ""

        cfs: list[str] = []
        single = row.get("counterfactual")
        if isinstance(single, str) and single.strip():
            cfs.append(single)
        multi = row.get("counterfactuals")
        if isinstance(multi, list):
            for c in multi:
                if isinstance(c, str) and c.strip():
                    cfs.append(c)

        claim_stripped = claim.strip()
        for cf in cfs:
            cf_stripped = cf.strip()
            if cf_stripped and cf_stripped != claim_stripped:
                pairs.append({
                    "claim": claim_stripped,
                    "evidence": evidence.strip(),
                    "counterfactual": cf_stripped,
                })

    return pairs


class DPOEarlyStopTracker:
    """Early-stop monitor for the DPO training loop.

    Tracks two independent violation streaks:

    * **KL streak** — how many consecutive steps had ``kl > kl_max``.
    * **Margin streak** — how many consecutive steps had
      ``margin < margin_min``.

    Either streak reaching ``patience`` triggers an early stop and the
    ``reason`` attribute is populated with a human-readable string.  A
    single "good" step resets the corresponding streak to zero.

    This class is stateful but does not depend on torch, so it's unit-
    testable without a GPU.
    """

    def __init__(
        self,
        *,
        kl_max: float = 5.0,
        margin_min: float = 0.0,
        patience: int = 50,
    ) -> None:
        self.kl_max = float(kl_max)
        self.margin_min = float(margin_min)
        self.patience = int(patience)
        self._kl_streak = 0
        self._margin_streak = 0
        self.reason: Optional[str] = None

    def update(self, kl: float, margin: float) -> bool:
        """Record one training step's (kl, margin). Returns ``True`` if
        an early-stop condition just tripped.
        """
        if kl > self.kl_max:
            self._kl_streak += 1
        else:
            self._kl_streak = 0

        if margin < self.margin_min:
            self._margin_streak += 1
        else:
            self._margin_streak = 0

        if self._kl_streak >= self.patience:
            self.reason = (
                f"KL > {self.kl_max} for {self.patience} consecutive steps"
            )
            return True
        if self._margin_streak >= self.patience:
            self.reason = (
                f"reward margin < {self.margin_min} for "
                f"{self.patience} consecutive steps"
            )
            return True
        return False

    @property
    def kl_streak(self) -> int:
        return self._kl_streak

    @property
    def margin_streak(self) -> int:
        return self._margin_streak


def format_reward_histogram(
    rewards: "collections.deque[float] | list[float]",
    *,
    n_buckets: int = 10,
) -> str:
    """Build a printable histogram of recent reward margins.

    Used by the training loop for log_every-step diagnostic output.
    Pure function — testable without torch.

    Args:
        rewards: Iterable of per-step reward-margin floats.
        n_buckets: Number of histogram buckets.

    Returns:
        A single-line string like
        ``"reward hist [min=-0.12 max=0.48 mean=0.19]: 1 3 5 8 12 7 4 2 1 1"``
        or ``"reward hist: <empty>"`` if the input is empty.
    """
    values = list(rewards)
    if not values:
        return "reward hist: <empty>"
    lo = min(values)
    hi = max(values)
    mean = sum(values) / len(values)
    if hi == lo:
        # Degenerate: all values equal — one bucket gets everything.
        counts = [0] * n_buckets
        counts[0] = len(values)
    else:
        width = (hi - lo) / n_buckets
        counts = [0] * n_buckets
        for v in values:
            idx = min(int((v - lo) / width), n_buckets - 1)
            counts[idx] += 1
    bars = " ".join(str(c) for c in counts)
    return (
        f"reward hist [min={lo:.3f} max={hi:.3f} mean={mean:.3f}]: {bars}"
    )


# ---------------------------------------------------------------------------
# Heavy helpers — torch required.  Imported lazily inside Modal.
# ---------------------------------------------------------------------------


def _dpo_classifier_loss(
    policy_logits_chosen: Any,
    policy_logits_rejected: Any,
    ref_logits_chosen: Any,
    ref_logits_rejected: Any,
    *,
    beta: float,
    target_label: int = 1,
) -> tuple[Any, float, float, float]:
    """DPO loss adapted to a 2-class classifier.

    Computes log p(target_label | x) = logit[target] - logsumexp(logits)
    for each of {policy, reference} × {chosen, rejected}, then applies
    the Rafailov 2023 objective:

        loss = -E[log σ(β (policy_logratio - ref_logratio))]
        policy_logratio = log π_θ(y|chosen) - log π_θ(y|rejected)
        ref_logratio    = log π_ref(y|chosen) - log π_ref(y|rejected)

    Args:
        policy_logits_chosen / policy_logits_rejected: 2-class logits
            from the trainable model.  Shape ``(B, 2)``.
        ref_logits_chosen / ref_logits_rejected: 2-class logits from
            the frozen reference.  Shape ``(B, 2)``.
        beta: DPO temperature (typically 0.1).
        target_label: Which class we're comparing (1 = contradicted).

    Returns:
        ``(loss_tensor, margin_float, kl_proxy_float, chosen_reward_mean)``:

        * ``loss_tensor`` is a scalar torch Tensor ready for backward().
        * ``margin_float`` is the mean (chosen_reward - rejected_reward),
          detached to a python float — fed into ``DPOEarlyStopTracker``.
        * ``kl_proxy_float`` is the mean absolute log-ratio deviation
          between policy and reference on chosen inputs — a cheap
          proxy for KL used by the early-stop rule.
        * ``chosen_reward_mean`` is the mean chosen reward (for logging).
    """
    import torch
    import torch.nn.functional as F

    def log_prob(logits: Any) -> Any:
        return logits[:, target_label] - torch.logsumexp(logits, dim=-1)

    policy_chosen_lp = log_prob(policy_logits_chosen)
    policy_rejected_lp = log_prob(policy_logits_rejected)
    ref_chosen_lp = log_prob(ref_logits_chosen)
    ref_rejected_lp = log_prob(ref_logits_rejected)

    policy_ratio = policy_chosen_lp - policy_rejected_lp
    ref_ratio = ref_chosen_lp - ref_rejected_lp

    dpo_logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(dpo_logits).mean()

    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
        rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)
        margin = (chosen_reward - rejected_reward).mean().item()
        kl_proxy = (policy_chosen_lp - ref_chosen_lp).abs().mean().item()
        chosen_mean = chosen_reward.mean().item()

    return loss, margin, kl_proxy, chosen_mean


def _freeze_first_n_layers(encoder: Any, n: int) -> int:
    """Freeze embeddings + first ``n`` transformer layers of a RoBERTa-family
    encoder.

    Returns the number of parameters frozen.  Raises ``AttributeError``
    if the encoder shape is unrecognized.
    """
    frozen = 0

    def _freeze_module(module: Any) -> None:
        nonlocal frozen
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += p.numel()

    # RoBERTa AutoModel lays out: model.embeddings + model.encoder.layer[i]
    if hasattr(encoder, "embeddings") and hasattr(encoder, "encoder"):
        emb = encoder.embeddings
        enc = encoder.encoder
    elif hasattr(encoder, "roberta"):
        emb = encoder.roberta.embeddings
        enc = encoder.roberta.encoder
    else:
        raise AttributeError(
            "Expected RoBERTa-style encoder with .embeddings and "
            ".encoder.layer — got "
            f"{type(encoder).__name__}."
        )

    _freeze_module(emb)
    layers = enc.layer
    for i in range(min(n, len(layers))):
        _freeze_module(layers[i])
    return frozen


def _run_training(config_json: str) -> dict[str, Any]:
    """Full training loop.  Runs inside the Modal container.

    Imports torch / transformers lazily so the outer module stays
    importable without them (for unit tests of the pure helpers).
    """
    import random

    import torch
    from transformers import AutoModel, AutoTokenizer

    config = DPOTrainingConfig.from_json(config_json)

    # Reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DPO training on device=%s", device)

    pairs = load_preference_pairs(config.preference_data)
    logger.info("Loaded %d preference pairs", len(pairs))
    if not pairs:
        raise RuntimeError(
            f"No preference pairs found in {config.preference_data}"
        )

    tokenizer = AutoTokenizer.from_pretrained(config.hf_backbone)

    # ---- Policy (trainable) ----
    policy_encoder = AutoModel.from_pretrained(config.hf_backbone).to(device)
    policy_head = torch.nn.Linear(
        policy_encoder.config.hidden_size, 2
    ).to(device)

    state = torch.load(config.base_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "encoder" in state:
        policy_encoder.load_state_dict(state["encoder"], strict=False)
        if "head" in state:
            policy_head.load_state_dict(state["head"], strict=False)
    else:
        policy_encoder.load_state_dict(state, strict=False)
    policy_encoder.train()
    policy_head.train()

    frozen = _freeze_first_n_layers(
        policy_encoder, config.freeze_first_n_layers
    )
    logger.info(
        "Froze %d policy parameters in first %d layers",
        frozen, config.freeze_first_n_layers,
    )

    # ---- Reference (frozen) ----
    ref_encoder = AutoModel.from_pretrained(config.hf_backbone).to(device)
    ref_head = torch.nn.Linear(
        ref_encoder.config.hidden_size, 2
    ).to(device)
    if isinstance(state, dict) and "encoder" in state:
        ref_encoder.load_state_dict(state["encoder"], strict=False)
        if "head" in state:
            ref_head.load_state_dict(state["head"], strict=False)
    else:
        ref_encoder.load_state_dict(state, strict=False)
    for p in list(ref_encoder.parameters()) + list(ref_head.parameters()):
        p.requires_grad = False
    ref_encoder.eval()
    ref_head.eval()

    # ---- Optimizer ----
    trainable = [
        p for p in list(policy_encoder.parameters())
        + list(policy_head.parameters())
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    tracker = DPOEarlyStopTracker(
        kl_max=config.kl_max,
        margin_min=config.margin_min,
        patience=config.patience,
    )
    reward_buffer: collections.deque[float] = collections.deque(
        maxlen=config.log_every
    )

    def _forward(encoder: Any, head: Any, batch: list[dict[str, str]]) -> tuple[Any, Any]:
        """Run encoder+head on (counterfactual, evidence) and (claim, evidence)."""
        chosen_firsts = [b["counterfactual"] for b in batch]
        chosen_seconds = [b["evidence"] for b in batch]
        rejected_firsts = [b["claim"] for b in batch]
        rejected_seconds = [b["evidence"] for b in batch]

        chosen_enc = tokenizer(
            chosen_firsts, chosen_seconds,
            padding=True, truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)
        rejected_enc = tokenizer(
            rejected_firsts, rejected_seconds,
            padding=True, truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)

        chosen_out = encoder(
            **chosen_enc, return_dict=True
        ).last_hidden_state[:, 0, :]
        rejected_out = encoder(
            **rejected_enc, return_dict=True
        ).last_hidden_state[:, 0, :]
        return head(chosen_out), head(rejected_out)

    rng = random.Random(config.seed)
    global_step = 0
    stop_reason: Optional[str] = None
    history: list[dict[str, Any]] = []

    for epoch in range(config.num_epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            if not batch:
                continue

            policy_chosen_logits, policy_rejected_logits = _forward(
                policy_encoder, policy_head, batch
            )
            with torch.no_grad():
                ref_chosen_logits, ref_rejected_logits = _forward(
                    ref_encoder, ref_head, batch
                )

            loss, margin, kl_proxy, chosen_reward = _dpo_classifier_loss(
                policy_chosen_logits,
                policy_rejected_logits,
                ref_chosen_logits,
                ref_rejected_logits,
                beta=config.beta,
                target_label=config.target_label,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()

            reward_buffer.append(margin)
            history.append({
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "margin": float(margin),
                "kl_proxy": float(kl_proxy),
                "chosen_reward": float(chosen_reward),
            })

            if tracker.update(kl_proxy, margin):
                stop_reason = tracker.reason
                logger.warning(
                    "DPO early-stop triggered at step %d: %s",
                    global_step, stop_reason,
                )
                break

            if global_step % config.log_every == 0:
                logger.info(
                    "step=%d loss=%.4f margin=%.4f kl=%.4f chosen_reward=%.4f "
                    "| kl_streak=%d margin_streak=%d",
                    global_step, loss.item(), margin, kl_proxy,
                    chosen_reward,
                    tracker.kl_streak, tracker.margin_streak,
                )
                logger.info(format_reward_histogram(reward_buffer))

            global_step += 1

        if stop_reason is not None:
            break

    # ---- Save checkpoint ----
    # Reviewer-flagged (2026-04-14 pre-flight, BUG G): on abort the
    # prior code still wrote ``best_verifier.pt``, which meant
    # downstream eval pipelines would silently grab a diverged,
    # aborted checkpoint as "v4".  We now write to a distinct
    # filename on abort, drop a visible ``ABORTED`` marker file next
    # to it, and only emit ``best_verifier.pt`` on a clean run.
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stop_reason is None:
        ckpt_path = out_dir / "best_verifier.pt"
    else:
        ckpt_path = out_dir / "aborted_verifier.pt"
        marker_path = out_dir / "ABORTED"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(
                f"DPO training aborted at step {global_step}.\n"
                f"Reason: {stop_reason}\n"
                f"This checkpoint should NOT be used as v4. "
                f"Ship v3 instead and document the abort in the paper.\n"
            )
    torch.save(
        {
            "encoder": policy_encoder.state_dict(),
            "head": policy_head.state_dict(),
            "config": asdict(config),
            "stop_reason": stop_reason,
        },
        str(ckpt_path),
    )
    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "stop_reason": stop_reason}, f)

    return {
        "stop_reason": stop_reason,
        "steps": global_step,
        "pairs_loaded": len(pairs),
        "checkpoint": str(ckpt_path),
        "history_path": str(history_path),
        "final_loss": history[-1]["loss"] if history else None,
        "final_margin": history[-1]["margin"] if history else None,
        "final_kl": history[-1]["kl_proxy"] if history else None,
    }


# ---------------------------------------------------------------------------
# Modal wiring — guarded so tests can import this module without modal.
# ---------------------------------------------------------------------------

app: Any = None
volume: Any = None
train_dpo_refinement_remote: Any = None

try:
    import modal as _modal  # noqa: WPS433

    _image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            [
                "torch==2.3.0",
                "transformers==4.40.0",
                "numpy<2",
                "huggingface_hub<0.25",
            ]
        )
    )
    app = _modal.App(APP_NAME, image=_image)
    volume = _modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    @app.function(  # type: ignore[misc]
        gpu="H100",
        timeout=60 * 60 * 3,  # 3h cap
        volumes={"/data": volume},
    )
    def train_dpo_refinement_remote(config_json: str) -> dict[str, Any]:  # noqa: F811
        """Modal-entry stub that calls the heavy trainer."""
        return _run_training(config_json)

except Exception as _modal_err:  # noqa: BLE001
    logger.info(
        "Modal unavailable at module-import time (%s); "
        "pure helpers still importable.",
        _modal_err,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> DPOTrainingConfig:
    defaults = DPOTrainingConfig()
    parser = argparse.ArgumentParser(
        description="ClaimGuard DPO refinement trainer (Modal H100)."
    )
    parser.add_argument(
        "--base-checkpoint", default=defaults.base_checkpoint,
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--preference-data", default=defaults.preference_data,
    )
    parser.add_argument("--hf-backbone", default=defaults.hf_backbone)
    parser.add_argument(
        "--max-length", type=int, default=defaults.max_length,
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument(
        "--num-epochs", type=int, default=defaults.num_epochs,
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=defaults.gradient_clip,
    )
    parser.add_argument("--kl-max", type=float, default=defaults.kl_max)
    parser.add_argument(
        "--margin-min", type=float, default=defaults.margin_min,
    )
    parser.add_argument(
        "--patience", type=int, default=defaults.patience,
    )
    parser.add_argument(
        "--log-every", type=int, default=defaults.log_every,
    )
    parser.add_argument(
        "--freeze-first-n-layers",
        type=int, default=defaults.freeze_first_n_layers,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)

    return DPOTrainingConfig(
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        preference_data=args.preference_data,
        hf_backbone=args.hf_backbone,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        num_epochs=args.num_epochs,
        gradient_clip=args.gradient_clip,
        kl_max=args.kl_max,
        margin_min=args.margin_min,
        patience=args.patience,
        log_every=args.log_every,
        freeze_first_n_layers=args.freeze_first_n_layers,
        seed=args.seed,
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if argv is None:
        argv = sys.argv[1:]
    config = _parse_cli(argv)
    logger.info("DPO config: %s", asdict(config))

    if app is None or train_dpo_refinement_remote is None:
        logger.error(
            "Modal is not available in this environment — cannot launch "
            "the training run.  Install modal and re-run, or invoke "
            "_run_training() directly for local dry-runs."
        )
        return 1

    with app.run():
        result = train_dpo_refinement_remote.remote(config.to_json())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "APP_NAME",
    "DPOEarlyStopTracker",
    "DPOTrainingConfig",
    "VOLUME_NAME",
    "format_reward_histogram",
    "load_preference_pairs",
    "main",
]
