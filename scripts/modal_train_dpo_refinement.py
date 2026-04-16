"""Task 3c — Modal H100 counterfactual-consistency refinement for ClaimGuard v4.

This script fine-tunes the v3 verifier into v4 using **counterfactual
consistency regularization** (default) or the legacy DPO loss
(``--loss-mode dpo``, kept for research comparison only).  Upstream,
Task 3a identifies causal token spans
(``data/augmentation/causal_term_identifier.py``) and Task 3b generates
counterfactual paraphrases that preserve the causal tokens but vary
the surface form (``counterfactual_generator.py``).  Each
``{claim, evidence, counterfactual}`` triple gives a pair of inputs
``(x = claim + evidence, x' = counterfactual + evidence)`` that
SHOULD receive the same classifier output because they share the
same causal semantics and same ground-truth label.

Why consistency regularization, not DPO (2026-04-14 pre-flight fix)
-------------------------------------------------------------------
A prior version of this file used a DPO-style loss with
``chosen = counterfactual, rejected = original``.  For a binary
classifier where both sides carry the same true label (contradicted),
DPO's margin-maximization pulls the ORIGINAL claim's contradicted
score DOWN to expand the margin, which is the opposite of what we
want: we want the verifier to produce the SAME score on both sides,
regardless of surface form.

An Opus pre-flight reviewer flagged this as plan-breaking; a follow-
up literature dive confirmed that the correct formulation for this
problem is the **R-Drop / UDA** symmetric-KL consistency loss, not
DPO.  The canonical formulation (R-Drop, Liang et al., NeurIPS 2021,
arXiv:2106.14448; UDA, Xie et al., NeurIPS 2020, arXiv:1904.12848):

    L = CE(x,  y) + CE(x', y)
      + λ_cons · (1/2) · [KL(p(·|x)  || stop_grad p(·|x'))
                          + KL(p(·|x') || stop_grad p(·|x))]

where ``p(·|x)`` is the softmax over the 2 classes, ``y`` is the true
label, and ``λ_cons ≈ 0.5`` is the consistency weight (R-Drop's GLUE
sweep settled on ``α ∈ {0.1, 0.5, 1.0}`` depending on task).  The
``stop_grad`` on the KL targets follows UDA's Eq. 1 and prevents
double-counting gradients through the "target" side of each KL term.

This loss is already the standard in the text-classification
literature for counterfactual / augmentation invariance and does
not suffer from DPO's margin-maximization pathology.  See the
2026-04-14 literature review in the same commit's message for the
citation list.

Citation caveat: the original ClaimGuard sprint plan cited an
"ACL 2025 Dually Self-Improved" paper, which the 2026-04-14
literature review could not find in arXiv / OpenAlex / ACL Anthology
/ Semantic Scholar under any variant of the title.  The proposal
doc has been updated to cite R-Drop (NeurIPS 2021), UDA (NeurIPS
2020), and Kaushik et al. "Learning the Difference that Makes a
Difference with Counterfactually-Augmented Data" (ICLR 2020,
arXiv:1909.12434) instead, all of which are the actual source
material for this approach.

Loss mode summary
-----------------
* ``loss_mode="consistency"`` (default, correct): the loss above.
  Trains v4 to agree on (x, x') without biasing the absolute
  direction.  Literature-backed and the path that should be used
  for production runs.
* ``loss_mode="dpo"`` (legacy, broken): the DPO formulation with
  chosen=counterfactual and rejected=original.  Kept ONLY for
  research comparison against the consistency mode; DO NOT use
  this for the paper's headline v4 checkpoint.

Shared hyperparameters (both modes)
-----------------------------------
* lr = 5e-6                     — same as v3 fine-tune
* single epoch                  — the counterfactual pool is small
* batch_size = 8                — ~24 GB VRAM at max_length=256
* gradient_clip = 1.0           — standard
* freeze first 8 RoBERTa layers — trains only layers 9-24 + head
* log reward / CE stats every 100 steps

Consistency-mode-specific
-------------------------
* ce_weight = 1.0
* consistency_weight = 0.5 (R-Drop GLUE default range)
* no DPO early-stop (consistency loss is well-behaved; training runs
  the full epoch unless CE blows up above 2.5, which would indicate
  catastrophic label collapse)

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
    """Hyperparameters for the Task 3 refinement run.

    The filename + class name retain "DPO" for backward compatibility
    with existing references, but the default ``loss_mode`` is now
    ``"consistency"`` (R-Drop style symmetric-KL regularization), which
    is the literature-standard and correct approach for training a
    classifier to be invariant between an original claim and its
    counterfactual paraphrase.  See the top-of-file docstring for the
    full rationale.

    Defaults match the Task 3 plan.  All fields are JSON-serializable.
    """

    base_checkpoint: str = (
        "/data/checkpoints/verifier_binary_v3/best_verifier.pt"
    )
    output_dir: str = "/data/checkpoints/verifier_binary_v4_dpo"
    preference_data: str = "/data/counterfactual_preference_pairs_v3.json"
    # Mixed-mode only: path to the full v3 training data for sampling
    # faithful (label=0) examples to mix with the contradicted+cf pairs.
    # Ignored when loss_mode != "consistency_mixed".
    full_training_data: str = "/data/verifier_training_data_v3.json"
    # Mixed-mode only: number of faithful examples to sample per
    # contradicted+cf pair.  1.0 = balanced; 0.5 = under-sample
    # faithful (faster training, more reliance on existing v3 weights);
    # 2.0 = over-sample faithful (more conservative, less collapse risk).
    faithful_per_cf: float = 1.0
    hf_backbone: str = "roberta-large"
    max_length: int = 256
    batch_size: int = 8
    lr: float = 5e-6
    num_epochs: int = 1
    gradient_clip: float = 1.0
    log_every: int = 100
    freeze_first_n_layers: int = 8
    seed: int = 42
    target_label: int = 1  # contradicted class (used by old single-class consistency path)

    # Loss mode selection.
    #   "consistency_mixed" — DEFAULT for v4 (post-2026-04-15 fix).
    #     Uses per-example labels and mixes faithful (label=0) with
    #     contradicted+cf (label=1).  Faithful rows get plain CE.
    #     Contradicted+cf rows get CE on both branches + symmetric KL.
    #   "consistency" — LEGACY single-class path.  Hardcodes
    #     target_label=1 in the loss.  Trivially collapses on
    #     contradicted-only training data (verified empirically on
    #     2026-04-15: v4 OpenI acc dropped to 0.327 with per-class
    #     F1 [0.000, 0.493], i.e. predicting CONTRA for everything).
    #     KEPT ONLY FOR REPRODUCIBILITY of the failed run; do not use.
    #   "dpo" — Rafailov 2023 DPO with the chosen/rejected inversion
    #     bug.  Same do-not-use status, kept for ablation.
    loss_mode: str = "consistency_mixed"

    # Consistency-mode hyperparameters (R-Drop defaults)
    ce_weight: float = 1.0          # weight on CE(original) + CE(counterfactual)
    consistency_weight: float = 0.5  # λ on symmetric-KL consistency term
    ce_blowup_threshold: float = 2.5  # abort if mean CE exceeds this

    # DPO-mode hyperparameters (Rafailov 2023 defaults).  Only used
    # when loss_mode="dpo".  Left in the config for reproducibility
    # of the legacy path.
    beta: float = 0.1
    kl_max: float = 5.0
    margin_min: float = 0.0
    patience: int = 50

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


def load_mixed_training_data(
    counterfactual_pairs_path: "os.PathLike[str] | str",
    full_training_data_path: "os.PathLike[str] | str",
    *,
    faithful_per_cf: float = 1.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load mixed counterfactual + faithful training data for v4 v2.

    The 2026-04-15 v4 v1 run failed by trivial collapse because the
    training data was contradicted-only (Task 3a filtered to
    ``label=1`` claims, Task 3b paraphrased those, the trainer's
    ``_consistency_classifier_loss`` hardcoded ``target_label=1``).
    R-Drop on a single-class dataset converges to "always predict
    that class".  v4 OpenI dropped from 0.7545 to 0.327 with
    per-class F1 ``[0.000, 0.493]``.

    The fix is to MIX faithful (label=0) examples sampled from the
    full v3 training data into the same dataloader, with their
    actual labels.  Faithful rows have no counterfactual
    paraphrase (Task 3b only generated cf for contradicted), so
    they get a plain CE loss with no R-Drop term.  Contradicted
    rows with cf get the full R-Drop loss with target=1.

    The returned schema is unified::

        [
            {
                "claim": str,
                "evidence": str,
                "counterfactual": str | None,  # None for faithful
                "label": int,                  # 0 = faithful, 1 = contra
            },
            ...
        ]

    Rows are shuffled with the given seed so that mini-batches are
    a uniform mix of cf-pairs and faithful rows.

    Args:
        counterfactual_pairs_path: Path to the JSON file produced
            by ``scripts/generate_counterfactual_pairs.py``.  Each
            row contributes one (claim, evidence, cf, label=1)
            example.  Multi-cf rows expand to multiple examples.
        full_training_data_path: Path to the full v3 training data
            (e.g. ``/data/verifier_training_data_v3.json``).  Faithful
            (label=0) rows are sampled from here.
        faithful_per_cf: Ratio of faithful examples to cf-pair
            examples.  1.0 = balanced (n_faithful == n_cf_pairs).
            Values >1.0 over-sample faithful (more conservative,
            less collapse risk).  Values <1.0 under-sample
            (faster training, more reliance on existing v3
            weights).
        seed: RNG seed for the shuffle.

    Returns:
        A unified list of mixed training examples, shuffled.
    """
    import random

    # 1. Load the cf pairs (already in the right format, just add label=1)
    cf_rows = load_preference_pairs(counterfactual_pairs_path)
    cf_examples = [
        {
            "claim": r["claim"],
            "evidence": r["evidence"],
            "counterfactual": r["counterfactual"],
            "label": 1,
        }
        for r in cf_rows
    ]
    n_cf = len(cf_examples)

    # 2. Load the full v3 training data and filter to faithful (label=0)
    with open(full_training_data_path, "r", encoding="utf-8") as f:
        all_v3 = json.load(f)
    faithful_pool = [
        r for r in all_v3
        if r.get("label") == 0
        and isinstance(r.get("claim"), str)
        and r.get("claim", "").strip()
    ]

    # 3. Sample N faithful examples (with replacement if pool is too small)
    n_faithful_target = int(round(n_cf * float(faithful_per_cf)))

    # Defensive guard (pre-flight reviewer Finding 4, 2026-04-15):
    # if the faithful pool is empty but we asked for faithful samples,
    # fail loud rather than crashing on `rng.choice([])` later.  This
    # protects against a stale or malformed full_training_data file
    # that has no label=0 rows (e.g. wrong path, or a contradicted-
    # only training set passed by mistake).
    if n_faithful_target > 0 and not faithful_pool:
        raise RuntimeError(
            f"load_mixed_training_data: full_training_data at "
            f"{full_training_data_path} contains no label=0 (faithful) "
            f"rows.  Cannot sample {n_faithful_target} faithful "
            f"examples.  Check the file path and schema.  This is the "
            f"guard that would have prevented v4 v1 from collapsing — "
            f"if this raises, we are about to repeat the bug."
        )

    rng = random.Random(seed)
    if n_faithful_target <= len(faithful_pool):
        faithful_sample = rng.sample(faithful_pool, n_faithful_target)
    else:
        # With replacement
        faithful_sample = [
            rng.choice(faithful_pool) for _ in range(n_faithful_target)
        ]

    # 4. Convert faithful rows to the unified schema (no counterfactual)
    faithful_examples = []
    for r in faithful_sample:
        evidence = r.get("evidence", "")
        if isinstance(evidence, list):
            evidence = " [SEP] ".join(
                str(e).strip() for e in evidence[:2]
                if str(e).strip()
            )
        elif not isinstance(evidence, str):
            evidence = str(evidence)
        faithful_examples.append({
            "claim": r["claim"].strip(),
            "evidence": evidence.strip(),
            "counterfactual": None,
            "label": 0,
        })

    # 5. Mix and shuffle so batches contain both kinds
    mixed = cf_examples + faithful_examples
    rng.shuffle(mixed)

    logger.info(
        "Mixed dataset: %d cf-pair rows (label=1) + %d faithful rows "
        "(label=0) = %d total (faithful_per_cf=%.2f)",
        n_cf, len(faithful_examples), len(mixed), faithful_per_cf,
    )
    return mixed


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


def _consistency_mixed_loss(
    *,
    orig_logits: Any,
    cf_logits: Any,  # may be None if no cf-pairs in this batch
    labels: Any,  # tensor of per-example labels (int64), shape (B,)
    cf_mask: Any,  # tensor of per-example bool, True if row has a cf
    ce_weight: float = 1.0,
    consistency_weight: float = 0.5,
) -> tuple[Any, float, float, float]:
    """Mixed-data consistency loss for the v4 v2 trainer.

    Computes a unified loss over a batch that contains BOTH:

    * Contradicted+counterfactual rows (cf_mask=True, label=1):
      both branches contribute CE on their actual label, plus a
      symmetric-KL consistency term to pull the two branches'
      distributions together.
    * Faithful rows (cf_mask=False, label=0): only the original
      branch exists (no counterfactual was generated for faithful
      examples in Task 3b).  These rows contribute CE only.

    Loss formula::

        L = CE_orig(label)
          + (CE_cf(label_cf) if cf_mask.any() else 0)
          + λ_cons · (1/2) · [KL(p_orig || stop_grad p_cf)
                              + KL(p_cf || stop_grad p_orig)]
            applied only to the cf_mask subset

    The KL term has zero contribution from faithful rows, so the
    loss reduces to plain CE for those.

    The CRITICAL DIFFERENCE from ``_consistency_classifier_loss``
    (the legacy single-class version): per-example labels.  The
    legacy loss hardcoded ``target_label=1`` for every row, which
    works only on a contradicted-only training set.  On a mixed
    set with faithful rows (label=0) it would push the model toward
    "always predict 1" again — exactly the v1 collapse failure
    mode.  This loss reads the actual label from the batch and uses
    it in both CE branches.

    Args:
        orig_logits: 2-class logits on the original (claim, evidence)
            inputs for ALL rows in the batch.  Shape ``(B, 2)``.
        cf_logits: 2-class logits on the counterfactual (cf, evidence)
            inputs for the cf_mask=True subset.  Shape ``(N_cf, 2)``
            where N_cf <= B, OR None if cf_mask is all-False.
        labels: Per-example ground-truth labels.  Shape ``(B,)``,
            int64.  0 = faithful, 1 = contradicted.
        cf_mask: Per-example bool tensor of which rows have a
            counterfactual.  Shape ``(B,)``.
        ce_weight: Multiplier on CE terms.  Default 1.0.
        consistency_weight: Multiplier on symmetric-KL term.
            Default 0.5.

    Returns:
        ``(loss_tensor, ce_mean_float, consistency_kl_float,
        agreement_mean_float)``
    """
    import torch
    import torch.nn.functional as F

    # CE on all rows (their actual labels)
    ce_orig = F.cross_entropy(
        orig_logits, labels, reduction="mean",
    )

    if cf_logits is not None and cf_mask.any():
        # CE on the cf branch — per-example labels (sliced by mask)
        cf_labels = labels[cf_mask]
        ce_cf = F.cross_entropy(
            cf_logits, cf_labels, reduction="mean",
        )

        # Symmetric KL on the cf subset only
        orig_logits_cf = orig_logits[cf_mask]
        log_p_orig = F.log_softmax(orig_logits_cf, dim=-1)
        log_p_cf = F.log_softmax(cf_logits, dim=-1)
        p_orig = log_p_orig.exp()
        p_cf = log_p_cf.exp()
        kl_orig_to_cf = F.kl_div(
            log_p_orig, p_cf.detach(), reduction="batchmean",
        )
        kl_cf_to_orig = F.kl_div(
            log_p_cf, p_orig.detach(), reduction="batchmean",
        )
        consistency_kl = 0.5 * (kl_orig_to_cf + kl_cf_to_orig)

        loss = (
            ce_weight * (ce_orig + ce_cf)
            + consistency_weight * consistency_kl
        )

        with torch.no_grad():
            ce_mean = 0.5 * (ce_orig + ce_cf).item()
            cons_kl = consistency_kl.item()
            # Per-target P agreement signal (label-1 since most cf
            # rows are contradicted in our setup)
            p_orig_y = p_orig.gather(
                1, cf_labels.view(-1, 1).long(),
            ).squeeze(-1)
            p_cf_y = p_cf.gather(
                1, cf_labels.view(-1, 1).long(),
            ).squeeze(-1)
            agreement = (p_cf_y - p_orig_y).mean().item()
    else:
        # No cf-pairs in this batch — pure CE training step.
        loss = ce_weight * ce_orig
        with torch.no_grad():
            ce_mean = ce_orig.item()
            cons_kl = 0.0
            agreement = 0.0

    return loss, ce_mean, cons_kl, agreement


def _consistency_classifier_loss(
    logits_original: Any,
    logits_counterfactual: Any,
    *,
    target_label: int = 1,
    ce_weight: float = 1.0,
    consistency_weight: float = 0.5,
) -> tuple[Any, float, float, float]:
    """Counterfactual consistency loss — R-Drop / UDA style.

    For each pair ``(x, x')`` where ``x`` is the original claim+evidence
    input and ``x'`` is the counterfactual+evidence input, the loss is::

        L = CE(x, y) + CE(x', y)
          + λ_cons · (1/2) · [KL(p(·|x)  ‖ stop_grad p(·|x'))
                              + KL(p(·|x') ‖ stop_grad p(·|x))]

    where ``y`` is the ground-truth binary label (target_label, always
    1 = contradicted for ClaimGuard), ``p(·|·)`` is the softmax over
    the 2 classes, and ``stop_grad`` on the KL targets prevents
    double-counting gradients (UDA Eq. 1; R-Drop uses bidirectional
    with targets detached).

    The CE terms preserve label supervision on BOTH sides so the
    model doesn't collapse to a uniform distribution just to minimize
    the consistency KL.  The consistency term pulls the two outputs
    toward each other without biasing either's absolute direction —
    exactly the invariance we want for causal-token preservation.

    Hyperparameter choice: R-Drop on GLUE used ``α ∈ {0.1, 0.5, 1.0}``
    tuned per task.  UDA used ``λ=1.0`` across text tasks.  We default
    to ``0.5`` as a safe midpoint for RoBERTa-large binary classification.

    Args:
        logits_original: 2-class logits on the original (claim + evidence)
            inputs.  Shape ``(B, 2)``.
        logits_counterfactual: 2-class logits on the counterfactual +
            evidence inputs.  Same shape.
        target_label: Which class is "contradicted" in the checkpoint.
            v1/v3 binary uses 1.
        ce_weight: Multiplier on the (CE_orig + CE_cf) term.  Default 1.0.
        consistency_weight: Multiplier on the symmetric-KL term.  Default 0.5.

    Returns:
        ``(loss_tensor, ce_mean_float, consistency_kl_float, agreement_mean_float)``

        * ``loss_tensor`` — scalar torch Tensor ready for backward().
        * ``ce_mean_float`` — mean cross-entropy across both sides.
          Fed into the early-stop "CE blowup" check.
        * ``consistency_kl_float`` — mean symmetric KL across the
          batch.  Logged per step so we can see convergence.
        * ``agreement_mean_float`` — mean ``P(y=target|x') - P(y=target|x)``,
          which should trend toward 0 as the model becomes consistent.
          Logged for monitoring.
    """
    import torch
    import torch.nn.functional as F

    # Shared true-label vector.  Both sides share the same label
    # because the counterfactual is a semantics-preserving paraphrase.
    target = torch.full(
        (logits_original.shape[0],),
        fill_value=int(target_label),
        dtype=torch.long,
        device=logits_original.device,
    )

    ce_orig = F.cross_entropy(logits_original, target, reduction="mean")
    ce_cf = F.cross_entropy(logits_counterfactual, target, reduction="mean")

    # Symmetric KL with stop-grad on targets (UDA Eq. 1 / R-Drop).
    log_p_orig = F.log_softmax(logits_original, dim=-1)
    log_p_cf = F.log_softmax(logits_counterfactual, dim=-1)
    p_orig = log_p_orig.exp()
    p_cf = log_p_cf.exp()

    kl_orig_to_cf = F.kl_div(
        log_p_orig, p_cf.detach(), reduction="batchmean",
    )
    kl_cf_to_orig = F.kl_div(
        log_p_cf, p_orig.detach(), reduction="batchmean",
    )
    consistency_kl = 0.5 * (kl_orig_to_cf + kl_cf_to_orig)

    loss = (
        ce_weight * (ce_orig + ce_cf)
        + consistency_weight * consistency_kl
    )

    with torch.no_grad():
        ce_mean = 0.5 * (ce_orig + ce_cf).item()
        cons_kl = consistency_kl.item()
        # Agreement signal: P(target | cf) - P(target | orig).  This
        # should trend toward zero as the model becomes consistent
        # between the two inputs.  Signed so we can see which side
        # is higher on average.
        p_orig_y = p_orig[:, int(target_label)]
        p_cf_y = p_cf[:, int(target_label)]
        agreement = (p_cf_y - p_orig_y).mean().item()

    return loss, ce_mean, cons_kl, agreement


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

    D21 fix (2026-04-15): the policy and reference models are now
    full ``VerifierModel`` instances loaded via
    ``inference.verifier_model.load_verifier_checkpoint``.  The old
    code tried to load the v3 checkpoint into a plain
    ``AutoModel + Linear(hidden, 2)`` layout via ``strict=False``,
    which silently dropped every head key (verdict_head.0.weight,
    score_head.*, contrastive_proj.*, heatmap_encoder.*) and started
    refinement training from a random-init Linear head.  The fix
    loads the full architecture and would-have-been-trained model
    so the refinement actually starts from the v3 weights it
    purports to fine-tune.  See decisions.md D21 + D25.
    """
    import random

    import torch

    # D21 fix: import the canonical loader.  This module will fail
    # to import if the Modal image is missing
    # `add_local_python_source("inference")` (see image build at
    # top of file), surfacing D20 + D21 issues at container start
    # rather than after expensive setup.
    from inference.verifier_model import load_verifier_checkpoint

    config = DPOTrainingConfig.from_json(config_json)

    # Reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Refinement training on device=%s", device)

    # Load training data.  Branch on loss_mode:
    #   "consistency_mixed" — load mixed cf + faithful via
    #     load_mixed_training_data (DEFAULT for v4 v2)
    #   "consistency" / "dpo" — load contradicted-only cf pairs via
    #     load_preference_pairs (LEGACY single-class path)
    _loss_mode_lower = (config.loss_mode or "consistency_mixed").lower()
    if _loss_mode_lower == "consistency_mixed":
        pairs = load_mixed_training_data(
            counterfactual_pairs_path=config.preference_data,
            full_training_data_path=config.full_training_data,
            faithful_per_cf=config.faithful_per_cf,
            seed=config.seed,
        )
        logger.info("Loaded %d mixed training rows", len(pairs))
    else:
        pairs = load_preference_pairs(config.preference_data)
        logger.info("Loaded %d preference pairs", len(pairs))
    if not pairs:
        raise RuntimeError(
            f"No training rows found in {config.preference_data}"
        )

    # ---- Policy (trainable) — full VerifierModel via canonical loader ----
    # load_verifier_checkpoint raises RuntimeError if any non-allowed
    # key is missing, so a stale or wrong-architecture checkpoint
    # fails loudly here instead of silently producing garbage gradients.
    tokenizer, policy_model = load_verifier_checkpoint(
        checkpoint_path=config.base_checkpoint,
        hf_backbone=config.hf_backbone,
        device=device,
        num_classes=2,
    )
    policy_model.train()

    # Freeze the first N text_encoder layers (RoBERTa-style).  The
    # heatmap_encoder, verdict_head, score_head, and contrastive_proj
    # all stay trainable.  The heatmap path receives zero input at
    # training time (we use text-only forward), so its parameters get
    # zero gradients in practice but staying trainable is harmless.
    frozen = _freeze_first_n_layers(
        policy_model.text_encoder, config.freeze_first_n_layers
    )
    logger.info(
        "Froze %d policy parameters in first %d text_encoder layers",
        frozen, config.freeze_first_n_layers,
    )

    # ---- Reference (frozen) — second VerifierModel for DPO branch ----
    # In R-Drop / consistency mode the reference model is unused (we
    # only need the policy and a stop-gradient on its own outputs),
    # but we still load it so the legacy DPO branch keeps working.
    # In consistency mode the ref load is wasted ~1.5 GB GPU memory;
    # we accept that to keep the code simple.
    _ref_tokenizer, ref_model = load_verifier_checkpoint(
        checkpoint_path=config.base_checkpoint,
        hf_backbone=config.hf_backbone,
        device=device,
        num_classes=2,
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # ---- Optimizer ----
    # Only train parameters with requires_grad=True (i.e. text_encoder
    # layers 8-23 + heatmap_encoder + verdict_head + score_head +
    # contrastive_proj).  Embeddings and layers 0-7 are frozen by
    # _freeze_first_n_layers above.
    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    tracker = DPOEarlyStopTracker(
        kl_max=config.kl_max,
        margin_min=config.margin_min,
        patience=config.patience,
    )
    reward_buffer: collections.deque[float] = collections.deque(
        maxlen=config.log_every
    )

    def _forward(model: Any, batch: list[dict[str, str]]) -> tuple[Any, Any]:
        """Run the full VerifierModel on (counterfactual, evidence) and
        (claim, evidence) pairs.

        Returns ``(chosen_logits, rejected_logits)`` where each is the
        2-class verdict_logits tensor of shape (batch, 2).  The
        sigmoid_score head output is discarded — refinement only
        touches the verdict head's signal.

        D21 fix: takes a single ``model`` (full VerifierModel) instead
        of the old (encoder, head) tuple.  The model's forward returns
        ``(verdict_logits, sigmoid_score)``; we keep verdict_logits
        and ignore sigmoid_score.
        """
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

        chosen_logits, _chosen_score = model(
            input_ids=chosen_enc["input_ids"],
            attention_mask=chosen_enc["attention_mask"],
        )
        rejected_logits, _rejected_score = model(
            input_ids=rejected_enc["input_ids"],
            attention_mask=rejected_enc["attention_mask"],
        )
        return chosen_logits, rejected_logits

    def _forward_mixed(
        model: Any,
        batch: list[dict[str, Any]],
    ) -> tuple[Any, Any, Any, Any]:
        """Mixed forward: handles cf-pair AND faithful rows in one batch.

        Returns ``(orig_logits, cf_logits, labels_tensor, cf_mask)``:
          * orig_logits: ``(B, 2)`` for ALL rows
          * cf_logits: ``(N_cf, 2)`` for the cf-mask subset only,
            or None if cf_mask is all-False
          * labels_tensor: ``(B,)`` int64 with the actual per-row label
          * cf_mask: ``(B,)`` bool tensor
        """
        # 1. ALL rows go through the original branch
        orig_claims = [b["claim"] for b in batch]
        orig_evidences = [b["evidence"] for b in batch]
        orig_enc = tokenizer(
            orig_claims, orig_evidences,
            padding=True, truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)
        orig_logits, _ = model(
            input_ids=orig_enc["input_ids"],
            attention_mask=orig_enc["attention_mask"],
        )

        # 2. Per-example labels
        labels_tensor = torch.tensor(
            [int(b.get("label", 0)) for b in batch],
            dtype=torch.long, device=device,
        )

        # 3. cf_mask: which rows have a non-None counterfactual
        cf_mask = torch.tensor(
            [b.get("counterfactual") is not None for b in batch],
            dtype=torch.bool, device=device,
        )

        # 4. cf branch — only on the cf_mask=True rows
        cf_logits = None
        if cf_mask.any():
            cf_indices = [i for i, b in enumerate(batch)
                          if b.get("counterfactual") is not None]
            cf_claims = [batch[i]["counterfactual"] for i in cf_indices]
            cf_evidences = [batch[i]["evidence"] for i in cf_indices]
            cf_enc = tokenizer(
                cf_claims, cf_evidences,
                padding=True, truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(device)
            cf_logits, _ = model(
                input_ids=cf_enc["input_ids"],
                attention_mask=cf_enc["attention_mask"],
            )

        return orig_logits, cf_logits, labels_tensor, cf_mask

    rng = random.Random(config.seed)
    global_step = 0
    stop_reason: Optional[str] = None
    history: list[dict[str, Any]] = []

    loss_mode = (config.loss_mode or "consistency_mixed").lower()
    if loss_mode not in ("consistency_mixed", "consistency", "dpo"):
        raise ValueError(
            f"unknown loss_mode={loss_mode!r}, expected "
            "'consistency_mixed', 'consistency', or 'dpo'"
        )
    logger.info("Training loss_mode=%s", loss_mode)

    for epoch in range(config.num_epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            if not batch:
                continue

            if loss_mode == "consistency_mixed":
                # NEW MIXED PATH (default for v4 v2).  Handles per-
                # example labels and faithful (no-cf) rows in the
                # same batch.  Required because v4 v1 collapsed:
                # training data was contradicted-only and the loss
                # hardcoded target_label=1, so R-Drop converged
                # trivially to "always predict contra".  Mixed
                # training preserves the v3 ability to predict
                # label=0 by including faithful examples with their
                # actual labels.
                orig_logits, cf_logits, labels_t, cf_mask_t = (
                    _forward_mixed(policy_model, batch)
                )
                loss, ce_mean, cons_kl, agreement = (
                    _consistency_mixed_loss(
                        orig_logits=orig_logits,
                        cf_logits=cf_logits,
                        labels=labels_t,
                        cf_mask=cf_mask_t,
                        ce_weight=config.ce_weight,
                        consistency_weight=config.consistency_weight,
                    )
                )
                margin = ce_mean
                kl_proxy = cons_kl
                chosen_reward = agreement

            elif loss_mode == "consistency":
                # R-Drop / UDA consistency path (DEFAULT, CORRECT).
                # We only need the policy model — no reference — but
                # we compute BOTH sides of the pair in the same
                # forward call to get matched gradients.  The
                # reference model is still loaded for legacy/
                # comparison but is unused in this branch.
                policy_orig_logits, policy_cf_logits = _forward(
                    policy_model, batch
                )
                # Note: _forward returns (chosen, rejected) =
                # (counterfactual, original).  For the consistency
                # loss we want (original, counterfactual), so we
                # pass them in that order.
                loss, ce_mean, cons_kl, agreement = (
                    _consistency_classifier_loss(
                        logits_original=policy_cf_logits,  # "rejected" in _forward = original claim
                        logits_counterfactual=policy_orig_logits,  # "chosen" in _forward = counterfactual
                        target_label=config.target_label,
                        ce_weight=config.ce_weight,
                        consistency_weight=config.consistency_weight,
                    )
                )
                # Monitor signal for logging + early-stop.  In
                # consistency mode the abort condition is "CE blows
                # up past ce_blowup_threshold" — if either CE climbs
                # above 2.5 the model is losing its label signal.
                margin = ce_mean  # repurposed for the tracker
                kl_proxy = cons_kl  # repurposed for the tracker
                chosen_reward = agreement  # repurposed for logging

            else:
                # Legacy DPO path (BROKEN — kept for research
                # comparison only; see top-of-file docstring).
                policy_chosen_logits, policy_rejected_logits = _forward(
                    policy_model, batch
                )
                with torch.no_grad():
                    ref_chosen_logits, ref_rejected_logits = _forward(
                        ref_model, batch
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
                "loss_mode": loss_mode,
                "loss": float(loss.item()),
                "margin": float(margin),
                "kl_proxy": float(kl_proxy),
                "chosen_reward": float(chosen_reward),
            })

            # Early-stop logic.  In consistency mode we only abort
            # on catastrophic CE blowup (the R-Drop loss is well-
            # behaved and doesn't have DPO's margin-explosion risk).
            # In DPO mode we keep the original KL/margin tracker.
            if loss_mode == "consistency":
                if margin > config.ce_blowup_threshold:
                    stop_reason = (
                        f"consistency-mode CE blowup: mean CE "
                        f"{margin:.3f} > {config.ce_blowup_threshold}"
                    )
                    logger.warning(
                        "Consistency early-stop triggered at step %d: %s",
                        global_step, stop_reason,
                    )
                    break
            else:
                if tracker.update(kl_proxy, margin):
                    stop_reason = tracker.reason
                    logger.warning(
                        "DPO early-stop triggered at step %d: %s",
                        global_step, stop_reason,
                    )
                    break

            if global_step % config.log_every == 0:
                if loss_mode == "consistency":
                    logger.info(
                        "step=%d loss=%.4f ce_mean=%.4f cons_kl=%.4f "
                        "agreement=%+.4f",
                        global_step, loss.item(), margin, kl_proxy,
                        chosen_reward,
                    )
                else:
                    logger.info(
                        "step=%d loss=%.4f margin=%.4f kl=%.4f "
                        "chosen_reward=%.4f | kl_streak=%d margin_streak=%d",
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
    # D21 fix: write a flat VerifierModel state_dict under the
    # "model_state_dict" key so downstream eval (modal_run_evaluation,
    # demo_provenance_gate_failure, run_openi_recalibrated_eval) can
    # load it via the canonical inference.verifier_model.load_verifier_checkpoint
    # path without any further translation.  The old format
    # {"encoder": ..., "head": ...} was load-compatible with the
    # broken AutoModel + Linear loader pattern, which is exactly what
    # we removed in D21.  Writing in the new format closes the loop:
    # if anything tries to load this v4 checkpoint with the broken
    # pattern, the canonical loader will raise on hard-missing keys.
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "epoch": config.num_epochs,
            "loss": float(history[-1]["loss"]) if history else None,
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
        # D20 fix (2026-04-15): ship the in-repo `inference/` package
        # so the trainer can import `inference.verifier_model.load_verifier_checkpoint`
        # at module load.  Without this, the container crashes at
        # import with `ModuleNotFoundError: No module named 'inference'`
        # exactly like the Task 9 gate demo did before its fix.  Same
        # pattern as scripts/demo_provenance_gate_failure.py +
        # scripts/modal_build_retrieval_eval.py.
        .add_local_python_source("inference")
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
    # Consistency-mode (default) hyperparameters
    parser.add_argument(
        "--loss-mode",
        choices=("consistency_mixed", "consistency", "dpo"),
        default=defaults.loss_mode,
        help=(
            "'consistency_mixed' (default, post-2026-04-15 v4 v1 fix): "
            "R-Drop with per-example labels and mixed cf+faithful data. "
            "'consistency' is the broken single-class path that "
            "collapsed v4 v1 to predict-contra-everywhere — kept ONLY "
            "for ablation. 'dpo' is the legacy Rafailov 2023 DPO loss "
            "with the chosen/rejected inversion bug — kept for "
            "research comparison only."
        ),
    )
    parser.add_argument(
        "--full-training-data",
        default=defaults.full_training_data,
        help=(
            "consistency_mixed mode only: path to the full v3 training "
            "data file (sampled for label=0 faithful examples)."
        ),
    )
    parser.add_argument(
        "--faithful-per-cf",
        type=float, default=defaults.faithful_per_cf,
        help=(
            "consistency_mixed mode only: ratio of faithful examples "
            "to cf-pair examples.  1.0 = balanced (default).  "
            "Higher = more conservative, less collapse risk."
        ),
    )
    parser.add_argument(
        "--ce-weight", type=float, default=defaults.ce_weight,
    )
    parser.add_argument(
        "--consistency-weight",
        type=float, default=defaults.consistency_weight,
    )
    parser.add_argument(
        "--ce-blowup-threshold",
        type=float, default=defaults.ce_blowup_threshold,
    )
    args = parser.parse_args(argv)

    return DPOTrainingConfig(
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        preference_data=args.preference_data,
        full_training_data=args.full_training_data,
        faithful_per_cf=args.faithful_per_cf,
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
        loss_mode=args.loss_mode,
        ce_weight=args.ce_weight,
        consistency_weight=args.consistency_weight,
        ce_blowup_threshold=args.ce_blowup_threshold,
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
    # Loss functions — underscore-prefixed because they need torch
    # (lazy-imported inside the function body), so they're not part
    # of the pure-Python API surface.  Exposed here so unit tests
    # that DO have torch can import them directly.
    "_consistency_classifier_loss",
    "_dpo_classifier_loss",
]
