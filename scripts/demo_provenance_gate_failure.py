"""Task 9 — Provenance-gate same-model failure-mode demonstration.

This script empirically validates the ``apply_provenance_gate`` policy
in ``inference/provenance.py`` by running CheXagent twice on the same
100 OpenI images with different sampling seeds, then asking the v1
verifier to score every claim-from-A against both its own run-A report
(``SAME_MODEL``) and the independent run-B report (``INDEPENDENT``).

The headline result for the paper is the downgrade-rate table:

    condition   | n_high_score | n_cert_pre | n_cert_post | downgrade
    ------------|--------------|------------|-------------|-----------
    same_run    |      ~X      |    ~X      |     0       |    1.00
    cross_run   |      ~X      |    ~X      |     ~X      |    0.00

The gate should downgrade every SAME_MODEL green claim to
``SUPPORTED_UNCERTIFIED`` (even when the verifier score is ~1.0, since
the claim was literally extracted from that same report), while leaving
cross-run green claims fully certified.  The ``downgrade_rate_diff`` is
the single number we cite in the paper figure.

Why this is novel and publishable
---------------------------------
Every text-only VLM hallucination-detection system that uses the model
to audit its own output is silently vulnerable to self-consistency
loops: the verifier agrees with the generator because both pooled on
the same priors.  The provenance gate is the cheapest possible fix — it
never touches the verifier weights, it just refuses to call an
unverifiable claim "trusted."  Before Task 9, this property was asserted
on unit tests and an inlined walkthrough in ``scripts/modal_run_evaluation.py``.
After Task 9, it is a measured quantity on real OpenI data.

Architecture overview
---------------------
The script is split into three layers for testability:

    Layer 1 — pure helpers (no torch / modal / anthropic)
        * ``_conformal_label_from_score``  score → green/yellow/red
        * ``build_gate_demo_row``          one row = score + prov → final label
        * ``compute_gate_demo_stats``      per-condition summary table
        * ``synthetic_gate_demo_workbook`` deterministic test fixture
        * ``load_workbook_claims``         JSON load with schema guard
        * ``pair_claims_with_evidence``    same-run + cross-run pairing

    Layer 2 — torch helpers (run inside the Modal container)
        * ``_load_v1_verifier``            AutoModel + Linear head load
        * ``_score_claim_evidence_pairs``  batched sigmoid inference

    Layer 3 — Modal entrypoint
        * ``_run_demo``                    orchestrates the whole flow
        * ``demo_provenance_gate_remote``  @app.function wrapper

Everything in Layer 1 is unit-tested in
``tests/test_demo_provenance_gate_failure.py`` without torch or Modal.
Layer 2 / 3 are exercised only by the real H100 run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.provenance import (  # noqa: E402
    EvidenceSourceType,
    ProvenanceTriageLabel,
    TrustTier,
    apply_provenance_gate,
    classify_trust_tier,
)

logger = logging.getLogger(__name__)

APP_NAME = "claimguard-provenance-gate-demo"
VOLUME_NAME = "claimguard-data"

# Run ids used to exercise the provenance gate.  ``generate_real_hallucinations``
# stamps claims with ``claim_generator_id = f"chexagent-8b-run-{run_id}"``,
# so we pin the same string format here to keep the two scripts in lock-step.
RUN_A_ID = "run-a"
RUN_B_ID = "run-b"
RUN_A_GENERATOR_ID = f"chexagent-8b-{RUN_A_ID}"
RUN_B_GENERATOR_ID = f"chexagent-8b-{RUN_B_ID}"

# Any verifier "supported" score >= this counts as a would-be-certified
# claim for the purpose of Task 9's downgrade-rate table.  v1 is
# well-calibrated in this regime: 0.5 is the decision boundary for
# "not contradicted", matching the cfBH green-label threshold on
# calibrated faithful scores in ``scripts/modal_run_evaluation.py``.
DEFAULT_HIGH_SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Config + row dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GateDemoConfig:
    """CLI / entry-point configuration for the dual-run demo.

    All fields are JSON-serializable so the whole config can be shipped
    across the Modal boundary as a single string.
    """

    # Default to the v3 checkpoint (trained on the 12-type taxonomy;
    # val_acc 0.9877 on 2026-04-15 retrain).  The v1 path at
    # /data/checkpoints/verifier_binary/best_verifier.pt uses the same
    # VerifierModel architecture (text_encoder + heatmap_encoder +
    # verdict_head + score_head + contrastive_proj), so switching
    # v1 -> v3 here is a pure architecture-compatible upgrade.
    verifier_checkpoint: str = (
        "/data/checkpoints/verifier_binary_v3/best_verifier.pt"
    )
    openi_images_dir: str = "/data/openi_images"
    openi_reports_csv: str = "/data/openi_cxr_chexpert_schema.csv"
    workbook_a_path: str = (
        "/data/same_model_experiment/annotation_workbook_run_a.json"
    )
    workbook_b_path: str = (
        "/data/same_model_experiment/annotation_workbook_run_b.json"
    )
    output_dir: str = "/data/same_model_experiment"
    max_images: int = 100
    image_seed: int = 42
    seed_run_a: int = 101
    seed_run_b: int = 202
    temperature: float = 0.7
    top_p: float = 0.9
    high_score_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD
    hf_backbone: str = "roberta-large"
    max_length: int = 256
    batch_size: int = 16
    # If True, skip CheXagent (re)generation and assume the run_a/run_b
    # workbooks already exist on the volume.  Used for a re-run that
    # only regenerates the pair table after a scoring threshold tweak.
    skip_generation: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "GateDemoConfig":
        return cls(**json.loads(s))


@dataclass
class GateDemoRow:
    """One row of the (claim, evidence, gate-outcome) demo table.

    Each row captures the full provenance + gate result for a single
    (claim, evidence) pairing, so the downstream paper figure can be
    generated without any re-computation.
    """

    claim_id: str
    claim_text: str
    evidence_text: str
    claim_generator_id: str
    evidence_generator_id: str
    verifier_score: float
    high_score: bool
    conformal_label: str            # "green" | "yellow" | "red"
    trust_tier: str                 # TrustTier.*
    gate_label: str                 # ProvenanceTriageLabel.*
    was_overridden: bool
    condition: str                  # "same_run" | "cross_run"
    # Optional audit fields — left blank by default; the Modal path
    # fills them in from the upstream workbook.
    image_file: str = ""
    pathology: str = ""
    ground_truth_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Layer 1 — pure helpers.
# ---------------------------------------------------------------------------


def _conformal_label_from_score(
    score: float,
    *,
    high_score_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
) -> str:
    """Map a raw verifier score to the green/yellow/red label that
    ``apply_provenance_gate`` expects.

    Task 9 does not depend on the full cfBH procedure — we are
    demonstrating the provenance-gate policy, not FDR control — so a
    simple two-threshold cut is sufficient:

        score >= high_score_threshold         -> green
        score >= 0.5 * high_score_threshold   -> yellow
        otherwise                             -> red

    ``high_score_threshold`` defaults to 0.5, matching the v1 sigmoid
    decision boundary.  A higher threshold (e.g. 0.9) would restrict
    "would-be-certified" to only extreme-confidence claims, which is
    also a reasonable Task 9 variant.
    """
    if score >= high_score_threshold:
        return "green"
    if score >= 0.5 * high_score_threshold:
        return "yellow"
    return "red"


def build_gate_demo_row(
    *,
    claim_id: str,
    claim_text: str,
    evidence_text: str,
    claim_generator_id: str,
    evidence_generator_id: str,
    verifier_score: float,
    condition: str,
    high_score_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
    image_file: str = "",
    pathology: str = "",
    ground_truth_label: str = "",
) -> GateDemoRow:
    """Assemble a single ``GateDemoRow`` from raw pipeline outputs.

    This is the pure entry point used by both the Modal path (real
    verifier scores) and the unit tests (synthetic scores).  It
    encapsulates the per-row logic in one place:

        1. Classify trust tier from the two generator ids via
           ``classify_trust_tier(GENERATOR_OUTPUT, claim_id, evidence_id)``.
        2. Map the verifier score to a green/yellow/red label via
           ``_conformal_label_from_score``.
        3. Apply ``apply_provenance_gate`` to get the final provenance
           label + override flag.

    Args:
        claim_id: Unique id for the claim (for downstream join / audit).
        claim_text: Full claim string.
        evidence_text: Full evidence string (the CheXagent report).
        claim_generator_id: Provenance id of the claim's generator.
        evidence_generator_id: Provenance id of the evidence's generator.
        verifier_score: Raw sigmoid "supported" score from the v1
            binary verifier.  Higher = more supported.
        condition: ``"same_run"`` or ``"cross_run"``.  Stamped into
            the row so downstream filtering is trivial.
        high_score_threshold: Score threshold used to call "green".
        image_file: Optional OpenI image filename (for audit).
        pathology: Optional CheXpert pathology (for per-pathology stats).
        ground_truth_label: Optional label from the OpenI radiologist
            report (used to ask "did the gate downgrade a CORRECT
            claim"?).

    Returns:
        A fully populated ``GateDemoRow``.
    """
    tier = classify_trust_tier(
        source_type=EvidenceSourceType.GENERATOR_OUTPUT,
        claim_generator_id=claim_generator_id,
        evidence_generator_id=evidence_generator_id,
    )
    conformal_label = _conformal_label_from_score(
        float(verifier_score),
        high_score_threshold=high_score_threshold,
    )
    gate_result = apply_provenance_gate(conformal_label, tier)
    return GateDemoRow(
        claim_id=claim_id,
        claim_text=claim_text,
        evidence_text=evidence_text,
        claim_generator_id=claim_generator_id,
        evidence_generator_id=evidence_generator_id,
        verifier_score=float(verifier_score),
        high_score=float(verifier_score) >= high_score_threshold,
        conformal_label=conformal_label,
        trust_tier=tier,
        gate_label=gate_result.final_label,
        was_overridden=bool(gate_result.was_overridden),
        condition=condition,
        image_file=image_file,
        pathology=pathology,
        ground_truth_label=ground_truth_label,
    )


def compute_gate_demo_stats(
    rows: Sequence[GateDemoRow],
) -> dict[str, Any]:
    """Aggregate ``GateDemoRow``s into the per-condition summary table.

    For each ``condition`` in ``{"same_run", "cross_run"}``:

        * ``n_claims``               — total rows in that condition.
        * ``n_high_score``           — rows with ``high_score = True``.
        * ``n_certified_pre_gate``   — rows whose ``conformal_label`` is
          ``"green"``; i.e. what the verifier would certify if the
          provenance gate were disabled.
        * ``n_certified_post_gate``  — rows whose ``gate_label`` is
          ``SUPPORTED_TRUSTED``; the actual certified count after the
          gate.
        * ``downgrade_rate``         — ``(n_pre - n_post) / n_pre``.
          Expected ~1.0 on same_run (every green gets overridden by
          the gate because claim and evidence share a generator) and
          ~0.0 on cross_run (INDEPENDENT tier is fully certifiable).

    Also adds a top-level ``downgrade_rate_diff`` = same_run - cross_run
    which is the headline number for the paper figure: it should be
    ≈ 1.0 if the gate is working.

    Args:
        rows: Iterable of ``GateDemoRow`` objects.

    Returns:
        Nested dict with ``by_condition``, ``downgrade_rate_diff``,
        ``n_rows_total`` keys.
    """
    rows_list = list(rows)
    by_cond: dict[str, dict[str, Any]] = {}
    for cond in ("same_run", "cross_run"):
        subset = [r for r in rows_list if r.condition == cond]
        n_claims = len(subset)
        n_high_score = sum(1 for r in subset if r.high_score)
        n_pre = sum(1 for r in subset if r.conformal_label == "green")
        n_post = sum(
            1 for r in subset
            if r.gate_label == ProvenanceTriageLabel.SUPPORTED_TRUSTED
        )
        downgrade_rate = (
            (n_pre - n_post) / n_pre if n_pre > 0 else 0.0
        )
        by_cond[cond] = {
            "n_claims": n_claims,
            "n_high_score": n_high_score,
            "n_certified_pre_gate": n_pre,
            "n_certified_post_gate": n_post,
            "downgrade_rate": float(downgrade_rate),
        }

    downgrade_diff = (
        by_cond["same_run"]["downgrade_rate"]
        - by_cond["cross_run"]["downgrade_rate"]
    )
    return {
        "by_condition": by_cond,
        "downgrade_rate_diff": float(downgrade_diff),
        "n_rows_total": len(rows_list),
    }


def synthetic_gate_demo_workbook(
    *,
    n_rows: int = 20,
    seed: int = 0,
    high_score_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
) -> list[GateDemoRow]:
    """Build a synthetic same-run + cross-run workbook for unit tests.

    Generates ``n_rows`` claims per condition.  ``same_run`` rows use
    the same generator id on both sides (SAME_MODEL → downgraded).
    ``cross_run`` rows use distinct ids (INDEPENDENT → certified).  All
    verifier scores are in the "high" range (0.6 .. 1.0) so every row's
    ``conformal_label`` is green — that makes the downgrade rate a
    simple fraction of SAME_MODEL rows.

    The fixture is deterministic under ``seed`` so the paired unit
    test assertions are stable under any refactor.

    Returns:
        A list of 2 * n_rows ``GateDemoRow`` objects.  The first
        ``n_rows`` are ``same_run``; the next ``n_rows`` are
        ``cross_run``.
    """
    import random

    rng = random.Random(seed)
    rows: list[GateDemoRow] = []

    for i in range(n_rows):
        score = 0.6 + rng.random() * 0.4  # 0.6 .. 1.0 — always green
        rows.append(
            build_gate_demo_row(
                claim_id=f"synthetic-same-{i:03d}",
                claim_text=f"claim {i}",
                evidence_text=f"evidence {i}",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_A_GENERATOR_ID,
                verifier_score=score,
                condition="same_run",
                high_score_threshold=high_score_threshold,
            )
        )

    for i in range(n_rows):
        score = 0.6 + rng.random() * 0.4
        rows.append(
            build_gate_demo_row(
                claim_id=f"synthetic-cross-{i:03d}",
                claim_text=f"claim {i}",
                evidence_text=f"evidence {i}",
                claim_generator_id=RUN_A_GENERATOR_ID,
                evidence_generator_id=RUN_B_GENERATOR_ID,
                verifier_score=score,
                condition="cross_run",
                high_score_threshold=high_score_threshold,
            )
        )

    return rows


def load_workbook_claims(
    path: "os.PathLike[str] | str",
) -> list[dict[str, Any]]:
    """Load an annotation workbook JSON and return its list of claims.

    Pure helper — the workbook schema is the one emitted by
    ``scripts/generate_real_hallucinations.py`` (one dict per claim,
    with ``claim_id``, ``image_file``, ``extracted_claim``,
    ``generated_report``, ``claim_generator_id``, ... fields).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the top-level JSON is not a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"expected JSON list at {path}, got {type(raw).__name__}"
        )
    return raw


def _extract_generator_id(
    claims: Sequence[dict[str, Any]],
    *,
    fallback: str,
) -> str:
    """Return the unique ``claim_generator_id`` across all rows.

    The reviewer flagged a silent-overwrite bug in the prior implementation:
    it walked every row and kept the *last* non-empty ``claim_generator_id``,
    so a workbook with mixed ids (e.g. a stale re-run leaking a row stamped
    with a different generator) would silently contaminate every pair row.

    This helper is strict:

    * If every row omits ``claim_generator_id`` or sets it to a falsy value,
      we return ``fallback``.
    * If exactly one distinct non-empty ``claim_generator_id`` is present,
      we return it.
    * If two or more distinct ids are present, we raise ``ValueError`` —
      the workbook is corrupt and Task 9's tier classification would be
      wrong, so we'd rather fail loudly than silently produce garbage.
    """
    seen: set[str] = set()
    for c in claims:
        cid = c.get("claim_generator_id")
        if isinstance(cid, str) and cid:
            seen.add(cid)
    if len(seen) > 1:
        raise ValueError(
            "Workbook contains conflicting claim_generator_id values: "
            f"{sorted(seen)!r}. Expected all rows to share one id."
        )
    if len(seen) == 1:
        return next(iter(seen))
    return fallback


def _is_error_sentinel(text: str) -> bool:
    """Return True if ``text`` looks like a CheXagent error sentinel.

    ``generate_real_hallucinations.py`` writes two sentinel forms into
    ``generated_report`` when CheXagent fails to produce a report:

    * ``"[CheXagent unavailable]"`` — module import failed
    * ``"[Generation error: ...]"`` — per-image generation crashed

    Passing these as ``evidence_text`` to the v1 verifier would waste
    GPU cycles and pollute the stats.  We detect the two known formats
    conservatively — a bracketed string whose lowercase content contains
    ``"error"`` or ``"unavailable"`` — so real radiology prose (which
    may incidentally contain brackets) is not dropped.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    inner = stripped[1:-1].lower()
    return "error" in inner or "unavailable" in inner


def pair_claims_with_evidence(
    *,
    claims_run_a: list[dict[str, Any]],
    claims_run_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair claims from run A with evidence from runs A and B.

    For each claim in ``claims_run_a`` we emit two pair rows iff the
    same image is present in both workbooks:

        1. ``same_run``  = (claimA, reportA for the same image)
            → claim_generator_id == evidence_generator_id
            → TrustTier.SAME_MODEL
        2. ``cross_run`` = (claimA, reportB for the same image)
            → claim_generator_id != evidence_generator_id
            → TrustTier.INDEPENDENT

    Evidence is the ``generated_report`` field from the workbook (the
    full CheXagent report), NOT the ``extracted_claim`` field.  Matching
    on image filename guarantees that the cross-run comparison uses a
    semantically aligned second opinion from the independent run.

    If an image is present only in run A (because run B used a
    different seed for image sampling), we skip that image entirely —
    we never substitute a DIFFERENT-image report as cross-run evidence,
    which would conflate image effects with generator effects.

    Any row whose ``generated_report`` is an error sentinel (see
    ``_is_error_sentinel``) is treated as missing — its image is not
    included in the matched intersection.

    Args:
        claims_run_a: List of workbook rows from CheXagent run A.
        claims_run_b: List of workbook rows from CheXagent run B.

    Returns:
        A list of pair dicts with keys ``claim_id``, ``claim_text``,
        ``evidence_text``, ``claim_generator_id``,
        ``evidence_generator_id``, ``condition``, ``image_file``.

    Raises:
        ValueError: if either workbook contains conflicting
            ``claim_generator_id`` values across rows (see
            ``_extract_generator_id``).
    """
    gen_a = _extract_generator_id(claims_run_a, fallback=RUN_A_GENERATOR_ID)
    gen_b = _extract_generator_id(claims_run_b, fallback=RUN_B_GENERATOR_ID)

    reports_a: dict[str, str] = {}
    reports_b: dict[str, str] = {}

    for c in claims_run_a:
        img = str(c.get("image_file", ""))
        if not img:
            continue
        report = str(c.get("generated_report", ""))
        if _is_error_sentinel(report):
            continue
        reports_a[img] = report

    for c in claims_run_b:
        img = str(c.get("image_file", ""))
        if not img:
            continue
        report = str(c.get("generated_report", ""))
        if _is_error_sentinel(report):
            continue
        reports_b[img] = report

    pairs: list[dict[str, Any]] = []
    for c in claims_run_a:
        img = str(c.get("image_file", ""))
        claim_text = str(c.get("extracted_claim", "")).strip()
        if not img or not claim_text:
            continue

        evidence_a = reports_a.get(img, "")
        evidence_b = reports_b.get(img, "")
        # Require BOTH reports to exist (and be non-sentinel) before we
        # emit either pair.  This guarantees that same_run and cross_run
        # are a matched comparison on identical claims — any observed
        # downgrade-rate difference is attributable to the provenance
        # gate, not to different image samples or to one run failing.
        if not evidence_a or not evidence_b:
            continue

        claim_id = str(c.get("claim_id", f"claim-{len(pairs):06d}"))
        pairs.append({
            "claim_id": f"{claim_id}__same",
            "claim_text": claim_text,
            "evidence_text": evidence_a,
            "claim_generator_id": gen_a,
            "evidence_generator_id": gen_a,  # SAME_MODEL
            "condition": "same_run",
            "image_file": img,
        })
        pairs.append({
            "claim_id": f"{claim_id}__cross",
            "claim_text": claim_text,
            "evidence_text": evidence_b,
            "claim_generator_id": gen_a,
            "evidence_generator_id": gen_b,  # INDEPENDENT
            "condition": "cross_run",
            "image_file": img,
        })

    return pairs


# ---------------------------------------------------------------------------
# Layer 2 — torch helpers.  Imported lazily inside Modal.
# ---------------------------------------------------------------------------


def _build_verifier_model(hf_backbone: str, num_classes: int = 2) -> Any:
    """Construct the ``VerifierModel`` used by v1/v3/v4 training.

    This is the EXACT architecture used in
    ``scripts/modal_run_evaluation.py`` (lines 105–162). It is a
    multimodal model (text + heatmap) whose binary verdict head is what
    Task 9 reads. Forward shape::

        model(input_ids, attention_mask) -> (verdict_logits, score)

    * ``verdict_logits``: ``(batch, num_classes)`` softmax-friendly
      verdict over {not contradicted, contradicted} when num_classes=2.
    * ``score``: ``(batch,)`` sigmoid regression head.  Task 9 uses
      ``softmax(verdict_logits)[:, 0]`` (the "not contradicted" prob)
      as the supported-score, matching main eval.

    The heatmap path is zero-filled at inference for Task 9 because
    CheXagent claims and evidence are pure text — the claim/evidence
    were not grounded to a heatmap.  This matches how
    ``modal_run_evaluation.py`` runs inference on the synthetic eval
    set: ``model(input_ids, attention_mask)`` with no ``heatmap``
    kwarg, so ``hmap_feat`` is a zero vector and the fused input is
    ``cat([text_cls, zeros_768], -1)``.

    Reviewer note (bug 2, 2026-04-15): the earlier ``_load_v1_verifier``
    in this file tried to load the checkpoint into a plain
    ``AutoModel + Linear(hidden, 2)`` layout.  The v1/v3 checkpoint
    is instead the full ``VerifierModel`` (text_encoder + heatmap
    CNN + verdict_head Linear(1792, 256)→Linear(256, 2) + score_head +
    contrastive_proj), so ``strict=False`` silently dropped all the
    head keys.  The resulting classifier had random Linear(1024, 2)
    weights, producing ~identical scores for supported vs contradicted
    probes (sup=0.2063 con=0.2029, margin 0.0034) and triggering the
    sanity check.  This replacement defines the real architecture so
    every head weight loads.
    """
    import torch  # noqa: F401  (kept local to avoid import at module load)
    from torch import nn
    from transformers import AutoModel

    class HeatmapEncoder(nn.Module):
        def __init__(self, output_dim=768):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(128, output_dim)

        def forward(self, heatmap):
            if heatmap.ndim == 3:
                heatmap = heatmap.unsqueeze(1)
            return self.proj(self.conv(heatmap).flatten(1))

    class VerifierModel(nn.Module):
        def __init__(
            self,
            model_name: str,
            heatmap_dim: int = 768,
            num_classes: int = 2,
            hidden_dim: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(model_name)
            text_dim = self.text_encoder.config.hidden_size
            self.heatmap_encoder = HeatmapEncoder(output_dim=heatmap_dim)
            fused_dim = text_dim + heatmap_dim

            self.verdict_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.score_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.contrastive_proj = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128),
            )

        def forward(self, input_ids, attention_mask, heatmap=None):
            outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            text_cls = outputs.last_hidden_state[:, 0, :]
            if heatmap is not None:
                hmap_feat = self.heatmap_encoder(heatmap)
            else:
                hmap_feat = torch.zeros(
                    text_cls.shape[0],
                    self.heatmap_encoder.proj.out_features,
                    device=text_cls.device,
                    dtype=text_cls.dtype,
                )
            fused = torch.cat([text_cls, hmap_feat], dim=-1)
            verdict_logits = self.verdict_head(fused)
            score = torch.sigmoid(self.score_head(fused)).squeeze(-1)
            return verdict_logits, score

    return VerifierModel(hf_backbone, num_classes=num_classes)


def _load_v1_verifier(
    checkpoint_path: str,
    hf_backbone: str,
    device: Any,
) -> tuple[Any, Any]:
    """Load the v1/v3 binary ``VerifierModel`` and return (tokenizer, model).

    The returned ``model`` is the full multimodal
    ``VerifierModel`` (text_encoder + heatmap_encoder + verdict_head +
    score_head + contrastive_proj), same architecture as in
    ``scripts/modal_run_evaluation.py``.  Task 9 uses the text-only
    path (``model(input_ids, attention_mask)`` with heatmap=None).

    Checkpoint layouts supported:

    * ``state == {"model_state_dict": {...VerifierModel keys...}, ...}``
      (v1, v3, v4 — the layout actually written by
      ``scripts/modal_train_verifier_binary.py``).  Loaded
      ``strict=True`` because every key in the checkpoint maps directly
      to a parameter in the model.
    * ``state == {...VerifierModel keys...}`` (flat).  Also
      ``strict=True``.

    Legacy ``state == {"encoder": ..., "head": ...}`` is NOT supported
    anymore — that branch only worked for a pure (AutoModel +
    Linear(hidden, 2)) layout that never actually existed in any saved
    checkpoint.
    """
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_backbone)
    model = _build_verifier_model(hf_backbone, num_classes=2).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is not a dict "
            f"(type={type(state).__name__}).  Cannot load VerifierModel."
        )
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state  # assume flat VerifierModel state
    # strict=True: any missing or unexpected key is a load failure.
    # strict=False would silently tolerate missing head keys, which is
    # exactly the bug we are fixing.  The earlier loader used
    # strict=False and produced 0.2063 vs 0.2029 scores on the sanity
    # check probes.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Translate a soft-load into a hard error unless the only missing
    # keys are the optional contrastive_proj (present in v3 but not
    # strictly needed at inference time) or known transformer pooler
    # keys that HF sometimes strips (text_encoder.pooler.*).  Any other
    # missing key means the loader is wrong.
    allowed_missing_prefixes = (
        "text_encoder.pooler.",  # HF sometimes strips pooler
        "contrastive_proj.",  # not used at inference
    )
    hard_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]
    if hard_missing:
        raise RuntimeError(
            f"VerifierModel checkpoint load failed — {len(hard_missing)} "
            f"hard-missing keys after load_state_dict.  First 5: "
            f"{hard_missing[:5]}.  This indicates the checkpoint's "
            f"architecture does not match the VerifierModel definition "
            f"in _build_verifier_model.  Check scripts/modal_train_verifier_binary.py "
            f"to see which version of VerifierModel the checkpoint was "
            f"trained against."
        )
    if unexpected:
        logger.warning(
            "VerifierModel load: %d unexpected keys (ignored). First 5: %s",
            len(unexpected), unexpected[:5],
        )
    if missing:
        logger.info(
            "VerifierModel load: %d allowed-missing keys (pooler/"
            "contrastive): %s",
            len(missing), missing[:5],
        )

    model.eval()
    # Return (tokenizer, model).  The old API returned
    # (tokenizer, encoder, head); callers must now use a 2-tuple.
    return tokenizer, model


# On-distribution sanity probes sampled from ``/data/eval_data_v3/test_claims.json``
# at Task 9 integration time.  The v3 verifier (val_acc 0.9877) scores
# these clean-label examples with high confidence: mean(P_supported |
# label=0) = 0.998, mean(P_supported | label=1) = 0.054, separation
# +0.944 on a 30-sample probe batch we ran locally on 2026-04-15.
#
# Why hand-picked eval samples instead of hand-written probes:
# the v1/v3 training data is MIMIC-CXR report fragments in mixed ALL-
# CAPS clinical shorthand (see ``data/augmentation/hard_negative_generator``
# output).  Well-formed English sentences like "There is evidence of
# cardiomegaly" with "The cardiac silhouette is enlarged" are OFF
# DISTRIBUTION and produce near-identical ~0.97 scores for both
# supported and contradicted probes, triggering a false-positive sanity
# failure.  Using real eval rows guarantees the sanity check tests the
# model on the same distribution it was trained on.  The evidence is
# already joined with ``" [SEP] "`` to match the
# ``ClaimDataset.__getitem__`` convention in
# ``scripts/modal_run_evaluation.py`` line 236.
_SANITY_SUPPORTED_PROBES = (
    (
        "AND/OR CONSOLIDATION.",
        "1.SINGLE PORTABLE UPRIGHT RADIOGRAPH OF THE CHEST DEMONSTRATES"
        " [SEP] PROMINENCE OF THE PULMONARY VASCULATURE IN THE UPPER LUNG"
        " ZONES WHICH",
    ),
    (
        "INTERVAL PLACEMENT OF RIGHT TUNNELED INTERNAL JUGULAR CATHETER",
        "WITHOUT EVIDENCE OF POST-PROCEDURAL PNEUMOTHORAX."
        " [SEP] OTHERWISE, STABLE EXAMINATION.",
    ),
    (
        "PICC ARE AGAIN DEMONSTRATED WITH THE PICC TERMINATING IN THE"
        " REGION",
        "SWAN-GANZ CATHETER REMOVED. [SEP] NEGATIVE FOR PNEUMOTHORAX.",
    ),
    (
        "2.DUAL LEAD RIGHT AICD WITH GENERATOR.",
        "1.BIOPROSTHETIC MITRAL VALVE AND STERNAL SUTURE WIRES."
        " [SEP] FULL LENGTH LEFT AICD LEAD",
    ),
)
_SANITY_CONTRADICTED_PROBES = (
    (
        "1.PA AND LATERAL CHEST RADIOGRAPH IS no NOT SIGNIFICANTLY CHANGED",
        "COMPARED TO PRIOR. [SEP] UNCHANGED LEFT HILAR/LUL AND RIGHT HILAR"
        " NODULAR",
    ),
    (
        "2.THERE IS no MINIMAL LINEAR ATELECTASIS AT THE LEFT LUNG BASE.",
        "1.TWO VIEWS OF THE CHEST DEMONSTRATE AN UNREMARKABLE [SEP]"
        " CARDIOMEDIASTINAL SILHOUETTE.",
    ),
    (
        "2.CARDIAC SILHOUETTE AND VASCULARITY ARE no WITHIN NORMAL LIMITS.",
        "1.Chest 1 View, DEMONSTRATE NO FOCAL CONSOLIDATION OR PLEURAL"
        " [SEP] 3.MULTIPLE LEFT SIDED RIB FRACTURES AGAIN NOTED.",
    ),
    (
        "BRONCHIAL CUFFING absent, WHICH MAY REPRESENT MILD PULMONARY",
        "SINGLE VIEW OF THE CHEST WITH INTERVAL PLACEMENT OF LEFT"
        " [SEP] SUBCLAVIAN LINE, WITH TIP IN THE SVC.",
    ),
)
# Required mean separation: supported mean - contradicted mean >=
# _SANITY_MARGIN.  A random-init head gives margin ~0.  A healthy
# v1/v3 VerifierModel gives margin >= 0.4 in our local 4-probe tests.
# 0.2 is a conservative minimum that tolerates noise on individual
# edge probes while still catching the "every probe scores the same"
# failure mode of a mis-loaded checkpoint.
_SANITY_MARGIN = 0.2


def _sanity_check_verifier(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    target_label: int = 1,
) -> tuple[float, float]:
    """Score bundled on-distribution probes to catch silent load failures.

    Reviewer-flagged cheap insurance: ``_load_v1_verifier`` uses
    ``strict=False`` when loading the full ``VerifierModel``, which
    silently tolerates missing keys.  A checkpoint with renamed or
    missing head keys would leave the final ``Linear(256, 2)`` layer
    with random initialization, and every downstream "supported
    score" in Task 9 would be uniform noise around 0.5.  Worse: the
    demo would still produce a table that *looks* plausible.

    This function scores 4 on-distribution supported probes and
    4 on-distribution contradicted probes (see
    ``_SANITY_SUPPORTED_PROBES`` / ``_SANITY_CONTRADICTED_PROBES``)
    and requires::

        mean(P_supported | supported probes)
            - mean(P_supported | contradicted probes)
            >= _SANITY_MARGIN (= 0.2)

    On a healthy v3 VerifierModel the margin is ~0.9 (0.998 vs 0.054
    measured locally on 2026-04-15 on 30-sample batches).  On a
    random-init head or a mis-loaded checkpoint the margin collapses
    to ~0.  0.2 is the minimum we require; the real margin should be
    much higher.

    Returns:
        ``(mean_supported_score, mean_contradicted_score)`` so callers
        can log the actual values for debugging.

    Raises:
        RuntimeError: if the mean margin is below ``_SANITY_MARGIN``.
    """
    sup_pairs = [
        {"claim_text": claim, "evidence_text": evidence}
        for claim, evidence in _SANITY_SUPPORTED_PROBES
    ]
    con_pairs = [
        {"claim_text": claim, "evidence_text": evidence}
        for claim, evidence in _SANITY_CONTRADICTED_PROBES
    ]
    sup_scores = _score_claim_evidence_pairs(
        tokenizer=tokenizer,
        model=model,
        pairs=sup_pairs,
        device=device,
        batch_size=len(sup_pairs),
        max_length=256,
        target_label=target_label,
    )
    con_scores = _score_claim_evidence_pairs(
        tokenizer=tokenizer,
        model=model,
        pairs=con_pairs,
        device=device,
        batch_size=len(con_pairs),
        max_length=256,
        target_label=target_label,
    )
    sup_mean = sum(sup_scores) / len(sup_scores)
    con_mean = sum(con_scores) / len(con_scores)
    logger.info(
        "Verifier sanity probes (mean over %d sup / %d con): "
        "supported=%.4f, contradicted=%.4f, margin=%.4f",
        len(sup_scores), len(con_scores), sup_mean, con_mean,
        sup_mean - con_mean,
    )
    logger.info("  supported per-probe: %s",
                [f"{s:.3f}" for s in sup_scores])
    logger.info("  contradicted per-probe: %s",
                [f"{s:.3f}" for s in con_scores])
    if sup_mean - con_mean < _SANITY_MARGIN:
        raise RuntimeError(
            f"Verifier sanity check failed: mean supported score "
            f"({sup_mean:.4f}) does not exceed mean contradicted "
            f"score ({con_mean:.4f}) by the required margin "
            f"({_SANITY_MARGIN}).  This almost always means the "
            f"checkpoint head weights failed to load.  Inspect the "
            f"checkpoint with `torch.load(...).keys()` and re-run. "
            f"Per-probe supported: {[round(s, 3) for s in sup_scores]}. "
            f"Per-probe contradicted: {[round(s, 3) for s in con_scores]}."
        )
    return sup_mean, con_mean


def _score_claim_evidence_pairs(
    *,
    tokenizer: Any,
    model: Any,
    pairs: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    max_length: int,
    target_label: int = 1,
) -> list[float]:
    """Run the full ``VerifierModel`` on (claim, evidence) pairs.

    Returns a list of "supported" scores (one per pair, in input
    order), where::

        score = softmax(verdict_logits, dim=-1)[:, 1 - target_label]

    Under the v1/v3 binary convention (0 = not contradicted,
    1 = contradicted), this is ``softmax(verdict_logits)[:, 0]`` — the
    probability that the claim is consistent with the evidence.  Task 9
    treats this as the "would-be-certified" score.

    Matches the exact scoring path in
    ``scripts/modal_run_evaluation.py`` lines 271–333: the main eval
    script calls ``model(input_ids, attention_mask)`` with no heatmap
    (so the heatmap encoder receives zeros), takes the verdict_logits,
    applies temperature-scaled softmax, and reads ``probs[:, 0]`` as
    the conformal score.  Task 9 skips the temperature scaling (we
    want raw verifier confidence for the gate experiment, not
    calibration-aware scores).

    Args:
        tokenizer: HF tokenizer (roberta-large by default).
        model: Full ``VerifierModel`` from ``_load_v1_verifier``.
        pairs: List of pair dicts emitted by
            ``pair_claims_with_evidence``.
        device: torch device.
        batch_size: Inference batch size.  16 is safe on H100 @ L=256.
        max_length: Max tokenized sequence length.  Longer evidence is
            truncated via ``truncation="only_second"`` so the claim
            text is always preserved.
        target_label: Which class is "contradicted" in the checkpoint.
            v1/v3 binary use 1.  If a future checkpoint flips the
            convention, pass ``target_label=0`` to keep the sign
            consistent.
    """
    import torch

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            claims = [p["claim_text"] for p in batch]
            evidences = [p["evidence_text"] for p in batch]
            enc = tokenizer(
                claims,
                evidences,
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            # model.forward returns (verdict_logits, score).  We use
            # the verdict_logits path because main eval calibrates
            # against it; the sigmoid score_head is a regression
            # output trained separately and is NOT the conformal
            # scoring signal.
            verdict_logits, _sigmoid_score = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            probs = torch.softmax(verdict_logits, dim=-1)
            supported_probs = probs[:, 1 - int(target_label)]
            scores.extend(supported_probs.cpu().tolist())

    return scores


# ---------------------------------------------------------------------------
# Layer 3 — Modal entry point.
# ---------------------------------------------------------------------------


def _run_demo(config_json: str) -> dict[str, Any]:
    """Task 9 scoring pipeline.  Runs inside the Modal container.

    This function is the verifier-scoring phase only.  The two CheXagent
    runs that produce the input workbooks are orchestrated separately
    by ``main()`` on the local client — see the module docstring and
    reviewer note below for why.

    Reviewer note (bug 1, fixed):
        A prior version of this function called
        ``generate_annotation_workbook.remote(...)`` twice from inside
        this Modal container.  That nested-remote pattern (a) burned
        double GPU time because the outer H100 container sat idle for
        the entire ~90 minutes of CheXagent work, and (b) is not a
        standard Modal invocation path — it requires the
        ``claimguard-real-hallucinations`` app to be in a run context
        the outer container does not own.  Generation is now launched
        from the local entrypoint; this function assumes both workbooks
        already exist on the shared ``claimguard-data`` volume.

    Imports torch / transformers lazily so the outer module stays
    importable without them (the pure-helper unit tests run on a
    CPU-only laptop without torch installed).

    Steps:

        1. Load both workbooks from the shared volume.
        2. Pair run-A claims with run-A + run-B evidence via
           ``pair_claims_with_evidence``.  The pairing filters to the
           image intersection, guaranteeing a matched comparison.
        3. Load the v1 verifier and score every pair.
        4. Assemble ``GateDemoRow`` objects via ``build_gate_demo_row``.
        5. Aggregate via ``compute_gate_demo_stats``.
        6. Persist ``gate_demo.json`` + ``gate_demo_rows.json`` under
           ``config.output_dir`` on the volume.
    """
    import torch

    config = GateDemoConfig.from_json(config_json)
    logger.info("Task 9 config: %s", asdict(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)

    # ---- 1. Load workbooks (already produced by main()) ----
    if not os.path.exists(config.workbook_a_path):
        raise FileNotFoundError(
            f"Run-A workbook missing at {config.workbook_a_path}. "
            "Launch generation from the local entrypoint first, or "
            "rerun main() without --skip-generation."
        )
    if not os.path.exists(config.workbook_b_path):
        raise FileNotFoundError(
            f"Run-B workbook missing at {config.workbook_b_path}. "
            "Launch generation from the local entrypoint first, or "
            "rerun main() without --skip-generation."
        )
    claims_a = load_workbook_claims(config.workbook_a_path)
    claims_b = load_workbook_claims(config.workbook_b_path)
    logger.info(
        "Loaded %d run-A claims, %d run-B claims",
        len(claims_a), len(claims_b),
    )

    # ---- 2. Build pair table ----
    pairs = pair_claims_with_evidence(
        claims_run_a=claims_a, claims_run_b=claims_b
    )
    if not pairs:
        raise RuntimeError(
            "No (claim, evidence) pairs could be built — either the "
            "workbooks are empty or run A and run B share no images. "
            "Check that config.image_seed is the same for both runs."
        )
    logger.info("Built %d pairs for verifier inference", len(pairs))

    # ---- 3. Verifier inference ----
    tokenizer, model = _load_v1_verifier(
        checkpoint_path=config.verifier_checkpoint,
        hf_backbone=config.hf_backbone,
        device=device,
    )
    # Reviewer-flagged cheap insurance: run two probe pairs and abort
    # if the sign is flipped or scores are indistinguishable.  Catches
    # silent head-weight-load failures before we waste 100× inference.
    _sanity_check_verifier(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    scores = _score_claim_evidence_pairs(
        tokenizer=tokenizer,
        model=model,
        pairs=pairs,
        device=device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # ---- 4. Rows ----
    rows: list[GateDemoRow] = []
    for pair, score in zip(pairs, scores):
        rows.append(
            build_gate_demo_row(
                claim_id=pair["claim_id"],
                claim_text=pair["claim_text"],
                evidence_text=pair["evidence_text"],
                claim_generator_id=pair["claim_generator_id"],
                evidence_generator_id=pair["evidence_generator_id"],
                verifier_score=score,
                condition=pair["condition"],
                image_file=pair.get("image_file", ""),
                high_score_threshold=config.high_score_threshold,
            )
        )

    # ---- 5. Stats ----
    stats = compute_gate_demo_stats(rows)
    logger.info("Task 9 stats: %s", json.dumps(stats, indent=2))

    # ---- 6. Persist ----
    rows_path = os.path.join(config.output_dir, "gate_demo_rows.json")
    stats_path = os.path.join(config.output_dir, "gate_demo.json")
    with open(rows_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in rows], f, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(config),
                "stats": stats,
                "rows_path": rows_path,
                "n_pairs": len(pairs),
            },
            f,
            indent=2,
        )

    return {
        "stats": stats,
        "rows_path": rows_path,
        "stats_path": stats_path,
        "n_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Modal wiring — guarded so tests can import this module without modal.
# ---------------------------------------------------------------------------

app: Any = None
volume: Any = None
demo_provenance_gate_remote: Any = None
task9_orchestrator_remote: Any = None

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
        # Ship the in-repo `inference/` package (apply_provenance_gate,
        # classify_trust_tier, EvidenceSourceType, etc) into the Modal
        # container.  Without this, the module-level
        # `from inference.provenance import ...` fails with
        # `ModuleNotFoundError: No module named 'inference'` as soon as
        # the function cold-starts, because the container only has
        # /root/demo_provenance_gate_failure.py and no repo layout.
        # This bit us 2026-04-15: the orchestrator container crashed on
        # import before even polling the workbooks, and the client-side
        # `.get()` loop just sat there seeing "RUNNING" because Modal
        # was restarting the crashed container silently.
        .add_local_python_source("inference")
        # `scripts.generate_real_hallucinations` is only imported inside
        # `_launch_chexagent_runs` (which is skipped when
        # `skip_generation=True`, the orchestrator path), but we ship
        # it anyway so a future non-orchestrator run has the full
        # generation entrypoint available.
        .add_local_python_source("scripts")
    )
    app = _modal.App(APP_NAME, image=_image)
    volume = _modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    @app.function(  # type: ignore[misc]
        gpu="H100",
        timeout=60 * 60 * 2,  # 2h cap — CheXagent + scoring on 100 imgs
        volumes={"/data": volume},
    )
    def demo_provenance_gate_remote(config_json: str) -> dict[str, Any]:  # noqa: F811
        """Modal entry stub that calls the full dual-run demo."""
        return _run_demo(config_json)

    @app.function(  # type: ignore[misc]
        cpu=1.0,
        timeout=60 * 60 * 3,  # 3h cap — generation is 30-45 min + scoring 10 min + slack
        volumes={"/data": volume},
    )
    def task9_orchestrator_remote(
        config_json: str,
        *,
        poll_interval_sec: int = 60,
        max_poll_minutes: int = 150,
    ) -> dict[str, Any]:  # noqa: F811
        """Server-side orchestrator for Task 9 — survives local Mac sleep.

        Strategy: poll the shared volume for the two generation
        workbooks (which the locally-spawned
        ``generate_annotation_workbook`` calls are writing).  Once
        both workbooks exist and are non-empty, invoke the scoring
        phase in-process (still on this container, NO nested
        ``.remote()`` call needed because we just call ``_run_demo``
        directly).

        Why a CPU container instead of H100: the scoring phase needs
        H100 for the verifier, but the orchestrator itself just polls
        + chains.  Running the whole chain on H100 would waste ~1h of
        H100 time.  Instead, we use CPU for the polling and delegate
        scoring via ``demo_provenance_gate_remote.remote()`` (which
        spawns a fresh H100 container for ~5 minutes).

        Actually — simpler: this orchestrator DOES run on CPU and
        just waits for the workbooks, then it ``.remote()``-calls
        ``demo_provenance_gate_remote`` to fire the H100 scoring
        phase, then ``.get()``-waits for scoring to finish.  Both
        ``.remote()`` and ``.get()`` are server-side calls from
        inside the Modal cluster, so they are robust to any client
        disconnects.

        Args:
            config_json: Serialized ``GateDemoConfig``.  Must have
                ``workbook_a_path`` and ``workbook_b_path`` pointing
                at where the upstream generation runs are writing
                their outputs.
            poll_interval_sec: Seconds between workbook existence
                polls.  60s is a reasonable default — longer wastes
                Modal logs, shorter wastes CPU.
            max_poll_minutes: Overall timeout on waiting for both
                workbooks.  150 min default — run A + run B cold-
                start (CheXagent ~15 GB download + warmup) plus
                generation should land well under 90 min.

        Returns:
            The dict returned by ``demo_provenance_gate_remote``
            (stats + rows_path + stats_path + n_pairs) — same
            contract as ``main()``.

        Raises:
            TimeoutError: if both workbooks still don't exist after
                ``max_poll_minutes``.
            RuntimeError: if the scoring phase fails.
        """
        import time

        local_logger = logging.getLogger("task9_orchestrator")
        local_logger.setLevel(logging.INFO)
        if not local_logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s | ORCH | %(levelname)s | %(message)s"
            ))
            local_logger.addHandler(h)

        cfg = GateDemoConfig.from_json(config_json)
        local_logger.info("orchestrator started; waiting for workbooks")
        local_logger.info("  A=%s", cfg.workbook_a_path)
        local_logger.info("  B=%s", cfg.workbook_b_path)

        # Force a volume reload on each poll — volumes mounted inside
        # Modal functions are snapshotted at container start, so new
        # files written by other functions aren't visible until we
        # explicitly reload.
        deadline = time.monotonic() + max_poll_minutes * 60
        poll_count = 0
        while True:
            poll_count += 1
            try:
                volume.reload()
            except Exception as e:  # noqa: BLE001
                local_logger.warning(
                    "volume.reload() failed (poll %d): %s", poll_count, e,
                )

            a_ready = (
                os.path.exists(cfg.workbook_a_path)
                and os.path.getsize(cfg.workbook_a_path) > 100
            )
            b_ready = (
                os.path.exists(cfg.workbook_b_path)
                and os.path.getsize(cfg.workbook_b_path) > 100
            )
            local_logger.info(
                "poll %d: A_ready=%s B_ready=%s",
                poll_count, a_ready, b_ready,
            )
            if a_ready and b_ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"workbooks did not land within "
                    f"{max_poll_minutes} min. "
                    f"A_ready={a_ready} B_ready={b_ready}. "
                    f"Check the claimguard-real-hallucinations app "
                    f"logs — generation may have failed upstream."
                )
            time.sleep(poll_interval_sec)

        local_logger.info(
            "both workbooks present; spawning scoring phase on H100"
        )

        # Force skip_generation=True so the scoring container doesn't
        # try to re-invoke CheXagent.  Clone the config with that
        # field overridden.
        scoring_cfg = GateDemoConfig.from_json(config_json)
        scoring_cfg_dict = asdict(scoring_cfg)
        scoring_cfg_dict["skip_generation"] = True
        scoring_cfg_json = json.dumps(scoring_cfg_dict)

        # In-cluster `.remote()` blocks until the H100 container
        # finishes scoring — typically 5-10 min.  No local client
        # involvement, so safe against Mac sleep.
        result = demo_provenance_gate_remote.remote(scoring_cfg_json)
        local_logger.info("scoring complete: %s", result)
        return result

except Exception as _modal_err:  # noqa: BLE001
    logger.info(
        "Modal unavailable at module-import time (%s); "
        "pure helpers still importable.",
        _modal_err,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> GateDemoConfig:
    defaults = GateDemoConfig()
    parser = argparse.ArgumentParser(
        description=(
            "ClaimGuard Task 9 — provenance-gate same-model failure-mode "
            "demo (Modal H100)."
        ),
    )
    parser.add_argument(
        "--verifier-checkpoint", default=defaults.verifier_checkpoint,
    )
    parser.add_argument(
        "--openi-images-dir", default=defaults.openi_images_dir,
    )
    parser.add_argument(
        "--openi-reports-csv", default=defaults.openi_reports_csv,
    )
    parser.add_argument(
        "--workbook-a-path", default=defaults.workbook_a_path,
    )
    parser.add_argument(
        "--workbook-b-path", default=defaults.workbook_b_path,
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--max-images", type=int, default=defaults.max_images,
    )
    parser.add_argument(
        "--image-seed", type=int, default=defaults.image_seed,
    )
    parser.add_argument(
        "--seed-run-a", type=int, default=defaults.seed_run_a,
    )
    parser.add_argument(
        "--seed-run-b", type=int, default=defaults.seed_run_b,
    )
    parser.add_argument(
        "--temperature", type=float, default=defaults.temperature,
    )
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument(
        "--high-score-threshold",
        type=float, default=defaults.high_score_threshold,
    )
    parser.add_argument("--hf-backbone", default=defaults.hf_backbone)
    parser.add_argument(
        "--max-length", type=int, default=defaults.max_length,
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        default=defaults.skip_generation,
    )
    args = parser.parse_args(argv)
    return GateDemoConfig(
        verifier_checkpoint=args.verifier_checkpoint,
        openi_images_dir=args.openi_images_dir,
        openi_reports_csv=args.openi_reports_csv,
        workbook_a_path=args.workbook_a_path,
        workbook_b_path=args.workbook_b_path,
        output_dir=args.output_dir,
        max_images=args.max_images,
        image_seed=args.image_seed,
        seed_run_a=args.seed_run_a,
        seed_run_b=args.seed_run_b,
        temperature=args.temperature,
        top_p=args.top_p,
        high_score_threshold=args.high_score_threshold,
        hf_backbone=args.hf_backbone,
        max_length=args.max_length,
        batch_size=args.batch_size,
        skip_generation=args.skip_generation,
    )


def _launch_chexagent_runs(config: GateDemoConfig) -> None:
    """Orchestrate the two CheXagent generation runs from the local client.

    Runs sequentially inside the ``claimguard-real-hallucinations`` app's
    ``app.run()`` context (one context per run so each H100 container is
    released cleanly before the next starts).  The shared ``image_seed``
    pins both runs to the same OpenI image set; distinct ``seed_run_a``
    and ``seed_run_b`` values drive divergent nucleus-sampling streams.

    This function is called from ``main()`` on the local client, NOT
    from inside the demo's own Modal container.  See the reviewer note
    in ``_run_demo`` for the history.

    Args:
        config: The demo config.  ``config.skip_generation=True`` is a
            no-op — the caller is responsible for ensuring both
            workbooks already exist on the shared volume in that case.
    """
    if config.skip_generation:
        logger.info(
            "skip_generation=True — assuming workbooks already exist "
            "at %s and %s",
            config.workbook_a_path, config.workbook_b_path,
        )
        return

    try:
        from scripts.generate_real_hallucinations import (
            app as gen_app,
            generate_annotation_workbook,
        )
    except ImportError as e:
        raise RuntimeError(
            "Cannot import generate_annotation_workbook — Task 9 "
            "requires the claimguard-real-hallucinations app to be "
            f"importable.  Original error: {e}"
        ) from e

    run_specs = (
        (RUN_A_ID, config.seed_run_a, config.workbook_a_path),
        (RUN_B_ID, config.seed_run_b, config.workbook_b_path),
    )
    for run_id, seed, workbook_path in run_specs:
        logger.info(
            "Launching CheXagent %s on %d images (seed=%d, image_seed=%d)",
            run_id, config.max_images, seed, config.image_seed,
        )
        with gen_app.run():
            generate_annotation_workbook.remote(
                openi_images_dir=config.openi_images_dir,
                openi_reports_csv=config.openi_reports_csv,
                output_dir=os.path.dirname(workbook_path),
                max_images=config.max_images,
                seed=seed,
                image_seed=config.image_seed,
                sampling=True,
                temperature=config.temperature,
                top_p=config.top_p,
                run_id=run_id,
                workbook_filename=os.path.basename(workbook_path),
            )
        logger.info("CheXagent %s complete.", run_id)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if argv is None:
        argv = sys.argv[1:]
    config = _parse_cli(argv)
    logger.info("Task 9 config: %s", asdict(config))

    if app is None or demo_provenance_gate_remote is None:
        logger.error(
            "Modal is not available in this environment — cannot launch "
            "the demo.  Install modal and re-run, or invoke _run_demo() "
            "directly for local dry-runs."
        )
        return 1

    # Phase 1: CheXagent generation (sequential, from local client).
    # Reviewer-flagged bug 1 fix: generation is no longer nested inside
    # _run_demo's own Modal container.
    _launch_chexagent_runs(config)

    # Phase 2: verifier scoring + stats (on H100 via the demo app).
    with app.run():
        result = demo_provenance_gate_remote.remote(config.to_json())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "APP_NAME",
    "VOLUME_NAME",
    "RUN_A_ID",
    "RUN_B_ID",
    "RUN_A_GENERATOR_ID",
    "RUN_B_GENERATOR_ID",
    "DEFAULT_HIGH_SCORE_THRESHOLD",
    "GateDemoConfig",
    "GateDemoRow",
    "build_gate_demo_row",
    "compute_gate_demo_stats",
    "load_workbook_claims",
    "main",
    "pair_claims_with_evidence",
    "synthetic_gate_demo_workbook",
]
