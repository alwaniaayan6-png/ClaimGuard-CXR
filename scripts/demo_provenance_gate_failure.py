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

    verifier_checkpoint: str = (
        "/data/checkpoints/verifier_binary/best_verifier.pt"
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


def _load_v1_verifier(
    checkpoint_path: str,
    hf_backbone: str,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Load the v1 binary verifier and return (tokenizer, encoder, head).

    v1 / v3 / v4 checkpoints all use the same two-head layout::

        state = {"encoder": <AutoModel state_dict>,
                 "head":    <Linear(hidden, 2) state_dict>}

    Legacy checkpoints may instead ship a flat ``model_state_dict`` with
    ``text_encoder.*`` prefixes (the ``VerifierModel`` layout from
    ``scripts/modal_run_evaluation.py``).  We handle both by detecting
    the key layout at load time; the head weights from the legacy
    shape are dropped since Task 9 only needs the encoder.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_backbone)
    encoder = AutoModel.from_pretrained(hf_backbone).to(device)
    head = torch.nn.Linear(encoder.config.hidden_size, 2).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "encoder" in state:
        encoder.load_state_dict(state["encoder"], strict=False)
        if "head" in state:
            head.load_state_dict(state["head"], strict=False)
    elif isinstance(state, dict) and "model_state_dict" in state:
        encoder_state = {
            k.replace("text_encoder.", ""): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("text_encoder.")
        }
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        encoder.load_state_dict(state, strict=False)

    encoder.eval()
    head.eval()
    return tokenizer, encoder, head


# Two radiology-appropriate probe pairs used by ``_sanity_check_verifier``.
# Both use the SAME claim with opposite-polarity evidence so a correctly
# loaded verifier produces a clear score flip; a silently broken load
# (strict=False tolerates missing head keys, leaving random weights)
# produces near-identical scores.  The minimum acceptable margin is
# ``_SANITY_MARGIN`` (0.1 — small enough to tolerate noise, large enough
# to catch a random-weight head).
_SANITY_SUPPORTED = (
    "There is evidence of cardiomegaly.",
    "Findings: The cardiac silhouette is enlarged. "
    "Impression: Cardiomegaly.",
)
_SANITY_CONTRADICTED = (
    "There is evidence of cardiomegaly.",
    "Findings: The heart is normal in size. "
    "Impression: No cardiomegaly.",
)
_SANITY_MARGIN = 0.1


def _sanity_check_verifier(
    *,
    tokenizer: Any,
    encoder: Any,
    head: Any,
    device: Any,
    target_label: int = 1,
) -> tuple[float, float]:
    """Score two probe pairs to catch silent checkpoint-load failures.

    Reviewer-flagged cheap insurance: ``_load_v1_verifier`` uses
    ``strict=False`` when loading both encoder and head, which silently
    tolerates missing keys.  A checkpoint with renamed or missing head
    keys would leave the ``Linear(hidden, 2)`` layer with random
    initialization, and every downstream "supported score" in Task 9
    would be uniform noise around 0.5.  Worse: the demo would still
    produce a table that *looks* plausible.

    This function runs two probe pairs through the loaded verifier:

        * A near-identical (supported) pair with cardiomegaly evidence.
        * The same claim with opposite-polarity (contradicted) evidence.

    A correctly loaded verifier produces a clear sign flip; we require
    the supported-pair score to exceed the contradicted-pair score by
    at least ``_SANITY_MARGIN`` (= 0.1).  If it doesn't, we raise
    ``RuntimeError`` so the whole demo aborts before wasting H100 time
    on garbage inference.

    Returns:
        ``(supported_score, contradicted_score)`` so callers can log
        the actual values for debugging.

    Raises:
        RuntimeError: if the score margin is below ``_SANITY_MARGIN``.
    """
    pairs = [
        {
            "claim_text": _SANITY_SUPPORTED[0],
            "evidence_text": _SANITY_SUPPORTED[1],
        },
        {
            "claim_text": _SANITY_CONTRADICTED[0],
            "evidence_text": _SANITY_CONTRADICTED[1],
        },
    ]
    scores = _score_claim_evidence_pairs(
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
        pairs=pairs,
        device=device,
        batch_size=2,
        max_length=128,
        target_label=target_label,
    )
    sup_score, con_score = scores[0], scores[1]
    logger.info(
        "Verifier sanity probes — supported=%.4f, contradicted=%.4f, "
        "margin=%.4f",
        sup_score, con_score, sup_score - con_score,
    )
    if sup_score - con_score < _SANITY_MARGIN:
        raise RuntimeError(
            f"Verifier sanity check failed: supported-probe score "
            f"({sup_score:.4f}) does not exceed contradicted-probe "
            f"score ({con_score:.4f}) by the required margin "
            f"({_SANITY_MARGIN}).  This almost always means the "
            f"checkpoint head weights failed to load (strict=False "
            f"silently ignored missing keys).  Inspect the checkpoint "
            f"with `torch.load(...).keys()` and re-run."
        )
    return sup_score, con_score


def _score_claim_evidence_pairs(
    *,
    tokenizer: Any,
    encoder: Any,
    head: Any,
    pairs: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    max_length: int,
    target_label: int = 1,
) -> list[float]:
    """Run the verifier on a list of (claim, evidence) pairs.

    Returns a list of "supported" scores (one per pair, in input order),
    where score = ``softmax(logits)[not_contradicted_class]``.  Under
    the v1 binary convention (0 = supported, 1 = contradicted), this
    is ``softmax(logits)[0]`` — the probability that the claim is
    consistent with the evidence.  Task 9 treats this as the "would-be-
    certified" score.

    Args:
        tokenizer: HF tokenizer (roberta-large by default).
        encoder: HF AutoModel loaded from the v1 checkpoint.
        head: torch.nn.Linear(hidden_size, 2) loaded from the checkpoint.
        pairs: List of pair dicts emitted by ``pair_claims_with_evidence``.
        device: torch device.
        batch_size: Inference batch size.  16 is safe on H100 @ L=256.
        max_length: Max tokenized sequence length.  Longer evidence is
            truncated via ``truncation="only_second"`` so the claim text
            is always preserved.
        target_label: Which class is "contradicted" in the checkpoint.
            v1 binary uses 1.  If a future checkpoint flips the convention,
            pass ``target_label=0`` to keep the sign consistent.
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
            out = encoder(
                **enc, return_dict=True
            ).last_hidden_state[:, 0, :]
            logits = head(out)
            probs = torch.softmax(logits, dim=-1)
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
    tokenizer, encoder, head = _load_v1_verifier(
        checkpoint_path=config.verifier_checkpoint,
        hf_backbone=config.hf_backbone,
        device=device,
    )
    # Reviewer-flagged cheap insurance: run two probe pairs and abort
    # if the sign is flipped or scores are indistinguishable.  Catches
    # silent head-weight-load failures before we waste 100× inference.
    _sanity_check_verifier(
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
        device=device,
    )
    scores = _score_claim_evidence_pairs(
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
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
        timeout=60 * 60 * 2,  # 2h cap — CheXagent + scoring on 100 imgs
        volumes={"/data": volume},
    )
    def demo_provenance_gate_remote(config_json: str) -> dict[str, Any]:  # noqa: F811
        """Modal entry stub that calls the full dual-run demo."""
        return _run_demo(config_json)

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
