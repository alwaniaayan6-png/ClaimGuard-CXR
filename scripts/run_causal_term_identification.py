"""Task 3a — run integrated-gradients causal-term identification on
v3 contradicted training claims.

Bridges v3 training data (`/data/verifier_training_data_v3.json`,
30k examples) and the Task 3b counterfactual generator
(`scripts/generate_counterfactual_pairs.py`).  Filters to label=1
(contradicted) claims, instantiates `CausalTermIdentifier` with the
v3 checkpoint, runs the cross-encoder forward + integrated gradients
on each (claim, evidence) pair, and writes a JSONL file with the
top-K causal token spans.

Output schema (one row per claim, JSONL)::

    {
        "claim_id": "v3_train_000123",
        "claim": "Dense right lower lobe consolidation.",
        "evidence": "joined evidence text",
        "causal_tokens": ["consolidation", "right", "lower"],
        "causal_spans_full": [
            {"text": "consolidation", "source": "claim",
             "score": 0.0421, "start_char": 6, "end_char": 19},
            ...
        ],
        "label": 1,
        "negative_type": "finding_substitution"
    }

The `causal_tokens` field is the simple list `[span.text for span in
top_k_spans]` — it's what `generate_counterfactual_pairs.py`
consumes.  The full `causal_spans_full` field preserves source +
score + char offsets for downstream auditing and is dropped when
the JSONL is loaded by the counterfactual driver.

Usage on Modal (H100, ~25 min for 30k claims, ~$4)::

    modal run --detach scripts/run_causal_term_identification.py \\
        --training-data /data/verifier_training_data_v3.json \\
        --output-jsonl /data/causal_spans_v3.jsonl \\
        --max-claims 30000 \\
        --top-k 5

Smoke test (CPU, 10 claims, ~30s)::

    modal run scripts/run_causal_term_identification.py \\
        --training-data /data/verifier_training_data_v3.json \\
        --output-jsonl /data/causal_spans_v3_smoke.jsonl \\
        --max-claims 10 \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any, Optional

# Make the in-repo packages importable when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

APP_NAME = "claimguard-causal-term-id"
VOLUME_NAME = "claimguard-data"


# ---------------------------------------------------------------------------
# Pure helpers (importable + unit-testable without modal/torch/captum)
# ---------------------------------------------------------------------------


def filter_contradicted_claims(
    claims: list[dict[str, Any]],
    max_claims: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Filter to contradicted claims and cap at ``max_claims``.

    Args:
        claims: List of training-data rows in the v3 schema (must
            have ``label`` field; 1 = contradicted, 0 = not).
        max_claims: Optional cap on the number of contradicted
            claims to return.  Selection is stable (first N in
            input order) so the same input file always produces the
            same output.

    Returns:
        Filtered list of contradicted claims, with stable ordering.
    """
    contra = [c for c in claims if c.get("label") == 1]
    if max_claims is not None and max_claims > 0:
        contra = contra[:max_claims]
    return contra


def make_claim_id(idx: int, prefix: str = "v3_train") -> str:
    """Generate a stable claim_id for the JSONL output."""
    return f"{prefix}_{idx:06d}"


def join_evidence(evidence: Any) -> str:
    """Convert the evidence field (str or list[str]) into a single
    string suitable for the cross-encoder.

    The training data uses a list of evidence chunks per claim.  We
    join with ' [SEP] ' to match the training-time tokenization
    (see ``ClaimDataset.__getitem__`` in
    ``scripts/modal_run_evaluation.py``).  Non-string entries are
    stringified.
    """
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        return " [SEP] ".join(
            str(e).strip() for e in evidence[:2] if str(e).strip()
        )
    return str(evidence)


def span_to_dict(span: Any) -> dict[str, Any]:
    """Serialize a CausalSpan to a JSON-safe dict.

    Drops the ``token_indices`` tuple because it's only useful for
    debugging and adds ~10x to the JSONL size for 30k rows.
    """
    return {
        "text": span.text,
        "source": span.source,
        "score": float(span.score),
        "start_char": int(span.start_char),
        "end_char": int(span.end_char),
    }


def build_output_row(
    *,
    claim_id: str,
    claim: str,
    evidence: str,
    spans: list[Any],
    label: int,
    negative_type: str,
) -> dict[str, Any]:
    """Build one JSONL row from the causal-term identification result."""
    return {
        "claim_id": claim_id,
        "claim": claim,
        "evidence": evidence,
        "causal_tokens": [s.text for s in spans],
        "causal_spans_full": [span_to_dict(s) for s in spans],
        "label": int(label),
        "negative_type": str(negative_type),
    }


# ---------------------------------------------------------------------------
# Heavy entry point — runs inside Modal H100 container
# ---------------------------------------------------------------------------


def _run_identification(
    *,
    training_data_path: str,
    output_jsonl_path: str,
    checkpoint_path: str,
    hf_backbone: str,
    top_k: int,
    n_ig_steps: int,
    max_claims: Optional[int],
    log_every: int,
) -> dict[str, Any]:
    """Full Task 3a identification loop.

    Imports torch/captum lazily so the outer module stays importable
    in CPU-only environments (for unit tests of the pure helpers).
    """
    import time

    # Lazy imports — the Modal container has these, the local
    # CPU may not.
    from data.augmentation.causal_term_identifier import (  # noqa: E402
        CausalTermIdentifier,
    )

    logger.info("Loading v3 training data from %s", training_data_path)
    with open(training_data_path, "r", encoding="utf-8") as f:
        all_claims = json.load(f)
    logger.info("Loaded %d total training rows", len(all_claims))

    contra = filter_contradicted_claims(all_claims, max_claims=max_claims)
    logger.info(
        "Filtered to %d contradicted rows (cap=%s)",
        len(contra), max_claims,
    )

    if not contra:
        raise RuntimeError(
            f"No contradicted (label=1) claims in {training_data_path}; "
            f"refusing to launch IG identification on an empty set."
        )

    logger.info(
        "Building CausalTermIdentifier(checkpoint=%s, backbone=%s, "
        "top_k=%d, n_ig_steps=%d)",
        checkpoint_path, hf_backbone, top_k, n_ig_steps,
    )
    cti = CausalTermIdentifier(
        model_path=checkpoint_path,
        hf_backbone=hf_backbone,
        target_label=1,  # contradicted = 1
        top_k=top_k,
        n_ig_steps=n_ig_steps,
    )
    logger.info("CausalTermIdentifier ready on device=%s", cti.device)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    n_written = 0
    n_failed = 0
    started = time.time()
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(contra):
            try:
                claim = str(row["claim"]).strip()
                evidence = join_evidence(row.get("evidence", ""))
                if not claim:
                    n_failed += 1
                    continue
                spans = cti.identify(
                    claim=claim,
                    evidence=evidence,
                    top_k=top_k,
                    max_length=256,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "claim_id=%d failed: %s",
                    idx, type(e).__name__,
                )
                n_failed += 1
                continue

            out_row = build_output_row(
                claim_id=make_claim_id(idx),
                claim=claim,
                evidence=evidence,
                spans=spans,
                label=int(row.get("label", 1)),
                negative_type=str(row.get("negative_type", "")),
            )
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_written += 1

            if (idx + 1) % log_every == 0:
                elapsed = time.time() - started
                rate = (idx + 1) / max(elapsed, 1.0)
                eta = (len(contra) - (idx + 1)) / max(rate, 0.01)
                logger.info(
                    "[%d/%d] written=%d failed=%d rate=%.1f/s eta=%.0fs",
                    idx + 1, len(contra), n_written, n_failed, rate, eta,
                )

    elapsed = time.time() - started
    summary = {
        "training_data_path": training_data_path,
        "output_jsonl_path": output_jsonl_path,
        "n_total_rows": len(all_claims),
        "n_contradicted_input": len(contra),
        "n_written": n_written,
        "n_failed": n_failed,
        "wall_time_sec": round(elapsed, 1),
        "rate_per_sec": round(n_written / max(elapsed, 1.0), 2),
    }
    logger.info("DONE: %s", json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Modal wiring — guarded so unit tests can import without modal
# ---------------------------------------------------------------------------

app: Any = None
volume: Any = None
run_causal_term_identification_remote: Any = None

try:
    import modal as _modal  # noqa: WPS433

    _image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            [
                "torch==2.3.0",
                "transformers==4.40.0",
                "captum==0.7.0",
                "numpy<2",
                "huggingface_hub<0.25",
            ]
        )
        # D20 fix: ship in-repo packages so the container can import
        # `inference.verifier_model.load_verifier_checkpoint` (used
        # by CausalTermIdentifier) and
        # `data.augmentation.causal_term_identifier`.
        .add_local_python_source("inference")
        .add_local_python_source("data")
    )
    app = _modal.App(APP_NAME, image=_image)
    volume = _modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    @app.function(  # type: ignore[misc]
        gpu="H100",
        timeout=60 * 60 * 2,  # 2h cap
        volumes={"/data": volume},
    )
    def run_causal_term_identification_remote(  # noqa: F811
        training_data_path: str,
        output_jsonl_path: str,
        checkpoint_path: str = (
            "/data/checkpoints/verifier_binary_v3/best_verifier.pt"
        ),
        hf_backbone: str = "roberta-large",
        top_k: int = 5,
        n_ig_steps: int = 50,
        max_claims: Optional[int] = None,
        log_every: int = 200,
    ) -> dict[str, Any]:
        """Modal entry stub for Task 3a IG identification."""
        return _run_identification(
            training_data_path=training_data_path,
            output_jsonl_path=output_jsonl_path,
            checkpoint_path=checkpoint_path,
            hf_backbone=hf_backbone,
            top_k=top_k,
            n_ig_steps=n_ig_steps,
            max_claims=max_claims,
            log_every=log_every,
        )

    @app.local_entrypoint()  # type: ignore[misc]
    def main(
        training_data: str = "/data/verifier_training_data_v3.json",
        output_jsonl: str = "/data/causal_spans_v3.jsonl",
        checkpoint: str = (
            "/data/checkpoints/verifier_binary_v3/best_verifier.pt"
        ),
        hf_backbone: str = "roberta-large",
        top_k: int = 5,
        n_ig_steps: int = 50,
        max_claims: Optional[int] = None,
        log_every: int = 200,
    ) -> None:
        """Local entry point — calls the remote H100 function.

        Defaults run the full v3 training set (no cap).  Pass
        ``--max-claims 10`` for a smoke test.
        """
        print("Launching Task 3a (causal-term identification) on Modal...")
        print(f"  training: {training_data}")
        print(f"  checkpoint: {checkpoint}")
        print(f"  output:   {output_jsonl}")
        print(f"  top_k={top_k}, n_ig_steps={n_ig_steps}, "
              f"max_claims={max_claims}")
        result = run_causal_term_identification_remote.remote(
            training_data_path=training_data,
            output_jsonl_path=output_jsonl,
            checkpoint_path=checkpoint,
            hf_backbone=hf_backbone,
            top_k=top_k,
            n_ig_steps=n_ig_steps,
            max_claims=max_claims,
            log_every=log_every,
        )
        print("\n=== TASK 3A COMPLETE ===")
        print(json.dumps(result, indent=2))

except Exception as _modal_err:  # noqa: BLE001
    logger.info(
        "Modal unavailable at module-import time (%s); "
        "pure helpers still importable.",
        _modal_err,
    )


# ---------------------------------------------------------------------------
# Local CLI (does NOT use Modal — for unit tests + CPU smoke runs)
# ---------------------------------------------------------------------------


def _local_cli(argv: list[str]) -> int:
    """Run the identification loop locally without Modal.

    For CPU smoke tests only — IG attribution on RoBERTa-large is
    very slow on CPU (~30s per claim).
    """
    parser = argparse.ArgumentParser(
        description="Task 3a causal-term identification (LOCAL CPU)."
    )
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v1_best_verifier.pt",
        help="Local checkpoint path (NOT the Modal volume path).",
    )
    parser.add_argument("--hf-backbone", default="roberta-large")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-ig-steps", type=int, default=20)
    parser.add_argument("--max-claims", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    summary = _run_identification(
        training_data_path=args.training_data,
        output_jsonl_path=args.output_jsonl,
        checkpoint_path=args.checkpoint,
        hf_backbone=args.hf_backbone,
        top_k=args.top_k,
        n_ig_steps=args.n_ig_steps,
        max_claims=args.max_claims,
        log_every=args.log_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_local_cli(sys.argv[1:]))
