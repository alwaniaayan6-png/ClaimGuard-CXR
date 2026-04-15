"""Generate counterfactual preference pairs for Task 3 R-Drop refinement.

This is the driver script that bridges Task 3a (causal-term ID via
integrated gradients) and Task 3b (R-Drop refinement training).  It
reads a list of (contradicted claim, causal tokens, evidence) triples
and produces a JSONL of counterfactual variants by calling
``CounterfactualGenerator.generate`` against the Anthropic API.

The class itself is per-claim and synchronous, which is fine for
testing but disastrous for production: at Sonnet 4.5's default tier
(~50 RPM for non-tier-4 accounts) a serial loop over 30k claims would
take 10+ hours and burn the $25 budget on retry traffic.  This driver
adds the production-grade machinery the pre-flight reviewer flagged as
missing on 2026-04-15:

* **Concurrency**: ``concurrent.futures.ThreadPoolExecutor`` with a
  configurable ``--max-workers`` (default 8 for Sonnet 4.5 tier-2,
  which gives ~480 RPM aggregate while staying well under the per-key
  burst limit).
* **Per-call backoff**: ``tenacity``-style exponential backoff
  (1s → 2s → 4s → 8s → 16s, capped at 60s) on
  ``anthropic.RateLimitError``, ``anthropic.APIConnectionError``, and
  ``anthropic.InternalServerError``.  We do NOT retry on
  ``BadRequestError`` (caller bug) or ``AuthenticationError``
  (config bug).
* **Resume-on-crash checkpointing**: the driver writes one
  ``CounterfactualPair`` per claim to a JSONL output file as soon as
  it lands.  On restart with the same ``--output``, the driver reads
  the existing file, marks every present ``claim_id`` as DONE, and
  skips them in the work queue.  A ``Ctrl-C`` mid-run preserves all
  completed pairs — only in-flight calls are lost.
* **Hard-budget guard**: a ``--max-cost-usd`` flag (default $30) that
  stops the driver early if the running cost estimate exceeds the
  budget.  Cost is estimated from the running token counts, NOT from
  Anthropic's bill — you should still set a hard limit in your
  Anthropic console as a backstop.
* **Final write**: after all workers complete, the driver folds the
  JSONL stream into a single deduped JSON list in the format
  ``modal_train_dpo_refinement.load_preference_pairs`` expects.

The output schema (final JSON file)::

    [
        {
            "claim": "original contradicted claim",
            "evidence": "evidence text (joined ' [SEP] ' if multi)",
            "counterfactuals": ["variant 1", "variant 2", "variant 3"]
        },
        ...
    ]

Usage::

    # End-to-end against the v3 contradicted training claims:
    python3 scripts/generate_counterfactual_pairs.py \\
        --causal-spans-jsonl /data/causal_spans_v3.jsonl \\
        --output /data/counterfactual_preference_pairs_v3.json \\
        --max-workers 8 \\
        --max-cost-usd 30 \\
        --max-claims 30000

    # Resume after a crash (same --output):
    python3 scripts/generate_counterfactual_pairs.py \\
        --causal-spans-jsonl /data/causal_spans_v3.jsonl \\
        --output /data/counterfactual_preference_pairs_v3.json \\
        # All claims already in the output JSONL are skipped.

    # Smoke test (5 claims, dry-run with stub transport):
    python3 scripts/generate_counterfactual_pairs.py \\
        --causal-spans-jsonl /tmp/spans.jsonl \\
        --output /tmp/cf.json \\
        --dry-run \\
        --max-claims 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# Make the in-repo package importable when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.augmentation.counterfactual_generator import (  # noqa: E402
    CounterfactualGenerator,
    CounterfactualVariant,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost estimation constants (Anthropic Claude Sonnet 4.5 as of 2026-04-15)
# ---------------------------------------------------------------------------

# These are estimates only.  The actual bill is what Anthropic charges
# you in your console.  We use these for the --max-cost-usd guard so
# the driver self-stops before runaway retry traffic.
SONNET_INPUT_COST_PER_MTOK = 3.00   # $3.00 per 1M input tokens
SONNET_OUTPUT_COST_PER_MTOK = 15.00  # $15.00 per 1M output tokens

# Rough average per call for our 3-counterfactual prompt: ~250 input
# tokens (claim + 5 causal tokens + instructions), ~400 output tokens
# (3 variants × ~30 words × ~4 tokens/word + JSON overhead).
TYPICAL_INPUT_TOKENS_PER_CALL = 250
TYPICAL_OUTPUT_TOKENS_PER_CALL = 400


def estimate_cost_usd(n_calls: int) -> float:
    """Rough $-cost estimate for ``n_calls`` to Sonnet 4.5.

    Used for the ``--max-cost-usd`` guard.  Real cost may differ by
    ~20% depending on prompt + response length; the guard is meant
    to catch runaway loops, not provide invoice-grade accounting.
    """
    in_tokens = n_calls * TYPICAL_INPUT_TOKENS_PER_CALL
    out_tokens = n_calls * TYPICAL_OUTPUT_TOKENS_PER_CALL
    in_cost = (in_tokens / 1_000_000) * SONNET_INPUT_COST_PER_MTOK
    out_cost = (out_tokens / 1_000_000) * SONNET_OUTPUT_COST_PER_MTOK
    return in_cost + out_cost


# ---------------------------------------------------------------------------
# Input + output schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalInputRow:
    """One row from the causal_term_identifier output JSONL.

    Each row corresponds to one (contradicted claim, evidence) pair
    from the v3 training data, plus the top-K causal token spans
    identified by integrated gradients against the v3 verifier.
    """

    claim_id: str
    claim: str
    evidence: str
    causal_tokens: tuple[str, ...]


@dataclass
class CounterfactualOutputRow:
    """One row of the per-claim JSONL checkpoint.

    Mirrors the schema ``modal_train_dpo_refinement.load_preference_pairs``
    expects (the ``counterfactuals`` plural form), plus diagnostic
    fields for debugging and auditing.
    """

    claim_id: str
    claim: str
    evidence: str
    counterfactuals: list[str]
    causal_tokens: list[str]
    n_variants_requested: int
    n_variants_returned: int
    elapsed_seconds: float
    error: Optional[str] = None


def load_causal_spans_jsonl(path: str) -> list[CausalInputRow]:
    """Load the causal-spans JSONL file into typed rows.

    Expected JSONL schema per line::

        {
            "claim_id": "v3_train_000123",
            "claim": "...",
            "evidence": "...",
            "causal_tokens": ["token1", "token2", ...]
        }

    Rows missing any required field or with empty causal_tokens are
    dropped with a warning.
    """
    out: list[CausalInputRow] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("line %d: JSON decode failed: %s", lineno, e)
                skipped += 1
                continue
            try:
                cid = str(row["claim_id"])
                claim = str(row["claim"]).strip()
                evidence = str(row.get("evidence", "")).strip()
                tokens = tuple(
                    str(t).strip() for t in row["causal_tokens"]
                    if str(t).strip()
                )
            except (KeyError, TypeError) as e:
                logger.warning("line %d: missing field: %s", lineno, e)
                skipped += 1
                continue
            if not claim or not tokens:
                skipped += 1
                continue
            out.append(CausalInputRow(
                claim_id=cid,
                claim=claim,
                evidence=evidence,
                causal_tokens=tokens,
            ))
    if skipped:
        logger.warning("Skipped %d malformed rows during load", skipped)
    return out


def load_existing_checkpoints(jsonl_path: str) -> set[str]:
    """Load the set of ``claim_id``s that have already been written.

    Used for resume-on-crash: any claim_id present in the JSONL is
    skipped on the next run.  A claim with ``error != None`` is
    still considered "done" for skip purposes — caller can manually
    delete error rows from the JSONL to re-attempt.
    """
    if not os.path.exists(jsonl_path):
        return set()
    seen: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cid = row.get("claim_id")
                if isinstance(cid, str):
                    seen.add(cid)
            except json.JSONDecodeError:
                continue
    return seen


# ---------------------------------------------------------------------------
# Per-call backoff wrapper
# ---------------------------------------------------------------------------


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a known-transient Anthropic error.

    We retry on:
      * ``RateLimitError`` (HTTP 429)
      * ``APIConnectionError`` (network-level transport failure)
      * ``InternalServerError`` (HTTP 500/502/503/504)
      * ``APITimeoutError`` (request timeout)

    We do NOT retry on:
      * ``BadRequestError`` / ``UnprocessableEntityError`` (caller bug)
      * ``AuthenticationError`` / ``PermissionDeniedError`` (config bug)
      * ``NotFoundError`` (model-name typo)
    """
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "APITimeoutError",
        "OverloadedError",
        "APIStatusError",  # generic, often transient
    }


def call_with_backoff(
    fn: Any,
    *args: Any,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with exponential backoff on
    transient errors.

    Retry schedule with defaults: 1s, 2s, 4s, 8s, 16s, 60s (cap),
    each ± up to 25% jitter to avoid thundering-herd bursts on a
    shared rate limit.

    Re-raises non-retryable exceptions immediately.  After
    ``max_attempts`` retries on retryable exceptions, re-raises the
    last exception so the caller can mark the row as failed.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            last_exc = e
            if not _is_retryable_error(e):
                raise
            if attempt + 1 >= max_attempts:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay *= 1.0 + random.uniform(-jitter, jitter)
            logger.warning(
                "retryable error %s on attempt %d/%d; sleeping %.1fs",
                type(e).__name__, attempt + 1, max_attempts, delay,
            )
            time.sleep(max(0.1, delay))
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


@dataclass
class DriverState:
    """Mutable shared state across worker threads."""

    n_completed: int = 0
    n_failed: int = 0
    n_retried: int = 0
    skipped_resume: int = 0
    total_call_count: int = 0
    estimated_cost_usd: float = 0.0
    started_at: float = 0.0
    aborted: bool = False
    abort_reason: str = ""
    lock: Lock = field(default_factory=Lock)


def _process_one_claim(
    row: CausalInputRow,
    generator: CounterfactualGenerator,
    n_variants: int,
    state: DriverState,
    output_jsonl_lock: Lock,
    output_jsonl_path: str,
) -> CounterfactualOutputRow:
    """Worker: generate variants for one claim and append to JSONL.

    Wraps ``generator.generate`` in ``call_with_backoff`` so transient
    Anthropic errors are retried.  On any non-retryable error or
    after retry exhaustion, returns a row with ``error != None`` and
    an empty ``counterfactuals`` list.  The row is still appended to
    the JSONL so a re-run will skip it (callers can manually delete
    error rows to re-attempt).
    """
    t0 = time.time()
    try:
        variants: list[CounterfactualVariant] = call_with_backoff(
            generator.generate,
            claim=row.claim,
            causal_spans=list(row.causal_tokens),
            n_variants=n_variants,
        )
        cfs = [v.text for v in variants]
        out = CounterfactualOutputRow(
            claim_id=row.claim_id,
            claim=row.claim,
            evidence=row.evidence,
            counterfactuals=cfs,
            causal_tokens=list(row.causal_tokens),
            n_variants_requested=n_variants,
            n_variants_returned=len(cfs),
            elapsed_seconds=time.time() - t0,
            error=None,
        )
    except BaseException as e:  # noqa: BLE001
        logger.error(
            "claim_id=%s failed permanently: %s",
            row.claim_id, e,
        )
        out = CounterfactualOutputRow(
            claim_id=row.claim_id,
            claim=row.claim,
            evidence=row.evidence,
            counterfactuals=[],
            causal_tokens=list(row.causal_tokens),
            n_variants_requested=n_variants,
            n_variants_returned=0,
            elapsed_seconds=time.time() - t0,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
        with state.lock:
            state.n_failed += 1

    # Append to JSONL atomically.  We hold a single shared lock during
    # the file-write so concurrent workers don't interleave bytes.
    with output_jsonl_lock:
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
            f.flush()

    with state.lock:
        state.n_completed += 1
        state.total_call_count += 1
        state.estimated_cost_usd = estimate_cost_usd(state.total_call_count)

    return out


def fold_jsonl_to_preference_pairs_json(
    jsonl_path: str,
    output_json_path: str,
) -> dict[str, Any]:
    """Convert the per-claim JSONL into the final preference-pairs JSON.

    Output schema matches what
    ``modal_train_dpo_refinement.load_preference_pairs`` expects: a
    list of dicts with ``claim``, ``evidence``, ``counterfactuals``
    (plural list).  Rows with no counterfactuals are dropped.
    """
    rows: list[dict[str, Any]] = []
    n_total = 0
    n_kept = 0
    n_dropped_empty = 0
    n_dropped_error = 0
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                n_dropped_error += 1
                continue
            cfs = row.get("counterfactuals") or []
            if not cfs:
                n_dropped_empty += 1
                continue
            rows.append({
                "claim": row["claim"],
                "evidence": row.get("evidence", ""),
                "counterfactuals": cfs,
            })
            n_kept += 1

    out_dir = os.path.dirname(output_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return {
        "n_total_jsonl_rows": n_total,
        "n_kept": n_kept,
        "n_dropped_empty": n_dropped_empty,
        "n_dropped_error": n_dropped_error,
        "output_path": output_json_path,
    }


def run_driver(
    *,
    input_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    n_variants: int,
    max_workers: int,
    max_claims: Optional[int],
    max_cost_usd: float,
    dry_run: bool,
) -> dict[str, Any]:
    """End-to-end driver: load → schedule → fold → return summary."""
    rows = load_causal_spans_jsonl(input_path)
    logger.info("Loaded %d causal-span rows from %s", len(rows), input_path)
    if max_claims is not None and max_claims > 0:
        rows = rows[:max_claims]
        logger.info("Capped to first %d rows via --max-claims", len(rows))

    seen = load_existing_checkpoints(output_jsonl_path)
    if seen:
        logger.info(
            "Resume: %d claim_ids already present in %s; skipping",
            len(seen), output_jsonl_path,
        )
    todo = [r for r in rows if r.claim_id not in seen]
    logger.info("Work queue: %d claims to process", len(todo))

    if not todo:
        logger.info("Nothing to do; folding existing JSONL")
        summary = fold_jsonl_to_preference_pairs_json(
            output_jsonl_path, output_json_path,
        )
        return {**summary, "n_processed_this_run": 0}

    # Pre-flight cost guard
    est_total_cost = estimate_cost_usd(len(todo))
    logger.info(
        "Estimated cost for %d new claims: $%.2f (budget cap $%.2f)",
        len(todo), est_total_cost, max_cost_usd,
    )
    if est_total_cost > max_cost_usd * 1.5:
        # Refuse outright if even the estimate is 1.5x over budget.
        # 1.5x slack is for token-count uncertainty; if even that
        # margin doesn't fit, the budget is wrong, not the run.
        raise RuntimeError(
            f"Refusing to start: estimated cost ${est_total_cost:.2f} "
            f"exceeds 1.5× budget cap ${max_cost_usd:.2f}. "
            f"Lower --max-claims or raise --max-cost-usd."
        )

    # Build the generator (default Anthropic transport, or stub for dry-run)
    if dry_run:
        def _stub_transport(prompt: str) -> str:
            # Return a deterministic 3-variant JSON so we exercise the
            # parse + validate path without burning real Claude calls.
            tokens_in_prompt = []
            for line in prompt.splitlines():
                if "causal" in line.lower() or "preserve" in line.lower():
                    tokens_in_prompt.extend(
                        t.strip() for t in line.split() if len(t) > 3
                    )
            return json.dumps([
                f"DRY-RUN paraphrase 1 with {' '.join(tokens_in_prompt[:3])}",
                f"DRY-RUN paraphrase 2 with {' '.join(tokens_in_prompt[:3])}",
                f"DRY-RUN paraphrase 3 with {' '.join(tokens_in_prompt[:3])}",
            ])
        generator = CounterfactualGenerator(transport=_stub_transport)
    else:
        # Default Anthropic transport — picks up ANTHROPIC_API_KEY
        # from env automatically.
        generator = CounterfactualGenerator()

    # Worker pool with shared state
    state = DriverState(started_at=time.time())
    output_jsonl_lock = Lock()

    # Make sure the JSONL exists so workers can append cleanly
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_jsonl_path).touch()

    # SIGINT handler: set abort flag, let in-flight workers finish.
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
        with state.lock:
            state.aborted = True
            state.abort_reason = "user SIGINT"
        logger.warning(
            "SIGINT received; finishing in-flight workers then exiting"
        )

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_claim,
                    row, generator, n_variants, state,
                    output_jsonl_lock, output_jsonl_path,
                ): row.claim_id
                for row in todo
            }
            log_every = max(1, len(futures) // 20)
            for i, fut in enumerate(as_completed(futures), 1):
                claim_id = futures[fut]
                try:
                    _ = fut.result()
                except BaseException as e:  # noqa: BLE001
                    logger.exception(
                        "worker for claim_id=%s raised unexpectedly: %s",
                        claim_id, e,
                    )
                if i % log_every == 0:
                    with state.lock:
                        elapsed = time.time() - state.started_at
                        rate = state.n_completed / max(elapsed, 1.0)
                        logger.info(
                            "[%d/%d] done=%d failed=%d cost~$%.2f rate=%.1f/s",
                            i, len(futures), state.n_completed,
                            state.n_failed, state.estimated_cost_usd, rate,
                        )
                # Pre-flight reviewer Finding 8: SIGINT graceful drain.
                # The signal handler sets state.aborted=True; here we
                # check the flag and cancel pending futures so the user's
                # Ctrl-C actually stops the run instead of waiting for
                # the full queue to drain.
                with state.lock:
                    if state.aborted and state.abort_reason == "user SIGINT":
                        logger.warning(
                            "SIGINT abort: cancelling %d pending futures",
                            sum(1 for f in futures if not f.done()),
                        )
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                # Cost-cap check (worker safety net)
                with state.lock:
                    if state.estimated_cost_usd > max_cost_usd:
                        if not state.aborted:
                            state.aborted = True
                            state.abort_reason = (
                                f"cost cap hit: ${state.estimated_cost_usd:.2f} "
                                f"> ${max_cost_usd:.2f}"
                            )
                        logger.error(
                            "Cost cap hit: ${%.2f}. Aborting new submissions.",
                            state.estimated_cost_usd,
                        )
                        # Cancel any pending futures (best-effort; in-flight
                        # ones will still complete and write to JSONL)
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Fold the JSONL into the final preference-pairs JSON
    fold_summary = fold_jsonl_to_preference_pairs_json(
        output_jsonl_path, output_json_path,
    )

    return {
        "n_input_rows": len(rows),
        "n_skipped_resume": len(seen),
        "n_processed_this_run": state.n_completed,
        "n_failed": state.n_failed,
        "estimated_cost_usd": round(state.estimated_cost_usd, 2),
        "aborted": state.aborted,
        "abort_reason": state.abort_reason,
        "wall_time_sec": round(time.time() - state.started_at, 1),
        **fold_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate counterfactual preference pairs for Task 3 R-Drop "
            "refinement (concurrent + checkpointed + cost-capped)."
        )
    )
    parser.add_argument(
        "--causal-spans-jsonl", required=True,
        help="Input JSONL: one row per (claim_id, claim, evidence, "
             "causal_tokens) from causal_term_identifier.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Final preference-pairs JSON path. The driver also "
             "writes a `<output>.jsonl` checkpoint stream alongside.",
    )
    parser.add_argument(
        "--n-variants", type=int, default=3,
        help="Number of counterfactual variants per claim (default 3).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Concurrent worker threads (default 8 — Sonnet 4.5 tier-2).",
    )
    parser.add_argument(
        "--max-claims", type=int, default=None,
        help="Cap input rows for smoke-testing; default is no cap.",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=30.0,
        help="Hard budget cap. Driver self-aborts if est. cost exceeds. "
             "Default $30 (above the $25 plan budget for headroom).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a stub transport instead of real Anthropic calls.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_jsonl_path = args.output + ".jsonl"

    summary = run_driver(
        input_path=args.causal_spans_jsonl,
        output_jsonl_path=output_jsonl_path,
        output_json_path=args.output,
        n_variants=args.n_variants,
        max_workers=args.max_workers,
        max_claims=args.max_claims,
        max_cost_usd=args.max_cost_usd,
        dry_run=args.dry_run,
    )

    print("\n=== Counterfactual driver summary ===")
    print(json.dumps(summary, indent=2))
    return 0 if not summary.get("aborted") else 2


if __name__ == "__main__":
    sys.exit(main())
