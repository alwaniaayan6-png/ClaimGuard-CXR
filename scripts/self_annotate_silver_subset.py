"""Task 8a — Interactive self-annotation pass over a silver-standard subset.

Context
-------
Task 1 produced a silver-standard workbook: 200 (image, claim) rows
each graded by a 3-model ensemble (CheXbert diff, Claude Sonnet 4.5 w/
vision, MedGemma-4B), with a majority-vote label and ordinal
Krippendorff α ≥ 0.80.  Task 8 is a human-in-the-loop sanity check on
that silver pool: the user labels 100 claims sampled stratified over
the 5-class ordinal scale

    {SUPPORTED, CONTRADICTED, NOVEL_PLAUSIBLE, NOVEL_HALLUCINATED,
     UNCERTAIN}

and ``scripts/compute_user_vs_ensemble_alpha.py`` then computes
Krippendorff α between the 3 grader columns + the user column.  If
α ≥ 0.80 on the 4-coder matrix, the silver pool is declared internally
valid and goes into the paper headline.  If α < 0.80, the paper
coarsens per the plan's fallback ladder:

    (a) drop the UNCERTAIN class and recompute
    (b) coarsen to binary Supported-vs-rest and recompute
    (c) document the coarsening transparently

This script only handles part (1) — the labeling UI.  The α math and
fallback ladder live in the sibling script.

Workflow
--------
1. Read the silver workbook.
2. Stratified sample: ``n_per_class = 20`` rows per class → 100 total.
3. For each sampled row, display the OpenI image + the claim + the
   original GT radiologist report, then prompt for a single-letter
   label:

    s = SUPPORTED          c = CONTRADICTED
    p = NOVEL_PLAUSIBLE    h = NOVEL_HALLUCINATED
    u = UNCERTAIN          q = quit (save partial)

4. After every input, append the row with ``user_label`` +
   ``user_confidence`` (optional second prompt) to the output JSON
   atomically — so quitting halfway leaves a usable partial file.

Design notes
------------
* Interactive I/O is dependency-injected via callables so the unit
  tests can drive the loop without a TTY or PIL display:

      run_interactive(
          workbook=...,
          output_path=...,
          display_image_fn=lambda row: None,   # no-op in tests
          prompt_fn=lambda prompt: "s",        # canned responses
      )

* Pure helpers (``sample_stratified``, ``parse_label_key``,
  ``build_self_annotation_row``, ``coarsen_to_binary``) carry all the
  logic that matters for Task 8 reliability accounting.  They are
  unit-tested; the interactive loop is smoke-tested via the injected
  prompt_fn path.

* ``run_interactive`` uses atomic writes (tmp + rename) so a Ctrl-C
  mid-write never corrupts the output JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.compile_silver_standard_results import (  # noqa: E402
    GRADER_LABEL_KEYS,
    LABEL_TO_ORDINAL,
    VALID_LABELS,
    majority_vote,
)

logger = logging.getLogger("self_annotate_silver_subset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Single-letter → canonical label.  Chosen to be unambiguous under
# blind typing at 100-row cadence.  ``q`` is reserved for "quit /
# save partial" and is handled separately in the interactive loop.
LABEL_KEYS: dict[str, str] = {
    "s": "SUPPORTED",
    "c": "CONTRADICTED",
    "p": "NOVEL_PLAUSIBLE",
    "h": "NOVEL_HALLUCINATED",
    "u": "UNCERTAIN",
}

# Confidence ladder — same 3-point scale the Task 1 graders use, so
# the per-coder confidence columns can be aggregated uniformly.
CONFIDENCE_KEYS: dict[str, str] = {
    "l": "low",
    "m": "medium",
    "h": "high",
}

DEFAULT_N_PER_CLASS: int = 20
DEFAULT_SEED: int = 42
QUIT_KEY: str = "q"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------


def load_silver_workbook(path: str) -> list[dict[str, Any]]:
    """Read a silver workbook JSON file and return it as a list of rows.

    Raises ``FileNotFoundError`` if the path doesn't exist and
    ``ValueError`` if the top-level JSON isn't a list.  Caller is
    responsible for any schema validation past the shape check.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"silver workbook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            "silver workbook must be a JSON list of row dicts; "
            f"got {type(data).__name__}"
        )
    return data


def parse_label_key(key: str) -> Optional[str]:
    """Map a single-letter user input to a canonical ordinal label.

    Case-insensitive and whitespace-tolerant.  Returns ``None`` if the
    input doesn't match any of the 5 label keys — the caller is
    expected to re-prompt.  ``q`` is NOT returned here; the caller
    handles the quit path explicitly so it can distinguish "user wants
    to stop" from "user typed garbage."
    """
    if not isinstance(key, str):
        return None
    normalized = key.strip().lower()
    return LABEL_KEYS.get(normalized)


def parse_confidence_key(key: str) -> Optional[str]:
    """Map a single-letter confidence input to ``low/medium/high``.

    Empty / unrecognised input maps to ``"medium"`` — the same default
    used by the Task 1 grader prompts.  Returning the default instead
    of ``None`` lets the annotation loop proceed without an extra
    reprompt; the user can always go back and edit manually.
    """
    if not isinstance(key, str):
        return "medium"
    normalized = key.strip().lower()
    if not normalized:
        return "medium"
    return CONFIDENCE_KEYS.get(normalized, "medium")


def coarsen_to_binary(label: str) -> Optional[int]:
    """Collapse a 5-class label to the paper's binary Supported-vs-rest map.

    Used by the fallback α path in ``compute_user_vs_ensemble_alpha.py``
    when the full-ordinal α is below threshold.  Returns:

        * ``0`` for ``SUPPORTED``
        * ``1`` for any other valid label
        * ``None`` for unknown / missing labels (so Krippendorff treats
          the unit as missing, not as a third class)
    """
    if label == "SUPPORTED":
        return 0
    if label in set(VALID_LABELS):
        return 1
    return None


def sample_stratified(
    workbook: Sequence[dict[str, Any]],
    *,
    n_per_class: int = DEFAULT_N_PER_CLASS,
    seed: int = DEFAULT_SEED,
    class_key: str = "majority_label",
) -> list[dict[str, Any]]:
    """Deterministic stratified sample of ``n_per_class`` rows per class.

    Groups workbook rows by their ``majority_label`` field (or whatever
    ``class_key`` is set to — tests override this to an arbitrary key),
    shuffles each group under a per-class RNG seeded from ``seed`` +
    ordinal index, and takes the first ``n_per_class`` rows from each.

    If a class has fewer than ``n_per_class`` rows, we take what's
    there and log a warning; the paper math then uses a slightly
    under-balanced 4-coder matrix, which is fine because Krippendorff
    α tolerates missing units.

    Args:
        workbook: Silver-standard workbook rows.
        n_per_class: Target rows per class.  20 × 5 = 100 is the plan
            default and what the paper headline reports.
        seed: RNG seed for reproducibility.  Each class picks its
            subsample from a distinct derived RNG so adding new labels
            doesn't reshuffle existing classes' picks.
        class_key: Name of the field holding the class label.  Defaults
            to ``majority_label`` (Task 1 output).  Tests pass
            ``extracted_claim``-derived keys.

    Returns:
        A new list of row dicts.  Ordering is (class-order × sample-
        order-within-class) so the interactive loop traverses all
        labels of one class contiguously — reduces mental context-
        switching cost during the ~90-minute labeling session.
    """
    if n_per_class <= 0:
        return []

    # Group by class.  Unknown / missing labels are dropped silently.
    by_class: dict[str, list[dict[str, Any]]] = {
        lbl: [] for lbl in VALID_LABELS
    }
    for row in workbook:
        lbl = row.get(class_key, "")
        if lbl in by_class:
            by_class[lbl].append(row)

    sampled: list[dict[str, Any]] = []
    for ordinal_idx, lbl in enumerate(VALID_LABELS):
        rows = by_class[lbl]
        rng = random.Random(seed + ordinal_idx)
        rng.shuffle(rows)
        if len(rows) < n_per_class:
            logger.warning(
                "class %s has only %d rows, requested %d",
                lbl, len(rows), n_per_class,
            )
        sampled.extend(rows[:n_per_class])
    return sampled


def build_self_annotation_row(
    claim_row: dict[str, Any],
    *,
    user_label: str,
    user_confidence: str = "medium",
    annotated_at: Optional[str] = None,
) -> dict[str, Any]:
    """Wrap a silver row with a user-label overlay for the output file.

    Only the fields needed by Task 8's α computation are preserved —
    the original verifier score, evidence text, and grader columns —
    plus the new ``user_label`` / ``user_confidence`` / ``annotated_at``
    fields.  Image paths are kept so a human reviewer can re-audit any
    row without cross-referencing the source workbook.

    Raises ``ValueError`` if ``user_label`` is not in ``VALID_LABELS``
    (defensive — the interactive loop should always pass a validated
    label, but tests exercise this path directly).
    """
    if user_label not in set(VALID_LABELS):
        raise ValueError(
            f"user_label {user_label!r} not in VALID_LABELS {VALID_LABELS}"
        )
    out: dict[str, Any] = {
        "claim_id": str(claim_row.get("claim_id", "")),
        "image_file": str(claim_row.get("image_file", "")),
        "image_path": str(claim_row.get("image_path", "")),
        "extracted_claim": str(claim_row.get("extracted_claim", "")),
        "ground_truth_report": str(claim_row.get("ground_truth_report", "")),
        "majority_label": str(claim_row.get("majority_label", "")),
        "user_label": user_label,
        "user_confidence": user_confidence,
        "annotated_at": annotated_at or _now_iso(),
    }
    # Preserve every grader column so the 4-coder matrix can be built
    # from the output file alone (no need to re-load the silver
    # workbook if the user later re-runs the α computation).
    for key in GRADER_LABEL_KEYS:
        out[key] = str(claim_row.get(key, ""))
    if "verifier_score" in claim_row:
        out["verifier_score"] = claim_row["verifier_score"]
    return out


def _now_iso() -> str:
    """ISO-8601 timestamp in UTC, second precision — dependency-free."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_save_json(data: Any, path: str) -> None:
    """Write ``data`` to ``path`` atomically via tmp + rename.

    Used after every annotated row so a Ctrl-C mid-session leaves a
    consistent JSON file, not a half-written one.  The tmp filename
    lives in the same directory so the rename stays on one filesystem.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_existing_annotations(path: str) -> list[dict[str, Any]]:
    """Return the list of already-labeled rows, or ``[]`` if none exist.

    Used by the ``--resume`` path so re-running the script after a
    mid-session quit picks up where the user left off.  Invalid or
    non-list contents are treated as "start fresh" (with a warning) —
    we'd rather recover quietly than crash an annotator session.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning("existing annotation file is corrupt: %s", e)
        return []
    if not isinstance(data, list):
        logger.warning(
            "existing annotation file has wrong type %s; ignoring",
            type(data).__name__,
        )
        return []
    return data


def filter_unlabeled(
    sampled: Sequence[dict[str, Any]],
    already_labeled: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop rows whose ``claim_id`` is already in ``already_labeled``.

    Preserves original ``sampled`` ordering for the remaining rows.
    """
    labeled_ids = {
        str(r.get("claim_id", "")) for r in already_labeled
        if r.get("claim_id")
    }
    return [
        row for row in sampled
        if str(row.get("claim_id", "")) not in labeled_ids
    ]


# ---------------------------------------------------------------------------
# Interactive loop (dependency-injected)
# ---------------------------------------------------------------------------


def format_prompt(row: dict[str, Any], progress: tuple[int, int]) -> str:
    """Render the per-row prompt text.

    Factored out so tests can inspect the exact string the user sees
    without needing to run the full interactive loop.

    **Methodology critical (reviewer-flagged):** this prompt MUST NOT
    leak the silver graders' majority label or any grader column into
    the string shown to the user.  Task 8 uses the user as an
    independent fourth coder to validate the 3-grader ensemble — if
    the user sees the ensemble's answer before they label, the
    resulting Krippendorff α is a "how often does the human agree
    with what was shown to them" check, not an independent reliability
    estimate.  Reviewers who know the Krippendorff literature will
    flag this immediately.

    Args:
        row: The silver workbook row being labeled.
        progress: ``(done, total)`` — displayed as ``[N/M]``.
    """
    done, total = progress
    claim = str(row.get("extracted_claim", "")).strip()
    report = str(row.get("ground_truth_report", "")).strip()
    image = str(row.get("image_file", ""))
    return (
        f"\n=== [{done}/{total}] claim_id={row.get('claim_id', '?')} ===\n"
        f"Image: {image}\n"
        f"---- Ground-truth radiologist report ----\n{report}\n"
        f"---- Claim under review ----\n{claim}\n"
        f"\nYour label? "
        "[s]upported / [c]ontradicted / "
        "novel [p]lausible / novel [h]allucinated / "
        "[u]ncertain  ([q] = save+quit)\n> "
    )


def run_interactive(
    *,
    workbook: Sequence[dict[str, Any]],
    output_path: str,
    display_image_fn: Callable[[dict[str, Any]], None],
    prompt_fn: Callable[[str], str],
    confidence_fn: Optional[Callable[[str], str]] = None,
    save_fn: Callable[[Any, str], None] = atomic_save_json,
    quit_requested: Optional[Callable[[], bool]] = None,
) -> list[dict[str, Any]]:
    """Drive the self-annotation session.

    The three UI deps are callables so tests can drive the loop end-
    to-end without a TTY, a PIL window, or a filesystem:

        * ``display_image_fn(row)`` — shows the image.  Real usage
          calls ``PIL.Image.open(row['image_path']).show()``.  Tests
          pass a no-op.
        * ``prompt_fn(prompt_text)`` — returns the user's raw label
          input.  Real usage wraps ``input(prompt_text)``.
        * ``confidence_fn(prompt_text)`` — optional second prompt for
          the confidence tier.  Defaults to "medium" if omitted.
        * ``save_fn(data, path)`` — defaults to ``atomic_save_json``.
          Tests pass a mock to capture calls without touching disk.
        * ``quit_requested()`` — optional poll hook tests use to force
          an early quit without typing ``q``.

    Behavior:
        * Loads any existing annotations from ``output_path`` and skips
          already-labeled rows (resume support).
        * For each remaining row: display → prompt → validate → save.
        * Re-prompts on invalid input (unbounded retry loop — this is
          intentional for an interactive session).
        * ``q`` or a ``quit_requested()=True`` poll exits cleanly with
          the partial results saved.
        * Returns the full cumulative list of annotated rows
          (existing + newly added this session).

    Args:
        workbook: Rows to label, in traversal order.
        output_path: Where to save annotations (atomic writes after
            every successful label).
        display_image_fn: Image display callable.
        prompt_fn: Input prompt callable.
        confidence_fn: Optional confidence prompt callable.
        save_fn: JSON persistence callable.
        quit_requested: Optional poll hook for early-exit.

    Returns:
        The cumulative annotated rows at exit time.
    """
    all_rows: list[dict[str, Any]] = load_existing_annotations(output_path)
    remaining = filter_unlabeled(workbook, all_rows)
    total = len(workbook)

    if not remaining:
        logger.info(
            "All %d workbook rows are already annotated in %s",
            total, output_path,
        )
        return all_rows

    logger.info(
        "Starting self-annotation: %d new rows (%d already done)",
        len(remaining), len(all_rows),
    )

    n_already_done = len(all_rows)
    for i, row in enumerate(remaining):
        if quit_requested is not None and quit_requested():
            logger.info("Quit requested via poll hook — saving partial.")
            break

        try:
            display_image_fn(row)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "display_image_fn failed for row %s: %s",
                row.get("claim_id", "?"), e,
            )

        # Progress indexes against the CURRENT sample (not the
        # existing-annotations count) so resumes with a different
        # seed/n-per-class don't display N > total.  The reviewer
        # flagged the prior `done = len(all_rows) + 1` as a cosmetic
        # quirk that could exceed total on config changes.
        n_completed_this_session = i
        done = n_already_done + n_completed_this_session + 1
        if done > total:
            done = total
        prompt_text = format_prompt(row, (done, total))

        # Label prompt with retry loop for invalid input.
        while True:
            raw = prompt_fn(prompt_text)
            if raw is None:
                # Treat EOF as quit (Ctrl-D).
                logger.info("EOF on input stream — saving partial.")
                save_fn(all_rows, output_path)
                return all_rows
            normalized = str(raw).strip().lower()
            if normalized == QUIT_KEY:
                logger.info("User requested quit — saving partial.")
                save_fn(all_rows, output_path)
                return all_rows
            label = parse_label_key(normalized)
            if label is not None:
                break
            logger.info(
                "Invalid input %r — expected one of %s or %s",
                raw, sorted(LABEL_KEYS), QUIT_KEY,
            )

        # Confidence (optional second prompt).
        confidence = "medium"
        if confidence_fn is not None:
            raw_conf = confidence_fn(
                "Confidence? [l]ow / [m]edium / [h]igh (default m) > "
            )
            confidence = parse_confidence_key(raw_conf or "") or "medium"

        annotated = build_self_annotation_row(
            row,
            user_label=label,
            user_confidence=confidence,
        )
        all_rows.append(annotated)
        save_fn(all_rows, output_path)
        logger.info(
            "Labeled %s as %s (%d/%d)",
            annotated["claim_id"], label, done, total,
        )

    logger.info("Session complete. Total annotations: %d", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Real display impl (uses PIL)
# ---------------------------------------------------------------------------


def default_display_image(row: dict[str, Any]) -> None:  # pragma: no cover
    """PIL-backed image display for real interactive sessions.

    Guarded behind a lazy import so test environments without Pillow
    can still run the unit tests (which inject a no-op display).
    """
    path = row.get("image_path") or row.get("image_file")
    if not path:
        logger.warning("row has no image_path/image_file")
        return
    if not os.path.exists(path):
        logger.warning("image file not found: %s", path)
        return
    try:
        from PIL import Image  # noqa: WPS433 — lazy
        Image.open(path).show()
    except Exception as e:  # noqa: BLE001
        logger.warning("PIL failed to display %s: %s", path, e)


def default_prompt(prompt_text: str) -> str:  # pragma: no cover
    """``input()`` wrapper that tolerates EOF by returning ``"q"``."""
    try:
        return input(prompt_text)
    except EOFError:
        return QUIT_KEY


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ClaimGuard Task 8 — interactive self-annotation of 100 "
            "silver-standard claims, stratified over the 5-class "
            "ordinal scale."
        ),
    )
    parser.add_argument(
        "--workbook-path",
        default="results/silver_pilot/annotation_workbook_silver.json",
        help="Path to the Task 1 silver workbook JSON.",
    )
    parser.add_argument(
        "--output-path",
        default="results/self_annotation_100.json",
        help="Where to save user annotations (atomic write after each row).",
    )
    parser.add_argument(
        "--n-per-class",
        type=int, default=DEFAULT_N_PER_CLASS,
        help="Target samples per class (default %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--confidence", action="store_true",
        help="Prompt for per-row confidence (low/medium/high).",
    )
    parser.add_argument(
        "--no-image", action="store_true",
        help="Skip the PIL image display (useful over SSH).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)

    workbook = load_silver_workbook(args.workbook_path)
    logger.info(
        "Loaded %d rows from silver workbook at %s",
        len(workbook), args.workbook_path,
    )

    # Back-fill majority_label when absent (Task 1 stamps this, but the
    # raw grader workbook may not have run through compile_silver yet).
    for row in workbook:
        if not row.get("majority_label"):
            row["majority_label"] = majority_vote([
                row.get(k, "") for k in GRADER_LABEL_KEYS
            ])

    sampled = sample_stratified(
        workbook,
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    if not sampled:
        logger.error("No rows sampled — silver workbook may be empty.")
        return 2

    display_fn = (
        (lambda _row: None)
        if args.no_image
        else default_display_image
    )
    confidence_fn = default_prompt if args.confidence else None

    run_interactive(
        workbook=sampled,
        output_path=args.output_path,
        display_image_fn=display_fn,
        prompt_fn=default_prompt,
        confidence_fn=confidence_fn,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CONFIDENCE_KEYS",
    "DEFAULT_N_PER_CLASS",
    "DEFAULT_SEED",
    "LABEL_KEYS",
    "QUIT_KEY",
    "atomic_save_json",
    "build_self_annotation_row",
    "coarsen_to_binary",
    "filter_unlabeled",
    "format_prompt",
    "load_existing_annotations",
    "load_silver_workbook",
    "main",
    "parse_confidence_key",
    "parse_label_key",
    "run_interactive",
    "sample_stratified",
]
