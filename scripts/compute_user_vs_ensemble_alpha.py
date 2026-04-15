"""Task 8b — Compute Krippendorff α between the user and the silver graders.

Context
-------
After the user annotates 100 claims via
``scripts/self_annotate_silver_subset.py``, this script computes the
4-coder reliability between the user column and the 3 silver graders
(CheXbert diff, Claude Sonnet 4.5 vision, MedGemma-4B) on the matched
subset of rows.

The plan's headline ask (from ``rosy-discovering-bubble.md``):

    > compute Krippendorff's α (ordinal, not Cohen's κ) with the 3
    > ensemble graders.  Target α ≥ 0.80.
    >
    > If α < 0.80 on first pass: (a) drop UNCERTAIN and recompute;
    > (b) coarsen to binary Supported-vs-rest and recompute; (c)
    > document the coarsening transparently in the paper.

This script implements that fallback ladder explicitly:

    1. **full_ordinal**   — all 5 labels, ordinal metric
    2. **drop_uncertain** — drop units where any coder said UNCERTAIN,
       recompute on the remaining 4-class ordinal scale
    3. **binary_coarsen** — collapse to SUPPORTED / not-SUPPORTED,
       nominal metric (binary is already nominal = ordinal for k=2)

Each rung of the ladder is reported with a point estimate and a 95%
bootstrap CI (1000 resamples over units, seed=42, matching the Task 1
methodology).  The first rung to clear α ≥ 0.80 is recorded as
``passing_rung``; if none clear, the paper must transparently
document that even the binary coarsening fell short.

Output
------
Writes ``results/self_annotation_alpha.json`` (configurable) with:

    {
      "n_user_annotated": 100,
      "n_matched_with_silver": 98,
      "rungs": {
        "full_ordinal":   {"alpha": ..., "ci_low": ..., "ci_high": ...,
                            "n_units": ..., "passes": ...},
        "drop_uncertain": { ... },
        "binary_coarsen": { ... }
      },
      "passing_rung": "full_ordinal" | "drop_uncertain" |
                      "binary_coarsen" | null,
      "min_alpha_target": 0.80,
      "silver_workbook_path": "...",
      "self_annotation_path": "..."
    }

Plus a human-readable summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from evaluation.krippendorff_alpha import (  # noqa: E402
    alpha_with_bootstrap_ci,
)
from scripts.compile_silver_standard_results import (  # noqa: E402
    GRADER_LABEL_KEYS,
    LABEL_TO_ORDINAL,
    VALID_LABELS,
)
from scripts.self_annotate_silver_subset import (  # noqa: E402
    coarsen_to_binary,
    load_existing_annotations,
    load_silver_workbook,
)

logger = logging.getLogger("compute_user_vs_ensemble_alpha")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_ALPHA: float = 0.80
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_BOOTSTRAP_SEED: int = 42
DEFAULT_CI: float = 0.95

USER_COLUMN_KEY: str = "user_label"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def load_self_annotation(path: str) -> list[dict[str, Any]]:
    """Thin wrapper around ``load_existing_annotations`` that raises on empty.

    The compute script expects a non-empty annotation file; if the user
    hasn't labeled anything yet, we surface that immediately rather
    than producing a degenerate α report.
    """
    rows = load_existing_annotations(path)
    if not rows:
        raise ValueError(
            f"self-annotation file {path!r} is empty or missing — run "
            "scripts/self_annotate_silver_subset.py first."
        )
    return rows


def align_rows(
    silver: Sequence[dict[str, Any]],
    user: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around ``align_rows_with_drops``.

    Returns only the merged rows — used by call sites that don't care
    about drop-count diagnostics.  New code should prefer
    ``align_rows_with_drops`` so the per-cause drop counts surface
    in the report JSON (reviewer-flagged: silently-dropped "silver row
    has no valid grader labels" units bias α upward because they are
    likely the hardest claims).
    """
    merged, _drops = align_rows_with_drops(silver, user)
    return merged


def align_rows_with_drops(
    silver: Sequence[dict[str, Any]],
    user: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join silver rows with user labels on ``claim_id``.

    Returns ``(merged, drops)`` where ``drops`` breaks out the three
    silent-drop causes so ``compute_fallback_ladder``'s report can
    surface them explicitly.  The cause keys are:

        * ``"missing_claim_id"``  — user row has empty / missing
          ``claim_id``
        * ``"no_silver_match"``   — user row's claim id is not in
          the silver workbook
        * ``"invalid_user_label"`` — user label not in VALID_LABELS
          (e.g. a hand-edited typo in the annotation JSON)
        * ``"silver_no_valid_graders"`` — silver row exists but every
          grader column is empty / invalid.  This is the bias-worrying
          case: we drop these units even though the user labeled
          them, and they are likely the hardest claims (where every
          silver grader failed).

    Reviewer note (paper): case #4 is non-uniform on label — it
    biases α upward because the dropped rows are likely low-α units.
    Report it separately and caveat in the paper.

    A row survives all four filters only if:

        * it has a non-empty ``claim_id`` in both sources,
        * the user row's ``user_label`` is in ``VALID_LABELS``,
        * at least one silver grader column is in ``VALID_LABELS``.
    """
    silver_by_id: dict[str, dict[str, Any]] = {}
    for row in silver:
        cid = str(row.get("claim_id", ""))
        if cid:
            silver_by_id[cid] = row

    drops = {
        "missing_claim_id": 0,
        "no_silver_match": 0,
        "invalid_user_label": 0,
        "silver_no_valid_graders": 0,
    }
    merged: list[dict[str, Any]] = []
    for u in user:
        cid = str(u.get("claim_id", ""))
        if not cid:
            drops["missing_claim_id"] += 1
            continue
        if cid not in silver_by_id:
            drops["no_silver_match"] += 1
            logger.warning(
                "self-annotation row %s has no silver match; dropping",
                cid,
            )
            continue
        user_label = str(u.get(USER_COLUMN_KEY, "")).strip()
        if user_label not in set(VALID_LABELS):
            drops["invalid_user_label"] += 1
            logger.warning(
                "self-annotation row %s has invalid user_label %r; "
                "dropping", cid, user_label,
            )
            continue
        silver_row = silver_by_id[cid]
        grader_labels = [
            str(silver_row.get(k, "")).strip() for k in GRADER_LABEL_KEYS
        ]
        if not any(lbl in set(VALID_LABELS) for lbl in grader_labels):
            drops["silver_no_valid_graders"] += 1
            logger.warning(
                "silver row %s has no valid grader labels; dropping",
                cid,
            )
            continue
        merged.append({
            "claim_id": cid,
            "user_label": user_label,
            **{k: silver_row.get(k, "") for k in GRADER_LABEL_KEYS},
        })
    return merged, drops


def build_4coder_matrix(
    merged: Sequence[dict[str, Any]],
    *,
    coder_keys: Sequence[str] = (*GRADER_LABEL_KEYS, USER_COLUMN_KEY),
) -> np.ndarray:
    """Build a ``(4, n_units)`` ordinal float matrix for Krippendorff α.

    Missing / invalid labels become ``np.nan`` — the Krippendorff
    implementation treats those as missing codings and drops units
    with fewer than 2 valid coders automatically.

    Coder order: [chexbert, claude, medgemma, user] (matches the order
    in ``coder_keys``).  The fourth row (user) is what this script
    exists to add on top of Task 1's 3-row matrix.
    """
    n_coders = len(coder_keys)
    n_units = len(merged)
    matrix = np.full((n_coders, n_units), np.nan, dtype=np.float64)
    for j, row in enumerate(merged):
        for i, key in enumerate(coder_keys):
            label = row.get(key, "")
            ordinal = LABEL_TO_ORDINAL.get(label)
            if ordinal is not None:
                matrix[i, j] = float(ordinal)
    return matrix


def drop_uncertain_units(
    merged: Sequence[dict[str, Any]],
    *,
    coder_keys: Sequence[str] = (*GRADER_LABEL_KEYS, USER_COLUMN_KEY),
) -> list[dict[str, Any]]:
    """Drop any unit where at least one coder voted ``UNCERTAIN`` (strict).

    This is the STRICT unit-level interpretation of the plan's
    "drop UNCERTAIN and recompute" instruction.  It shrinks the
    sample fastest but is not the canonical Krippendorff move —
    see ``drop_uncertain_values`` for the canonical value-level
    nan-ification that is the rung-2 default in
    ``compute_fallback_ladder``.

    Exposed because some methodologists prefer the strict
    interpretation (it removes any possibility that UNCERTAIN acts
    as a near-miss connecting a coder's label to its ordinal
    neighbours).  Ship both and let the paper cite whichever is
    defended.
    """
    out: list[dict[str, Any]] = []
    for row in merged:
        if any(
            row.get(k, "") == "UNCERTAIN" for k in coder_keys
        ):
            continue
        out.append(row)
    return out


def drop_uncertain_values(
    merged: Sequence[dict[str, Any]],
    *,
    coder_keys: Sequence[str] = (*GRADER_LABEL_KEYS, USER_COLUMN_KEY),
) -> list[dict[str, Any]]:
    """Canonical Krippendorff drop: nan-ify every ``UNCERTAIN`` cell.

    The reviewer flagged that the plan's "drop UNCERTAIN" is
    ambiguous between unit-level and value-level.  Krippendorff (2018)
    §3.4 prefers value-level: set every UNCERTAIN coding to nan and
    let α's missing-data path handle units where the remaining
    coders still give ≥ 2 valid labels.  This is softer than the
    unit-level drop — units where only one coder said UNCERTAIN
    still contribute to α via their 3 remaining coders.

    Implementation: returns a new list of row dicts with every
    ``UNCERTAIN`` value in ``coder_keys`` replaced by the empty
    string.  The matrix builders already turn empty strings into
    ``np.nan`` via ``LABEL_TO_ORDINAL.get(label) is None``, so this
    is the minimal wire-level change.
    """
    out: list[dict[str, Any]] = []
    for row in merged:
        new_row = dict(row)
        for k in coder_keys:
            if new_row.get(k, "") == "UNCERTAIN":
                new_row[k] = ""
        out.append(new_row)
    return out


def build_binary_matrix(
    merged: Sequence[dict[str, Any]],
    *,
    coder_keys: Sequence[str] = (*GRADER_LABEL_KEYS, USER_COLUMN_KEY),
) -> np.ndarray:
    """Build a ``(n_coders, n_units)`` binary float matrix.

    SUPPORTED → 0, any other valid label → 1, unknown → ``np.nan``.

    With only two values the nominal and ordinal δ² metrics differ
    only by a constant scale factor, and Krippendorff's α =
    1 − Do/De is invariant under uniform scaling of δ² (both Do and
    De get multiplied by the same constant).  So the rung-3 α is
    identical whether we pass ``level="nominal"`` or ``level="ordinal"``
    to ``alpha_with_bootstrap_ci``; we use ``"nominal"`` in the
    report for clarity because "binary → nominal" is the standard
    reader expectation, and an earlier version of this comment got
    the proof wrong by claiming δ² itself was equal across metrics.
    """
    n_coders = len(coder_keys)
    n_units = len(merged)
    matrix = np.full((n_coders, n_units), np.nan, dtype=np.float64)
    for j, row in enumerate(merged):
        for i, key in enumerate(coder_keys):
            label = str(row.get(key, ""))
            binary = coarsen_to_binary(label)
            if binary is not None:
                matrix[i, j] = float(binary)
    return matrix


def _alpha_report(
    matrix: np.ndarray,
    *,
    min_alpha: float,
    n_bootstrap: int,
    seed: int,
    ci: float,
    level: str = "ordinal",
) -> dict[str, Any]:
    """Run ``alpha_with_bootstrap_ci`` and package the result.

    Degenerate cases (< 2 units, no valid codings) return NaN values
    with ``passes=False`` so the caller can still serialize the row
    and note that the rung was not computable.
    """
    n_coders, n_units = matrix.shape
    if n_units < 2:
        return {
            "alpha": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_units": int(n_units),
            "n_coders": int(n_coders),
            "level": level,
            "passes": False,
            "note": "n_units < 2 — Krippendorff α is undefined",
        }
    try:
        point, lo, hi = alpha_with_bootstrap_ci(
            matrix,
            level=level,  # type: ignore[arg-type]
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci=ci,
        )
    except ValueError as e:
        return {
            "alpha": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_units": int(n_units),
            "n_coders": int(n_coders),
            "level": level,
            "passes": False,
            "note": f"alpha raised ValueError: {e}",
        }
    passes = (
        not math.isnan(point)
        and point >= min_alpha
    )
    return {
        "alpha": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_units": int(n_units),
        "n_coders": int(n_coders),
        "level": level,
        "passes": bool(passes),
    }


def compute_fallback_ladder(
    merged: Sequence[dict[str, Any]],
    *,
    min_alpha: float = DEFAULT_MIN_ALPHA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_CI,
) -> dict[str, Any]:
    """Evaluate all three fallback rungs and record the first that passes.

    Rungs (plan order):

        1. ``full_ordinal``    — all 5 labels, ordinal metric
        2. ``drop_uncertain``  — value-level nan-ification of every
           UNCERTAIN coding (canonical Krippendorff 2018 interpretation
           of "drop UNCERTAIN"; softer than the strict unit-level
           drop — a unit keeps contributing to α via its remaining
           valid coders).  Use ``drop_uncertain_units`` directly if
           you need the stricter unit-level interpretation.
        3. ``binary_coarsen`` — SUPPORTED vs not-SUPPORTED, nominal
           metric (binary → nominal is the standard reader
           expectation; k=2 nominal α equals ordinal α under α's
           δ²-scale invariance).

    Returns a dict with a ``rungs`` sub-dict keyed by rung name and a
    top-level ``passing_rung`` string (or ``None`` if all rungs
    failed).  Callers serialize this dict directly to JSON.
    """
    full_matrix = build_4coder_matrix(merged)
    dropped_values = drop_uncertain_values(merged)
    drop_matrix = build_4coder_matrix(dropped_values)
    binary_matrix = build_binary_matrix(merged)

    rungs: dict[str, Any] = {
        "full_ordinal": _alpha_report(
            full_matrix,
            min_alpha=min_alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci=ci,
            level="ordinal",
        ),
        "drop_uncertain": _alpha_report(
            drop_matrix,
            min_alpha=min_alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci=ci,
            level="ordinal",
        ),
        "binary_coarsen": _alpha_report(
            binary_matrix,
            min_alpha=min_alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci=ci,
            level="nominal",
        ),
    }

    passing_rung: Optional[str] = None
    for name in ("full_ordinal", "drop_uncertain", "binary_coarsen"):
        if rungs[name].get("passes"):
            passing_rung = name
            break

    return {
        "rungs": rungs,
        "passing_rung": passing_rung,
        "min_alpha_target": float(min_alpha),
    }


def format_summary(report: dict[str, Any]) -> str:
    """Render a short human-readable summary of a ladder report.

    Used by ``main()`` to print to stdout so the user sees the result
    immediately after the JSON is written.  Kept pure so tests can
    assert against the exact string.
    """
    lines: list[str] = [
        "",
        "=== Task 8 — User vs Silver Ensemble Krippendorff α ===",
        f"n_user_annotated     : {report.get('n_user_annotated', '?')}",
        f"n_matched_with_silver: {report.get('n_matched_with_silver', '?')}",
        f"min_alpha_target     : "
        f"{report.get('min_alpha_target', DEFAULT_MIN_ALPHA):.2f}",
        "",
    ]
    rungs = report.get("rungs", {})
    for name in ("full_ordinal", "drop_uncertain", "binary_coarsen"):
        rung = rungs.get(name, {})
        a = rung.get("alpha", float("nan"))
        lo = rung.get("ci_low", float("nan"))
        hi = rung.get("ci_high", float("nan"))
        n = rung.get("n_units", 0)
        passes = rung.get("passes", False)
        marker = "PASS" if passes else "fail"
        lines.append(
            f"  [{marker}] {name:<16}  "
            f"α={a:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  n={n}"
        )
    passing = report.get("passing_rung")
    if passing is None:
        lines.append("")
        lines.append(
            "RESULT: all rungs failed — silver pool reliability is "
            "below α=0.80 even under binary coarsening.  The paper "
            "must document this transparently."
        )
    else:
        lines.append("")
        lines.append(
            f"RESULT: passing rung = {passing}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ClaimGuard Task 8b — compute Krippendorff α between the "
            "user and the 3 silver graders, with the plan's fallback "
            "ladder (full → drop UNCERTAIN → binary coarsen)."
        ),
    )
    parser.add_argument(
        "--silver-workbook-path",
        default="results/silver_pilot/annotation_workbook_silver.json",
    )
    parser.add_argument(
        "--self-annotation-path",
        default="results/self_annotation_100.json",
    )
    parser.add_argument(
        "--output-path",
        default="results/self_annotation_alpha.json",
    )
    parser.add_argument(
        "--min-alpha", type=float, default=DEFAULT_MIN_ALPHA,
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--ci", type=float, default=DEFAULT_CI,
    )
    parser.add_argument(
        "--exit-nonzero-on-fail", action="store_true",
        help=(
            "Return exit code 2 if no rung passes — useful for CI "
            "pipelines that gate downstream analysis on α."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)

    silver = load_silver_workbook(args.silver_workbook_path)
    user = load_self_annotation(args.self_annotation_path)
    logger.info(
        "Loaded %d silver rows and %d user annotations",
        len(silver), len(user),
    )

    merged, drops = align_rows_with_drops(silver, user)
    logger.info(
        "Aligned %d of %d user rows to silver; drops=%s",
        len(merged), len(user), drops,
    )

    ladder = compute_fallback_ladder(
        merged,
        min_alpha=args.min_alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ci=args.ci,
    )
    report: dict[str, Any] = {
        "n_user_annotated": len(user),
        "n_matched_with_silver": len(merged),
        "drop_counts": drops,
        "silver_workbook_path": str(args.silver_workbook_path),
        "self_annotation_path": str(args.self_annotation_path),
        **ladder,
    }

    out_parent = os.path.dirname(os.path.abspath(args.output_path)) or "."
    os.makedirs(out_parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.output_path)

    print(format_summary(report))

    if args.exit_nonzero_on_fail and report.get("passing_rung") is None:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CI",
    "DEFAULT_MIN_ALPHA",
    "DEFAULT_N_BOOTSTRAP",
    "USER_COLUMN_KEY",
    "align_rows",
    "align_rows_with_drops",
    "build_4coder_matrix",
    "build_binary_matrix",
    "compute_fallback_ladder",
    "drop_uncertain_units",
    "drop_uncertain_values",
    "format_summary",
    "load_self_annotation",
    "main",
]
