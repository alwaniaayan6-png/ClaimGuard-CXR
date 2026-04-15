"""OpenI recalibrated conformal evaluation (Task 6).

Runs the custom cfBH pipeline (``ConformalClaimTriage``) on a 50/50
patient-level split of pre-scored OpenI claims and emits per-pathology
and global FDR + power tables at multiple α levels.

This is the *custom* side of the Task 6 comparison — the StratCP side
lives in ``scripts/baseline_stratcp.py`` and must be driven on the same
split (same seed, same input file) for a fair head-to-head.

Inputs
------
A scored-claims JSON file.  Each element must have at least these keys::

    {
        "claim": str,
        "pathology": str,          # CheXpert-14 label, e.g. "Pleural Effusion"
        "patient_id": str,         # used for patient-level splitting
        "label": int,              # 0 = faithful, 1 = contradicted
        "score": float,            # verifier faithfulness score (higher = more faithful)
    }

Extra keys (e.g. ``evidence``, ``negative_type``) are preserved but
ignored.

Outputs
-------
A JSON report under ``--output-json`` with the structure::

    {
        "config": { ... },
        "split": {"n_cal": int, "n_test": int, "n_patients_cal": int, "n_patients_test": int},
        "per_alpha": {
            "0.05": {
                "global": {"fdr": float, "power": float, "n_green": int, ...},
                "per_pathology": {
                    "Atelectasis": {"fdr": float, "power": float, "n_green": int, ...},
                    ...
                }
            },
            ...
        }
    }

Usage
-----
    python3 scripts/run_openi_recalibrated_eval.py \
        --scored-claims /path/to/openi_scored.json \
        --output-json results/openi_recal_custom.json \
        --alpha 0.05 0.10 0.15 0.20

A ``--dry-run`` mode builds a synthetic fixture so the script can be
smoke-tested without the Modal verifier inference path.  The dry-run
data is deterministic (seed 42) and contains 3 pathology strata with a
realistic mix of faithful/contradicted scores, which is enough to
exercise the calibration + triage code paths end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.conformal_triage import (  # noqa: E402
    ConformalClaimTriage,
    TriageResult,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScoredClaim:
    """A single OpenI (or any) claim that already has a verifier score."""

    claim: str
    pathology: str
    patient_id: str
    label: int  # 0 = faithful, 1 = contradicted
    score: float


@dataclass
class SplitArrays:
    """Column-major view of a claim split ready for the conformal API."""

    scores: np.ndarray
    labels: np.ndarray
    pathology: np.ndarray
    patient_ids: np.ndarray
    claims: list[str]

    @property
    def n(self) -> int:
        return int(self.scores.shape[0])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


REQUIRED_KEYS = ("claim", "pathology", "patient_id", "label", "score")


def load_scored_claims(path: Path) -> list[ScoredClaim]:
    """Load a scored-claims JSON file.

    Accepts either a list-of-dicts at top level, or a dict wrapping one
    under a ``"claims"`` key.  Validates that every required field is
    present; raises on the first missing or malformed row.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "claims" not in data:
            raise ValueError(
                f"Top-level dict in {path} has no 'claims' key; got "
                f"{sorted(data.keys())}"
            )
        data = data["claims"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of claim dicts")

    out: list[ScoredClaim] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a dict: {row!r}")
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            raise ValueError(
                f"Row {i} missing required fields {missing}; present: "
                f"{sorted(row.keys())}"
            )
        out.append(
            ScoredClaim(
                claim=str(row["claim"]),
                pathology=str(row["pathology"]),
                patient_id=str(row["patient_id"]),
                label=int(row["label"]),
                score=float(row["score"]),
            )
        )
    logger.info(f"Loaded {len(out)} scored claims from {path}")
    return out


def build_synthetic_scored_claims(
    n_patients: int = 1200,
    pathologies: Optional[list[str]] = None,
) -> list[ScoredClaim]:
    """Deterministic synthetic fixture for dry-run smoke testing.

    The cfBH machinery only produces non-empty rejection sets when the
    test *faithful* scores land at the upper tail of the calibration
    faithful distribution.  In real ClaimGuard evals this holds because
    the v1 verifier is highly confident on held-out faithful claims
    (peaks near 0.98) while the calibration pool still contains a
    long low-confidence tail from near-miss faithful examples.

    To match that behaviour in a pure smoke test (strictly a sanity
    check — we are not claiming this represents any real data
    distribution), we construct calibration claims with a *wide* score
    distribution and test faithful claims with a *tight high-confidence*
    distribution.  The FDR guarantee itself is only exact under true
    exchangeability; this fixture exists to exercise the plumbing.

    Structure:
      * ``n_patients`` single-claim patients split across
        ``pathologies`` (3 strata by default).
      * Calibration claims: faithful scores ~ N(0.55, 0.18) clipped to
        [0.0, 1.0] (broad), contradicted ~ N(0.15, 0.08).
      * Test claims: faithful scores ~ N(0.94, 0.03) clipped to
        [0.80, 1.00] (tight, high-confidence), contradicted
        ~ N(0.15, 0.08).
      * Patients 0 … n_patients/2 are labeled as the calibration side,
        but the actual split happens in ``patient_level_split`` using
        a deterministic shuffle — so the per-side score distribution
        is realised only on the side each patient lands.
    """
    rng = np.random.default_rng(42)
    pyrng = random.Random(42)
    if pathologies is None:
        pathologies = [
            "Pleural Effusion",
            "Cardiomegaly",
            "Pneumothorax",
        ]
    # Shuffle patient ids into two pools so we control which side of
    # the cal/test split each patient is sampled from.  We compute the
    # same shuffle the main driver uses (seed=42) so the cal/test
    # labels we draw here actually match the realised split.
    patient_ids = [f"synthP{p_idx:05d}" for p_idx in range(n_patients)]
    shuffled = list(patient_ids)
    random.Random(42).shuffle(shuffled)
    n_cal_side = n_patients // 2
    cal_side_ids = set(shuffled[:n_cal_side])

    out: list[ScoredClaim] = []
    for pid in patient_ids:
        patho = pyrng.choice(pathologies)
        is_cal_side = pid in cal_side_ids
        is_faithful = pyrng.random() < 0.7  # 70% faithful, 30% contradicted
        if is_faithful:
            label = 0
            if is_cal_side:
                score = float(rng.normal(0.55, 0.18))
            else:
                score = float(rng.normal(0.94, 0.03))
        else:
            label = 1
            score = float(rng.normal(0.15, 0.08))
        out.append(
            ScoredClaim(
                claim=f"{patho} claim for {pid}",
                pathology=patho,
                patient_id=pid,
                label=label,
                score=max(0.0, min(1.0, score)),
            )
        )
    logger.info(
        f"Built synthetic fixture: {len(out)} claims, "
        f"{n_patients} patients, {len(pathologies)} strata"
    )
    return out


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------


def patient_level_split(
    claims: list[ScoredClaim],
    seed: int = 42,
    cal_fraction: float = 0.5,
) -> tuple[list[ScoredClaim], list[ScoredClaim]]:
    """Split claims into cal/test by patient id.

    All claims from a given patient end up entirely in one side, so no
    cross-contamination is possible.  Splitting uses
    ``random.Random(seed).shuffle`` on the sorted unique patient list so
    the split is deterministic across runs on the same input file.
    """
    patients = sorted({c.patient_id for c in claims})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_cal = int(round(len(patients) * cal_fraction))
    cal_patients = set(patients[:n_cal])
    test_patients = set(patients[n_cal:])

    cal_claims = [c for c in claims if c.patient_id in cal_patients]
    test_claims = [c for c in claims if c.patient_id in test_patients]
    logger.info(
        f"Patient-level split (seed={seed}, cal_fraction={cal_fraction}): "
        f"{len(cal_patients)} cal patients / {len(test_patients)} test patients; "
        f"{len(cal_claims)} cal claims / {len(test_claims)} test claims"
    )
    return cal_claims, test_claims


def claims_to_arrays(claims: list[ScoredClaim]) -> SplitArrays:
    """Flatten a list of ``ScoredClaim`` into column-major numpy arrays."""
    if not claims:
        return SplitArrays(
            scores=np.empty(0, dtype=np.float64),
            labels=np.empty(0, dtype=np.int64),
            pathology=np.empty(0, dtype=object),
            patient_ids=np.empty(0, dtype=object),
            claims=[],
        )
    return SplitArrays(
        scores=np.asarray([c.score for c in claims], dtype=np.float64),
        labels=np.asarray([c.label for c in claims], dtype=np.int64),
        pathology=np.asarray([c.pathology for c in claims], dtype=object),
        patient_ids=np.asarray([c.patient_id for c in claims], dtype=object),
        claims=[c.claim for c in claims],
    )


# ---------------------------------------------------------------------------
# FDR + power reporting
# ---------------------------------------------------------------------------


def _fdr_power_block(
    triage_results: list[TriageResult],
    labels: np.ndarray,
) -> dict:
    """Core FDR + power stats for a block of triage results.

    FDR = #{green ∧ label==1} / max(#{green}, 1)
    Power = #{green ∧ label==0} / max(#{label==0}, 1)

    Under the inverted-calibration convention used by
    ``ConformalClaimTriage`` (calibrate on faithful, test upper-tail,
    accept → green), ``green`` means "accepted as faithful".  A
    *contradicted* claim labelled green is a false discovery of the
    faithful null.

    NB: This matches the FDR definition in ``modal_run_evaluation.py``
    and ``inference.conformal_triage.compute_fdr``.
    """
    n = len(triage_results)
    if n == 0:
        return {
            "fdr": 0.0,
            "power": 0.0,
            "n_green": 0,
            "n_total": 0,
            "n_false_discoveries": 0,
            "n_true_discoveries": 0,
            "n_faithful": int((labels == 0).sum()),
            "n_contradicted": int((labels == 1).sum()),
        }
    green_mask = np.array([r.is_accepted for r in triage_results], dtype=bool)
    n_green = int(green_mask.sum())
    green_labels = labels[green_mask]
    n_false = int((green_labels == 1).sum())
    n_true = int((green_labels == 0).sum())
    n_faithful = int((labels == 0).sum())
    fdr = n_false / max(n_green, 1)
    # Power = fraction of *faithful* claims correctly accepted as green.
    power = n_true / max(n_faithful, 1) if n_faithful > 0 else 0.0
    return {
        "fdr": float(fdr),
        "power": float(power),
        "n_green": n_green,
        "n_total": int(n),
        "n_false_discoveries": n_false,
        "n_true_discoveries": n_true,
        "n_faithful": n_faithful,
        "n_contradicted": int((labels == 1).sum()),
    }


def run_single_alpha(
    cal: SplitArrays,
    test: SplitArrays,
    alpha: float,
    min_group_size: int,
    seed: int,
) -> dict:
    """Calibrate + triage at a single α level, return the report block."""
    triager = ConformalClaimTriage(
        alpha=alpha,
        min_group_size=min_group_size,
        seed=seed,
    )
    triager.calibrate(
        cal.scores,
        cal.labels,
        cal.pathology,
        cal.patient_ids,
    )
    results = triager.triage(
        test.scores,
        test.pathology,
        claim_texts=test.claims,
    )

    # Global block
    global_block = _fdr_power_block(results, test.labels)

    # Per-pathology block
    per_patho: dict[str, dict] = {}
    for patho in sorted(set(test.pathology.tolist())):
        mask = test.pathology == patho
        subset = [r for r, m in zip(results, mask) if m]
        per_patho[patho] = _fdr_power_block(subset, test.labels[mask])

    return {
        "alpha": float(alpha),
        "global": global_block,
        "per_pathology": per_patho,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_evaluation(
    claims: list[ScoredClaim],
    alphas: list[float],
    seed: int,
    cal_fraction: float,
    min_group_size: int,
) -> dict:
    """End-to-end driver: split, calibrate, triage at each α, return report."""
    cal_claims, test_claims = patient_level_split(
        claims, seed=seed, cal_fraction=cal_fraction
    )
    cal = claims_to_arrays(cal_claims)
    test = claims_to_arrays(test_claims)

    per_alpha: dict[str, dict] = {}
    for alpha in alphas:
        logger.info(f"=== α = {alpha} ===")
        block = run_single_alpha(
            cal=cal,
            test=test,
            alpha=alpha,
            min_group_size=min_group_size,
            seed=seed,
        )
        per_alpha[f"{alpha:.4f}".rstrip("0").rstrip(".")] = block
        logger.info(
            f"  global FDR={block['global']['fdr']:.4f}  "
            f"power={block['global']['power']:.4f}  "
            f"n_green={block['global']['n_green']}/{block['global']['n_total']}"
        )

    return {
        "config": {
            "seed": seed,
            "cal_fraction": cal_fraction,
            "min_group_size": min_group_size,
            "alphas": list(alphas),
            "n_total_claims": len(claims),
        },
        "split": {
            "n_cal": cal.n,
            "n_test": test.n,
            "n_patients_cal": int(len(set(cal.patient_ids.tolist()))),
            "n_patients_test": int(len(set(test.patient_ids.tolist()))),
        },
        "per_alpha": per_alpha,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenI recalibrated conformal evaluation (Task 6)"
    )
    parser.add_argument(
        "--scored-claims",
        type=Path,
        default=None,
        help="Path to scored-claims JSON (required unless --dry-run).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the aggregated FDR/power JSON report.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
        help="Target FDR levels (space-separated).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for patient-level split (default 42).",
    )
    parser.add_argument(
        "--cal-fraction",
        type=float,
        default=0.5,
        help="Fraction of patients assigned to calibration (default 0.5).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=50,
        help="Minimum calibration claims per pathology group; smaller "
             "groups get pooled into Rare/Other (default 50, lower than "
             "the v1 200 because OpenI is smaller).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic synthetic fixture instead of a real "
             "scored-claims file.  Useful for CI smoke-testing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        claims = build_synthetic_scored_claims(n_patients=1500)
    else:
        if args.scored_claims is None:
            parser.error(
                "--scored-claims is required unless --dry-run is set"
            )
        claims = load_scored_claims(args.scored_claims)

    report = run_evaluation(
        claims=claims,
        alphas=sorted(args.alpha),
        seed=args.seed,
        cal_fraction=args.cal_fraction,
        min_group_size=args.min_group_size,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote report → {args.output_json}")

    # Summary print
    print("\n=== Summary ===")
    print(f"Split: {report['split']}")
    for alpha_str, block in report["per_alpha"].items():
        g = block["global"]
        print(
            f"  α={alpha_str}: global FDR={g['fdr']:.4f}  "
            f"power={g['power']:.4f}  "
            f"n_green={g['n_green']}/{g['n_total']}"
        )


if __name__ == "__main__":
    main()
