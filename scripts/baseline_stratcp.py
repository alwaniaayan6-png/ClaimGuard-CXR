"""StratCP baseline driver (Task 6).

Loads the *same* scored-claims file and *same* patient-level split used
by ``scripts/run_openi_recalibrated_eval.py``, runs the StratCP
predictor from ``inference.stratcp`` on it, and writes a comparison
CSV (and a machine-readable JSON) against the custom cfBH pipeline.

Why a separate driver?
----------------------
StratCP controls *per-stratum marginal miscoverage*, not BH-style FDR.
Running both procedures on the same split surfaces the mismatch the
paper wants to point out:

    - cfBH is certified to keep FDR ≤ α at every α level.
    - StratCP is certified to keep per-stratum miscoverage ≤ α, but
      that does NOT translate into FDR control on a mixture test set
      — the empirical FDR can blow well past α, especially in strata
      with rare contradicted examples.

The script emits a head-to-head table so the comparison is mechanical.

Score direction
---------------
The ClaimGuard verifier outputs a *faithful* probability in [0, 1]
(higher = more faithful).  StratCP, as implemented in
``inference/stratcp.py``, is an upper-tail rejection test: it treats
the calibration pool as the *null* distribution and rejects any test
point whose score exceeds the stratified upper-tail quantile.

So to use it as a contradicted-class *detector* we:

1. Define ``suspicion = 1.0 - faithful_score``.  Higher suspicion ⇒
   more likely contradicted.
2. Calibrate on **faithful** (``label == 0``) claims' suspicion
   values.  These are the null-class reference distribution.
3. Per-stratum quantile ``Q_s`` is then the upper tail of the
   faithful suspicion distribution — roughly the top-α most
   suspicious faithful claims in that stratum.
4. Reject (flag as contradicted) iff test suspicion ≥ Q_s.

This polarity is the correct one: the per-stratum *false rejection*
rate on truly-faithful test claims is bounded by α (split-CP Thm 1),
and genuinely contradicted claims have suspicion well above Q_s so
they are flagged with high probability (high power).

The intuitive pitfall is to calibrate on contradicted-label claims;
that gives Q_s as the upper tail of contradicted suspicion (above
most contradicted claims), so the flagging rate collapses to ~α of
the contradicted population.  We do NOT do that here.

Output
------
Two artifacts:

    <output-dir>/stratcp_vs_cfbh.csv
    <output-dir>/stratcp_vs_cfbh.json

CSV columns: ``alpha, stratum, method, n_test, n_rejected, fdr,
power, rejection_rate`` with rows for both methods (if a cfBH result
file was provided) or StratCP only.

Usage
-----
    python3 scripts/baseline_stratcp.py \\
        --scored-claims /path/to/openi_scored.json \\
        --cfbh-json results/openi_recal_custom.json \\
        --output-dir results/stratcp \\
        --alpha 0.05 0.10 0.15 0.20

    # Smoke test without a real scored file:
    python3 scripts/baseline_stratcp.py --dry-run --output-dir /tmp/scp
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the split + loader + fixture from the recalibrated driver so
# both scripts are *guaranteed* to see the same cal/test partitions.
from scripts.run_openi_recalibrated_eval import (  # noqa: E402
    ScoredClaim,
    SplitArrays,
    build_synthetic_scored_claims,
    claims_to_arrays,
    load_scored_claims,
    patient_level_split,
)
from inference.stratcp import StratCPPredictor  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Core StratCP driver
# ---------------------------------------------------------------------------


def suspicion(scores: np.ndarray) -> np.ndarray:
    """Flip faithful → suspicion (higher = more contradicted)."""
    return 1.0 - np.asarray(scores, dtype=np.float64)


def _fdr_power_from_mask(
    rejected: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """FDR + power stats for a boolean rejection mask.

    FDR is defined the same way as the cfBH driver for direct
    comparability:
        FDR = #{rejected ∧ label==0} / max(#{rejected}, 1)
    i.e. "rejected claim is actually faithful" (a false alarm).
    Power:
        power = #{rejected ∧ label==1} / max(#{label==1}, 1)
    i.e. "contradicted claim correctly rejected".

    Note the directionality: StratCP rejects when ``suspicion ≥ Q_s``,
    which semantically means "flag as contradicted".  So a rejection
    is a *positive detection* in the contradicted-is-positive frame.
    """
    n = int(rejected.shape[0])
    if n == 0:
        return {
            "n_total": 0,
            "n_rejected": 0,
            "n_false_rejections": 0,
            "n_true_rejections": 0,
            "fdr": 0.0,
            "power": 0.0,
            "rejection_rate": 0.0,
            "n_faithful": 0,
            "n_contradicted": 0,
        }
    labels = np.asarray(labels)
    n_rej = int(rejected.sum())
    rej_labels = labels[rejected]
    n_false = int((rej_labels == 0).sum())
    n_true = int((rej_labels == 1).sum())
    n_contradicted = int((labels == 1).sum())
    return {
        "n_total": n,
        "n_rejected": n_rej,
        "n_false_rejections": n_false,
        "n_true_rejections": n_true,
        "fdr": n_false / max(n_rej, 1),
        "power": (
            n_true / n_contradicted if n_contradicted > 0 else 0.0
        ),
        "rejection_rate": n_rej / n,
        "n_faithful": int((labels == 0).sum()),
        "n_contradicted": n_contradicted,
    }


def run_stratcp_at_alpha(
    cal: SplitArrays,
    test: SplitArrays,
    alpha: float,
    min_stratum_size: int,
) -> dict:
    """Fit + predict StratCP at a single α; return per-stratum + global.

    The predictor is calibrated on the **faithful** claims in ``cal``
    (label == 0), using ``suspicion = 1 - faithful_score``.  Faithful
    claims form the null distribution whose upper tail defines the
    rejection threshold; split-CP Theorem 1 bounds the *false
    rejection* rate on exchangeable faithful test claims at α per
    stratum.  Genuinely contradicted test claims have suspicion
    well above the faithful upper tail so they are flagged with
    high probability.
    """
    cal_faithful_mask = cal.labels == 0
    n_cal_faithful = int(cal_faithful_mask.sum())
    if n_cal_faithful == 0:
        raise ValueError(
            "StratCP needs at least one faithful calibration claim"
        )

    cal_sus = suspicion(cal.scores[cal_faithful_mask])
    cal_strata = cal.pathology[cal_faithful_mask]
    predictor = StratCPPredictor(
        alpha=alpha,
        min_stratum_size=min_stratum_size,
    )
    predictor.calibrate(cal_sus, cal_strata)

    test_sus = suspicion(test.scores)
    test_strata = test.pathology
    rejected = predictor.predict(test_sus, test_strata)

    # Global block — treat the whole test set as one pool.
    global_block = _fdr_power_from_mask(rejected, test.labels)
    global_block["alpha"] = float(alpha)
    global_block["method"] = "stratcp"
    global_block["stratum"] = "__global__"

    # Per-stratum block.
    per_stratum: dict[str, dict] = {}
    for patho in sorted(set(test_strata.tolist())):
        mask = test_strata == patho
        block = _fdr_power_from_mask(rejected[mask], test.labels[mask])
        block["alpha"] = float(alpha)
        block["method"] = "stratcp"
        block["stratum"] = patho
        per_stratum[patho] = block

    return {
        "alpha": float(alpha),
        "global": global_block,
        "per_stratum": per_stratum,
        "stratum_sizes": predictor.stratum_sizes(),
        "per_stratum_thresholds": predictor.per_stratum_thresholds(),
        "pooled_threshold": predictor.pooled_threshold(),
    }


# ---------------------------------------------------------------------------
# cfBH sidecar loader (for comparison rows)
# ---------------------------------------------------------------------------


def load_cfbh_report(path: Optional[Path]) -> Optional[dict]:
    """Load a cfBH report JSON from ``run_openi_recalibrated_eval``."""
    if path is None:
        return None
    if not path.exists():
        logger.warning(
            f"cfBH report {path} not found; emitting StratCP-only CSV"
        )
        return None
    return json.loads(path.read_text())


def cfbh_rows_for_csv(cfbh: dict) -> list[dict]:
    """Flatten the cfBH JSON into CSV rows with method='cfbh'."""
    rows: list[dict] = []
    for alpha_str, block in cfbh.get("per_alpha", {}).items():
        alpha = float(alpha_str)
        g = block["global"]
        rows.append({
            "alpha": f"{alpha:.4f}",
            "stratum": "__global__",
            "method": "cfbh",
            "n_test": g["n_total"],
            "n_rejected": g["n_green"],
            "fdr": f"{g['fdr']:.6f}",
            "power": f"{g['power']:.6f}",
            "rejection_rate": (
                f"{g['n_green'] / max(g['n_total'], 1):.6f}"
            ),
        })
        for patho, p in block.get("per_pathology", {}).items():
            rows.append({
                "alpha": f"{alpha:.4f}",
                "stratum": patho,
                "method": "cfbh",
                "n_test": p["n_total"],
                "n_rejected": p["n_green"],
                "fdr": f"{p['fdr']:.6f}",
                "power": f"{p['power']:.6f}",
                "rejection_rate": (
                    f"{p['n_green'] / max(p['n_total'], 1):.6f}"
                ),
            })
    return rows


def stratcp_rows_for_csv(stratcp: dict) -> list[dict]:
    """Flatten the StratCP results dict into CSV rows."""
    rows: list[dict] = []
    for alpha_str, block in stratcp["per_alpha"].items():
        alpha = float(alpha_str)
        g = block["global"]
        rows.append({
            "alpha": f"{alpha:.4f}",
            "stratum": "__global__",
            "method": "stratcp",
            "n_test": g["n_total"],
            "n_rejected": g["n_rejected"],
            "fdr": f"{g['fdr']:.6f}",
            "power": f"{g['power']:.6f}",
            "rejection_rate": f"{g['rejection_rate']:.6f}",
        })
        for patho, p in block["per_stratum"].items():
            rows.append({
                "alpha": f"{alpha:.4f}",
                "stratum": patho,
                "method": "stratcp",
                "n_test": p["n_total"],
                "n_rejected": p["n_rejected"],
                "fdr": f"{p['fdr']:.6f}",
                "power": f"{p['power']:.6f}",
                "rejection_rate": f"{p['rejection_rate']:.6f}",
            })
    return rows


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def run_baseline(
    claims: list[ScoredClaim],
    alphas: list[float],
    seed: int,
    cal_fraction: float,
    min_stratum_size: int,
) -> dict:
    cal_claims, test_claims = patient_level_split(
        claims, seed=seed, cal_fraction=cal_fraction
    )
    cal = claims_to_arrays(cal_claims)
    test = claims_to_arrays(test_claims)

    per_alpha: dict[str, dict] = {}
    for alpha in sorted(alphas):
        logger.info(f"=== StratCP α = {alpha} ===")
        block = run_stratcp_at_alpha(
            cal=cal,
            test=test,
            alpha=alpha,
            min_stratum_size=min_stratum_size,
        )
        key = f"{alpha:.4f}".rstrip("0").rstrip(".")
        per_alpha[key] = block
        g = block["global"]
        logger.info(
            f"  global FDR={g['fdr']:.4f}  power={g['power']:.4f}  "
            f"rejected={g['n_rejected']}/{g['n_total']}"
        )

    return {
        "config": {
            "seed": seed,
            "cal_fraction": cal_fraction,
            "min_stratum_size": min_stratum_size,
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


def write_comparison_csv(
    stratcp_report: dict,
    cfbh_report: Optional[dict],
    csv_path: Path,
) -> None:
    rows = stratcp_rows_for_csv(stratcp_report)
    if cfbh_report is not None:
        rows += cfbh_rows_for_csv(cfbh_report)
    rows.sort(
        key=lambda r: (float(r["alpha"]), r["stratum"], r["method"])
    )
    fieldnames = [
        "alpha", "stratum", "method", "n_test",
        "n_rejected", "fdr", "power", "rejection_rate",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote comparison CSV → {csv_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StratCP baseline driver (Task 6)"
    )
    parser.add_argument(
        "--scored-claims",
        type=Path,
        default=None,
        help="Scored-claims JSON (same format as "
             "run_openi_recalibrated_eval.py).  Required unless "
             "--dry-run is set.",
    )
    parser.add_argument(
        "--cfbh-json",
        type=Path,
        default=None,
        help="Optional cfBH result JSON to merge into the CSV for "
             "head-to-head comparison.  If omitted, only StratCP rows "
             "are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the output JSON + CSV.",
    )
    parser.add_argument(
        "--alpha", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument(
        "--min-stratum-size",
        type=int,
        default=20,
        help="StratCP strata smaller than this fall back to the "
             "pooled quantile (default 20).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a synthetic fixture (identical to the one in "
             "run_openi_recalibrated_eval.py --dry-run).",
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

    stratcp_report = run_baseline(
        claims=claims,
        alphas=args.alpha,
        seed=args.seed,
        cal_fraction=args.cal_fraction,
        min_stratum_size=args.min_stratum_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stratcp_vs_cfbh.json"
    csv_path = args.output_dir / "stratcp_vs_cfbh.csv"
    json_path.write_text(
        json.dumps(
            {
                "stratcp": stratcp_report,
                "cfbh": load_cfbh_report(args.cfbh_json),
            },
            indent=2,
        )
    )
    logger.info(f"Wrote StratCP report → {json_path}")

    write_comparison_csv(
        stratcp_report=stratcp_report,
        cfbh_report=load_cfbh_report(args.cfbh_json),
        csv_path=csv_path,
    )

    # Summary print
    print("\n=== StratCP summary ===")
    for alpha_str, block in stratcp_report["per_alpha"].items():
        g = block["global"]
        print(
            f"  α={alpha_str}: global FDR={g['fdr']:.4f}  "
            f"power={g['power']:.4f}  "
            f"rejected={g['n_rejected']}/{g['n_total']}"
        )


if __name__ == "__main__":
    main()
