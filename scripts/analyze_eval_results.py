"""Analyze and compare oracle vs retrieval-augmented eval results.

Reads eval JSONs from Modal volume and produces a comparison table
suitable for the NeurIPS 2026 paper (Table 1: Verifier quality, Table 2:
Conformal FDR control).

Usage:
    # Pull results from Modal first
    modal volume get claimguard-data /eval_results /tmp/oracle_res --force
    modal volume get claimguard-data /eval_results_retrieved /tmp/retr_res --force

    python3 scripts/analyze_eval_results.py \
        --oracle /tmp/oracle_res \
        --retrieval /tmp/retr_res
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load full_results.json produced by modal_run_evaluation.py."""
    path = results_dir / "full_results.json"
    if not path.exists():
        # Fall back to any JSON in the directory
        candidates = list(results_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No eval JSON found in {results_dir}")
        path = candidates[0]
    with open(path) as f:
        return json.load(f)


def print_verifier_table(oracle: dict, retrieval: dict | None = None) -> None:
    """Table 1: verifier classification metrics on test set."""
    print("\n" + "=" * 70)
    print("TABLE 1. Verifier Classification Metrics (test set)")
    print("=" * 70)
    header = f"{'Metric':<28} {'Oracle':>12}"
    if retrieval:
        header += f" {'Retrieval':>12} {'Δ (R-O)':>10}"
    print(header)
    print("-" * 70)

    # Detect 2-class vs 3-class by checking length of per_class_f1
    per_class = oracle["verifier_metrics"]["test"]["per_class_f1"]
    num_classes = len(per_class)

    if num_classes == 2:
        rows = [
            ("Accuracy", "accuracy"),
            ("Macro F1", "macro_f1"),
            ("NotContra F1", "per_class_f1", 0),
            ("Contradicted F1", "per_class_f1", 1),
        ]
    else:
        rows = [
            ("Accuracy", "accuracy"),
            ("Macro F1", "macro_f1"),
            ("Supported F1", "per_class_f1", 0),
            ("Contradicted F1", "per_class_f1", 1),
            ("Insufficient F1", "per_class_f1", 2),
        ]

    for row in rows:
        name = row[0]
        key = row[1]
        idx = row[2] if len(row) > 2 else None

        o = oracle["verifier_metrics"]["test"][key]
        if idx is not None:
            o = o[idx]

        line = f"{name:<28} {o:>12.4f}"
        if retrieval:
            r = retrieval["verifier_metrics"]["test"][key]
            if idx is not None:
                r = r[idx]
            diff = r - o
            line += f" {r:>12.4f} {diff:>+10.4f}"
        print(line)


def print_calibration_table(oracle: dict, retrieval: dict | None = None) -> None:
    """Temperature scaling + ECE diagnostics."""
    print("\n" + "=" * 70)
    print("TABLE 2. Calibration Diagnostics")
    print("=" * 70)
    header = f"{'Metric':<28} {'Oracle':>12}"
    if retrieval:
        header += f" {'Retrieval':>12}"
    print(header)
    print("-" * 70)

    for name, key in [
        ("Temperature", "temperature"),
        ("ECE (raw)", "ece_before_temp"),
        ("ECE (calibrated)", "ece_after_temp"),
    ]:
        o = oracle["calibration_diagnostics"][key]
        line = f"{name:<28} {o:>12.4f}"
        if retrieval:
            r = retrieval["calibration_diagnostics"][key]
            line += f" {r:>12.4f}"
        print(line)


def print_conformal_table(oracle: dict, retrieval: dict | None = None) -> None:
    """cfBH conformal FDR control at each alpha level."""
    print("\n" + "=" * 70)
    print("TABLE 3. Conformal FDR Control (cfBH, Jin & Candes 2023)")
    print("=" * 70)
    header = f"{'α':>6} {'FDR (O)':>10} {'Cov (O)':>10} {'N_green (O)':>12}"
    if retrieval:
        header += f" {'FDR (R)':>10} {'Cov (R)':>10} {'N_green (R)':>12}"
    print(header)
    print("-" * 70)

    for key, cr in oracle["conformal_results"].items():
        alpha = cr["alpha"]
        fdr_o = cr["fdr"]
        cov_o = cr["coverage_green_frac"]
        n_o = cr["n_green"]

        line = f"{alpha:>6.2f} {fdr_o:>10.4f} {cov_o:>10.4f} {n_o:>12d}"

        if retrieval and key in retrieval["conformal_results"]:
            crr = retrieval["conformal_results"][key]
            line += f" {crr['fdr']:>10.4f} {crr['coverage_green_frac']:>10.4f} {crr['n_green']:>12d}"

        print(line)


def print_decision_gate(oracle: dict) -> None:
    """Phase 8: Decision gate check."""
    print("\n" + "=" * 70)
    print("DECISION GATE")
    print("=" * 70)
    vm = oracle["verifier_metrics"]["test"]
    cd = oracle["calibration_diagnostics"]

    gates = [
        ("Accuracy ≥ 0.60", vm["accuracy"] >= 0.60, f"{vm['accuracy']:.4f}"),
        ("Macro F1 ≥ 0.55", vm["macro_f1"] >= 0.55, f"{vm['macro_f1']:.4f}"),
        ("ECE (cal) ≤ 0.10", cd["ece_after_temp"] <= 0.10, f"{cd['ece_after_temp']:.4f}"),
    ]

    # Check FDR control at each alpha (FDR ≤ alpha + 5%)
    for key, cr in oracle["conformal_results"].items():
        alpha = cr["alpha"]
        tolerance = alpha + 0.05
        gates.append((
            f"FDR(α={alpha}) ≤ {tolerance:.2f}",
            cr["fdr"] <= tolerance,
            f"{cr['fdr']:.4f}",
        ))

    all_pass = True
    for name, passed, value in gates:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:<8} {name:<30} (value: {value})")
        all_pass = all_pass and passed

    print("-" * 70)
    print(f"OVERALL: {'✓ ALL GATES PASSED' if all_pass else '✗ SOME GATES FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, help="Path to oracle eval results dir")
    parser.add_argument("--retrieval", help="Path to retrieval eval results dir (optional)")
    args = parser.parse_args()

    oracle = load_results(Path(args.oracle))
    retrieval = load_results(Path(args.retrieval)) if args.retrieval else None

    print_verifier_table(oracle, retrieval)
    print_calibration_table(oracle, retrieval)
    print_conformal_table(oracle, retrieval)
    print_decision_gate(oracle)


if __name__ == "__main__":
    main()
