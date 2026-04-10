"""Compute final paper-ready results LOCALLY from saved predictions.

Uses the INVERTED label-conditional cfBH procedure that was validated
interactively on 2026-04-05. This script does NOT require Modal compute.

Reads:
    /tmp/binary_res/cal_predictions.npz
    /tmp/binary_res/test_predictions.npz

Writes:
    /tmp/binary_res/final_paper_results.json
    /tmp/binary_res/final_paper_results.md (human-readable tables)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score,
)


def main() -> None:
    cal = np.load("/tmp/binary_res/cal_predictions.npz")
    test = np.load("/tmp/binary_res/test_predictions.npz")

    cal_supp = cal["supported_prob"]
    test_supp = test["supported_prob"]
    cal_labels = cal["labels"]  # Binary: 0=not-contra, 1=contra
    test_labels = test["labels"]
    cal_preds = cal["preds"]
    test_preds = test["preds"]

    print("=" * 70)
    print("ClaimGuard-CXR FINAL RESULTS (binary verifier, inverted cfBH)")
    print("=" * 70)
    print(f"n_cal={len(cal_labels)}, n_test={len(test_labels)}")
    print(f"cal: {np.bincount(cal_labels).tolist()}  (not-contra, contra)")
    print(f"test: {np.bincount(test_labels).tolist()}")
    print()

    results = {}

    # ======================================================================
    # TABLE 1: Verifier classification metrics
    # ======================================================================
    print("=" * 70)
    print("TABLE 1. Verifier Classification Metrics (test set, binary)")
    print("=" * 70)
    acc = accuracy_score(test_labels, test_preds)
    mf1 = f1_score(test_labels, test_preds, average="macro")
    per_class_f1 = f1_score(test_labels, test_preds, average=None).tolist()
    precision = precision_score(test_labels, test_preds, average=None).tolist()
    recall = recall_score(test_labels, test_preds, average=None).tolist()
    cm = confusion_matrix(test_labels, test_preds).tolist()

    # AUC using score
    auc = roc_auc_score((test_labels == 0).astype(int), test_supp)
    ap = average_precision_score((test_labels == 0).astype(int), test_supp)

    print(f"  Accuracy:             {acc:.4f}")
    print(f"  Macro F1:             {mf1:.4f}")
    print(f"  F1 (NotContra):       {per_class_f1[0]:.4f}")
    print(f"  F1 (Contradicted):    {per_class_f1[1]:.4f}")
    print(f"  Precision (NotContra):{precision[0]:.4f}")
    print(f"  Recall (NotContra):   {recall[0]:.4f}")
    print(f"  Precision (Contra):   {precision[1]:.4f}")
    print(f"  Recall (Contra):      {recall[1]:.4f}")
    print(f"  AUROC (P(NotContra)): {auc:.4f}")
    print(f"  AP (NotContra):       {ap:.4f}")
    print(f"  Confusion Matrix:     [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print()

    results["classification"] = {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "f1_notcontra": float(per_class_f1[0]),
        "f1_contra": float(per_class_f1[1]),
        "precision_notcontra": float(precision[0]),
        "recall_notcontra": float(recall[0]),
        "precision_contra": float(precision[1]),
        "recall_contra": float(recall[1]),
        "auroc": float(auc),
        "average_precision": float(ap),
        "confusion_matrix": cm,
    }

    # ======================================================================
    # TABLE 2: Calibration (already scaled in saved probs, report ECE)
    # ======================================================================
    print("=" * 70)
    print("TABLE 2. Calibration Diagnostics")
    print("=" * 70)

    def compute_ece(probs, labels, n_bins=10):
        confidences = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if i == n_bins - 1:
                mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(labels)) * abs(
                correct[mask].mean() - confidences[mask].mean()
            )
        return float(ece)

    # Reconstruct probs from supported_prob (binary: [supp, 1-supp])
    test_probs_cal = np.column_stack([test_supp, 1 - test_supp])
    ece = compute_ece(test_probs_cal, test_labels)
    print(f"  ECE (10-bin, calibrated):   {ece:.4f}")
    print(f"  (Temperature scaling was applied upstream)")
    print()
    results["calibration"] = {"ece_calibrated": ece}

    # ======================================================================
    # TABLE 3: INVERTED conformal cfBH FDR control
    # ======================================================================
    print("=" * 70)
    print("TABLE 3. Conformal FDR Control (INVERTED cfBH, global BH)")
    print("=" * 70)

    # Null H0: test claim is contradicted. Cal pool: cal label == 1 (contra).
    cal_contra_scores = cal_supp[cal_labels == 1]
    print(f"  Calibration pool: n={len(cal_contra_scores)} contradicted samples")
    print(f"  Score: P(not-contra) from temperature-calibrated softmax")
    print(f"  p-value = (|cal_contra >= test_score| + 1) / (n_cal + 1)")
    print()

    def conformal_pval(test_s, cal_s):
        cal_asc = np.sort(cal_s)
        counts = len(cal_asc) - np.searchsorted(cal_asc, test_s, side="left")
        return (counts + 1) / (len(cal_s) + 1)

    def bh(pvals, alpha):
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_p = pvals[sorted_idx]
        thresholds = np.arange(1, n + 1) / n * alpha
        below = sorted_p <= thresholds
        if not below.any():
            return np.zeros(n, dtype=bool)
        k = int(np.max(np.where(below)[0]))
        accepted = np.zeros(n, dtype=bool)
        accepted[sorted_idx[:k + 1]] = True
        return accepted

    pvals = conformal_pval(test_supp, cal_contra_scores)
    n_nc_total = int((test_labels == 0).sum())

    conformal_results = {}
    print(f"  {'alpha':>6} {'n_green':>9} {'coverage':>10} {'FDR(contra)':>12} {'Power':>8}")
    print(f"  {'-' * 52}")
    for alpha in [0.01, 0.05, 0.10, 0.15, 0.20]:
        accepted = bh(pvals, alpha)
        n_green = int(accepted.sum())
        if n_green > 0:
            n_contra_in_green = int((test_labels[accepted] == 1).sum())
            fdr = n_contra_in_green / n_green
            n_nc_green = int((test_labels[accepted] == 0).sum())
            power = n_nc_green / n_nc_total
            print(
                f"  {alpha:>6.2f} {n_green:>9d} {n_green / len(pvals):>10.4f} "
                f"{fdr:>12.4f} {power:>8.4f}"
            )
            conformal_results[f"alpha_{alpha}"] = {
                "alpha": alpha,
                "n_green": n_green,
                "coverage": float(n_green / len(pvals)),
                "fdr": float(fdr),
                "power": float(power),
                "n_contra_in_green": n_contra_in_green,
                "n_notcontra_in_green": n_nc_green,
            }
        else:
            print(f"  {alpha:>6.2f} {n_green:>9d} {0:>10.4f} {'NaN':>12} {0:>8.4f}")
    print()
    results["conformal"] = conformal_results

    # ======================================================================
    # Decision gate
    # ======================================================================
    print("=" * 70)
    print("DECISION GATE (NeurIPS 2026 submission criteria)")
    print("=" * 70)
    gates = [
        ("Accuracy >= 0.60", acc >= 0.60, f"{acc:.4f}"),
        ("Macro F1 >= 0.55", mf1 >= 0.55, f"{mf1:.4f}"),
        ("ECE <= 0.10", ece <= 0.10, f"{ece:.4f}"),
        ("AUROC >= 0.90", auc >= 0.90, f"{auc:.4f}"),
    ]
    for alpha_key, cr in conformal_results.items():
        a = cr["alpha"]
        tol = a + 0.05
        gates.append(
            (f"FDR(alpha={a}) <= {tol:.2f}", cr["fdr"] <= tol, f"{cr['fdr']:.4f}")
        )
    all_pass = True
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        mark = "+" if passed else "-"
        print(f"  [{mark}] {status:<6} {name:<30} (value: {value})")
        all_pass = all_pass and passed
    print(f"  {'-' * 50}")
    print(
        f"  OVERALL: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}"
    )
    results["decision_gate"] = {
        "all_pass": all_pass,
        "gates": [
            {"name": n, "pass": bool(p), "value": v} for n, p, v in gates
        ],
    }

    out_path = Path("/tmp/binary_res/final_paper_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()
