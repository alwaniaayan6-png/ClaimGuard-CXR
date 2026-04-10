"""Generate cross-dataset comparison figure: CheXpert Plus vs OpenI.

Shows that FDR control generalizes zero-shot to a different hospital's
chest X-ray dataset, even as accuracy drops 13pp.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

OUT_DIR = Path("/Users/aayanalwani/VeriFact/verifact/figures")
COLORS = {
    "chexpert": "#2E86AB",
    "openi": "#A23B72",
    "target": "#808080",
}


def extract(res_path):
    with open(res_path) as f:
        r = json.load(f)
    rows = []
    for key, cr in r["conformal_results"].items():
        rows.append((cr["alpha"], cr["fdr"], cr.get("power", 0)))
    rows.sort()
    alphas, fdrs, powers = zip(*rows)
    return list(alphas), list(fdrs), list(powers), r


def main():
    a_cx, f_cx, p_cx, res_cx = extract("/tmp/v2_final/full_results.json")
    a_oi, f_oi, p_oi, res_oi = extract("/tmp/openi_res/full_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    # FDR panel
    ax = axes[0]
    ax.plot([0, 0.25], [0, 0.25], color=COLORS["target"], linestyle="--",
            label=r"Target: FDR = $\alpha$", linewidth=1)
    ax.fill_between([0, 0.25], [0, 0.25], 0, alpha=0.08, color=COLORS["target"])
    ax.plot(a_cx, f_cx, "o-", color=COLORS["chexpert"], markersize=7, linewidth=1.8,
            label="CheXpert Plus (in-domain)")
    ax.plot(a_oi, f_oi, "s-", color=COLORS["openi"], markersize=7, linewidth=1.8,
            label="OpenI (cross-dataset, zero-shot)")
    ax.set_xlabel(r"Target FDR level $\alpha$")
    ax.set_ylabel("Observed FDR")
    ax.set_title("FDR control holds cross-dataset")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")

    # Power panel
    ax = axes[1]
    ax.plot(a_cx, p_cx, "o-", color=COLORS["chexpert"], markersize=7, linewidth=1.8,
            label="CheXpert Plus (in-domain)")
    ax.plot(a_oi, p_oi, "s-", color=COLORS["openi"], markersize=7, linewidth=1.8,
            label="OpenI (cross-dataset, zero-shot)")
    ax.set_xlabel(r"Target FDR level $\alpha$")
    ax.set_ylabel("Power (fraction of faithful flagged green)")
    ax.set_title("Power decreases on out-of-distribution data")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig7_cross_dataset.pdf")
    plt.savefig(OUT_DIR / "fig7_cross_dataset.png")
    plt.close()
    print(f"Saved fig7_cross_dataset to {OUT_DIR}/")

    # Also write a small cross-dataset summary table
    vm_cx = res_cx["verifier_metrics"]["test"]
    vm_oi = res_oi["verifier_metrics"]["test"]
    cd_cx = res_cx["calibration_diagnostics"]
    cd_oi = res_oi["calibration_diagnostics"]

    md = ["# Cross-Dataset Transfer Results\n"]
    md.append("Trained on CheXpert Plus, evaluated on OpenI (Indiana U) zero-shot.\n")
    md.append("| Metric | CheXpert Plus | OpenI | Δ |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Accuracy | {vm_cx['accuracy']:.4f} | {vm_oi['accuracy']:.4f} | {vm_oi['accuracy']-vm_cx['accuracy']:+.4f} |")
    md.append(f"| Macro F1 | {vm_cx['macro_f1']:.4f} | {vm_oi['macro_f1']:.4f} | {vm_oi['macro_f1']-vm_cx['macro_f1']:+.4f} |")
    md.append(f"| Contradicted F1 | {vm_cx['per_class_f1'][1]:.4f} | {vm_oi['per_class_f1'][1]:.4f} | {vm_oi['per_class_f1'][1]-vm_cx['per_class_f1'][1]:+.4f} |")
    md.append(f"| ECE (calibrated) | {cd_cx['ece_after_temp']:.4f} | {cd_oi['ece_after_temp']:.4f} | {cd_oi['ece_after_temp']-cd_cx['ece_after_temp']:+.4f} |")
    md.append(f"| Temperature | {cd_cx['temperature']:.4f} | {cd_oi['temperature']:.4f} | — |")
    md.append("\n## Conformal FDR Control (inverted cfBH)\n")
    md.append("| α | CheXpert n_green | CheXpert FDR | CheXpert Power | OpenI n_green | OpenI FDR | OpenI Power |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for ac, fc, pc, ao, fo, po in zip(a_cx, f_cx, p_cx, a_oi, f_oi, p_oi):
        # Get n_green
        ng_cx = res_cx["conformal_results"][f"alpha_{ac}"]["n_green"]
        ng_oi = res_oi["conformal_results"][f"alpha_{ao}"]["n_green"]
        md.append(f"| {ac:.2f} | {ng_cx:,} | {fc:.4f} | {pc:.4f} | {ng_oi:,} | {fo:.4f} | {po:.4f} |")

    (OUT_DIR / "cross_dataset_table.md").write_text("\n".join(md))
    print(f"Saved cross_dataset_table.md to {OUT_DIR}/")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
