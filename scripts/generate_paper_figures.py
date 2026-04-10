"""Generate publication-ready figures from v2 eval predictions.

Creates:
  fig1_score_distributions.pdf  — P(not-contra) distributions for contra vs not-contra
  fig2_reliability.pdf           — calibration reliability diagram
  fig3_fdr_power.pdf             — observed FDR + power across alpha levels
  fig4_confusion.pdf             — test confusion matrix (annotated)
  fig5_roc_pr.pdf                — ROC + PR curves
  fig6_per_neg_type.pdf          — per hard-neg-type accuracy bar chart

Inputs from /tmp/v2_final (oracle) and /tmp/v2_retr (retrieval augmented).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    confusion_matrix as cm_fn,
)

# Set publication-quality defaults
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
    "ps.fonttype": 42,
})

COLORS = {
    "not_contra": "#2E86AB",
    "contra": "#E63946",
    "oracle": "#2E86AB",
    "retrieval": "#F77F00",
    "target": "#808080",
}

OUT_DIR = Path("/Users/aayanalwani/VeriFact/verifact/figures")
OUT_DIR.mkdir(exist_ok=True)


def load_oracle():
    test = np.load("/tmp/v2_final/test.npz")
    cal = np.load("/tmp/v2_final/cal.npz")
    with open("/tmp/v2_final/full_results.json") as f:
        res = json.load(f)
    return test, cal, res


def load_retrieval():
    with open("/tmp/v2_retr/full_results.json") as f:
        res = json.load(f)
    return res


def fig1_score_distributions(test):
    """Histogram of P(not-contra) scores, split by true label."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    scores = test["supported_prob"]
    labels = test["labels"]

    nc_scores = scores[labels == 0]
    contra_scores = scores[labels == 1]

    bins = np.linspace(0, 1, 51)
    ax.hist(nc_scores, bins=bins, color=COLORS["not_contra"], alpha=0.7,
            label=f"Not contradicted (n={len(nc_scores):,})", density=True,
            edgecolor="white", linewidth=0.3)
    ax.hist(contra_scores, bins=bins, color=COLORS["contra"], alpha=0.7,
            label=f"Contradicted (n={len(contra_scores):,})", density=True,
            edgecolor="white", linewidth=0.3)

    ax.set_xlabel(r"Verifier score $P(\mathrm{not\text{-}contra})$")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by true label (test set, n=15,000)")
    ax.legend(loc="upper center", framealpha=0.95)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_score_distributions.pdf")
    plt.savefig(OUT_DIR / "fig1_score_distributions.png")
    plt.close()
    print(f"  saved fig1_score_distributions")


def fig2_reliability(res):
    """Reliability diagram (10-bin calibration curve)."""
    bins_raw = res["calibration_diagnostics"]["reliability_bins_before"]
    bins_cal = res["calibration_diagnostics"]["reliability_bins_after"]
    ece_raw = res["calibration_diagnostics"]["ece_before_temp"]
    ece_cal = res["calibration_diagnostics"]["ece_after_temp"]
    T = res["calibration_diagnostics"]["temperature"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)

    for ax, bins, ece, title in zip(
        axes, [bins_raw, bins_cal], [ece_raw, ece_cal],
        ["Before temperature scaling", f"After T={T:.3f}"],
    ):
        xs, ys, ns = [], [], []
        for b in bins:
            if b["n"] is None or b["n"] == 0:
                continue
            xs.append(b["conf"])
            ys.append(b["acc"])
            ns.append(b["n"])

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1,
                label="Perfect calibration")

        # Size proportional to bin count
        sizes = np.array(ns)
        sizes = 20 + 100 * sizes / sizes.max()
        ax.scatter(xs, ys, s=sizes, color=COLORS["oracle"], alpha=0.7,
                   edgecolor="white", linewidth=0.5, label="Bin (sized by count)",
                   zorder=3)
        ax.plot(xs, ys, color=COLORS["oracle"], alpha=0.5, linewidth=1, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

    axes[0].set_ylabel("Empirical accuracy")
    axes[0].legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_reliability.pdf")
    plt.savefig(OUT_DIR / "fig2_reliability.png")
    plt.close()
    print(f"  saved fig2_reliability")


def fig3_fdr_power(res_oracle, res_retrieval):
    """Observed FDR and power across alpha levels, with α=FDR target line."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    def extract(res):
        rows = []
        for key, cr in res["conformal_results"].items():
            rows.append((cr["alpha"], cr["fdr"], cr.get("power", 0),
                         cr["coverage_green_frac"], cr["n_green"]))
        rows.sort()
        return list(zip(*rows))

    alphas_o, fdrs_o, powers_o, covs_o, _ = extract(res_oracle)
    alphas_r, fdrs_r, powers_r, covs_r, _ = extract(res_retrieval)

    ax = axes[0]
    ax.plot([0, 0.25], [0, 0.25], color=COLORS["target"], linestyle="--",
            label=r"Target: $\mathrm{FDR} = \alpha$", linewidth=1)
    ax.plot(alphas_o, fdrs_o, "o-", color=COLORS["oracle"],
            label="Oracle evidence", markersize=7, linewidth=1.8)
    ax.plot(alphas_r, fdrs_r, "s-", color=COLORS["retrieval"],
            label="Retrieval-augmented", markersize=6, linewidth=1.8)
    ax.fill_between([0, 0.25], [0, 0.25], 0, alpha=0.1, color=COLORS["target"])
    ax.set_xlabel(r"Target FDR level $\alpha$")
    ax.set_ylabel("Observed FDR")
    ax.set_title("FDR control (inverted cfBH)")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1]
    ax.plot(alphas_o, powers_o, "o-", color=COLORS["oracle"],
            label="Oracle evidence", markersize=7, linewidth=1.8)
    ax.plot(alphas_r, powers_r, "s-", color=COLORS["retrieval"],
            label="Retrieval-augmented", markersize=6, linewidth=1.8)
    ax.set_xlabel(r"Target FDR level $\alpha$")
    ax.set_ylabel("Power (fraction of faithful flagged)")
    ax.set_title("Statistical power")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0.9, 1.005)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_fdr_power.pdf")
    plt.savefig(OUT_DIR / "fig3_fdr_power.png")
    plt.close()
    print(f"  saved fig3_fdr_power")


def fig4_confusion(test):
    """Annotated confusion matrix for oracle test predictions."""
    labels = test["labels"]
    preds = test["preds"]
    cm = cm_fn(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")

    classes = ["NotContra", "Contradicted"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (binary verifier)")

    # Annotate cells
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100
            text_color = "white" if val > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_confusion.pdf")
    plt.savefig(OUT_DIR / "fig4_confusion.png")
    plt.close()
    print(f"  saved fig4_confusion")


def fig5_roc_pr(test):
    """ROC + PR curves for binary classification."""
    labels = test["labels"]
    scores = test["supported_prob"]
    # Treat class 0 (not-contra) as positive; flip if needed
    y_true = (labels == 0).astype(int)
    y_score = scores

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["oracle"], linewidth=2,
            label=f"Verifier (AUROC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.plot(rec, prec, color=COLORS["oracle"], linewidth=2,
            label=f"Verifier (AP = {ap:.4f})")
    # Baseline = prevalence of positive class
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.4, linewidth=1,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_roc_pr.pdf")
    plt.savefig(OUT_DIR / "fig5_roc_pr.png")
    plt.close()
    print(f"  saved fig5_roc_pr")


def fig6_per_neg_type(res):
    """Per-hard-negative-type accuracy as horizontal bar chart."""
    neg_type_data = res.get("neg_type_results", {})
    if not neg_type_data:
        print("  skipped fig6_per_neg_type (no data)")
        return

    # Sort by accuracy descending, exclude 'none' and 'mismatched_evidence'
    types = sorted(
        [(k, v) for k, v in neg_type_data.items()
         if k not in {"none", "mismatched_evidence"}],
        key=lambda x: x[1]["accuracy"], reverse=True,
    )

    names = [t[0].replace("_", " ") for t in types]
    accs = [t[1]["accuracy"] for t in types]
    ns = [t[1]["n"] for t in types]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, color=COLORS["oracle"], alpha=0.8,
                   edgecolor="white", linewidth=0.5)

    for i, (bar, acc, n) in enumerate(zip(bars, accs, ns)):
        ax.text(acc + 0.003, i, f"{acc:.3f} (n={n})", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Test accuracy (binary)")
    ax.set_title("Per hard-negative type accuracy (oracle eval)")
    ax.set_xlim(0.85, 1.05)
    ax.axvline(0.98, color="k", linestyle="--", alpha=0.3, linewidth=1,
               label="Overall acc = 0.983")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_per_neg_type.pdf")
    plt.savefig(OUT_DIR / "fig6_per_neg_type.png")
    plt.close()
    print(f"  saved fig6_per_neg_type")


def main():
    print(f"Output directory: {OUT_DIR}")
    test, cal, res_o = load_oracle()
    res_r = load_retrieval()
    print("Loaded predictions and results")
    print()

    fig1_score_distributions(test)
    fig2_reliability(res_o)
    fig3_fdr_power(res_o, res_r)
    fig4_confusion(test)
    fig5_roc_pr(test)
    fig6_per_neg_type(res_o)

    print()
    print(f"All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
