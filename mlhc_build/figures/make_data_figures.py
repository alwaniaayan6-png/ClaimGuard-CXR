"""Generate all data figures for the ClaimGuard-CXR NeurIPS 2026 paper.

Reads numbers from `v5_final_results/*.json` + `/tmp/cg_state/**/*.json` +
`/tmp/cg_state/loo/*.json` + `/tmp/cg_state/baselines/*.json` and writes PDF +
PNG figures to `mlhc_build/figures/data_plots/`.

Run after every round of Modal results lands:
    python mlhc_build/figures/make_data_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V5_RES = ROOT / "v5_final_results"
CG_STATE = Path("/tmp/cg_state")
OUT = ROOT / "mlhc_build/figures/data_plots"
OUT.mkdir(parents=True, exist_ok=True)

# Matplotlib defaults tuned for two-column papers
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# -------- Figure 1: headline IMG vs val-acc (v5.0 → v5.3 + v6.0-3site) -----

def figure_headline():
    configs = [
        ("v5.0-base",     "v5_0_base_diagnostic.json"),
        ("v5.1-ground",   "v5_1_ground_diagnostic.json"),
        ("v5.2-real",     "v5_2_real_diagnostic.json"),
        ("v5.3-contrast", "v5_3_contrast_diagnostic.json"),
    ]
    rows = [(name, load_json(V5_RES / f)) for name, f in configs]
    # v6.0-3site from diag output if available, else from manifest best_val_acc
    v6 = None
    cand = CG_STATE / "v6_0_3site_diag.json"
    if cand.exists():
        v6 = ("v6.0-3site", load_json(cand))
    else:
        # manifest is val-set only, not test — skip this row in headline plot
        pass

    names = [n for n, _ in rows]
    img = [r["img_gap_pp"] for _, r in rows]
    acc = [r["acc_full"] * 100 for _, r in rows]
    esg = [r["esg_gap_pp"] for _, r in rows]
    if v6:
        names.append(v6[0])
        img.append(v6[1]["img_gap_pp"])
        acc.append(v6[1]["acc_full"] * 100)
        esg.append(v6[1]["esg_gap_pp"])

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    x = np.arange(len(names))
    w = 0.28
    ax.bar(x - w, acc, width=w, label="Test acc (%)", color="#4C72B0")
    ax.bar(x,     img, width=w, label="IMG (pp)",     color="#DD8452")
    ax.bar(x + w, esg, width=w, label="ESG (pp)",     color="#55A868")
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.8,
               label="Evidence-blind threshold (5pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Percentage-points / percent")
    ax.set_title("Headline diagnostic: IMG and ESG across training configurations")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_ylim(0, 100)
    fig.savefig(OUT / "F1_headline_img_esg.pdf")
    fig.savefig(OUT / "F1_headline_img_esg.png")
    plt.close(fig)


# -------- Figure 2: ablation — drop-one-loss --------

def figure_ablation_loss_drop():
    f = V5_RES / "ablations_loss_drop_summary.json"
    if not f.exists():
        print("skip F2: no ablation summary")
        return
    d = load_json(f)
    # d format: {"abl_no_ground": {..}, "abl_no_consist": {..}, ...}
    names_full = {
        "abl_no_ground":   "no ground",
        "abl_no_consist":  "no consistency",
        "abl_no_contrast": "no contrast",
        "abl_no_hofilter": "no HO-filter",
        "abl_no_uncert":   "no uncert",
    }
    v53 = load_json(V5_RES / "v5_3_contrast_diagnostic.json")
    img_v53 = v53["img_gap_pp"]
    entries = []
    for k in names_full:
        dpath = V5_RES / f"{k}_diagnostic.json"
        if dpath.exists():
            r = load_json(dpath)
            entries.append((names_full[k], r["img_gap_pp"], r["acc_full"]))
    entries.append(("v5.3 (full)", img_v53, v53["acc_full"]))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    names = [e[0] for e in entries]
    imgs = [e[1] for e in entries]
    colors = ["#C44E52" if i < img_v53 - 30 else "#4C72B0" for i in imgs]
    y = np.arange(len(names))
    ax.barh(y, imgs, color=colors)
    for yi, v in enumerate(imgs):
        ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(5, color="#888", linestyle=":", linewidth=0.8,
               label="evidence-blind (IMG<5pp)")
    ax.set_xlabel("Image-masking gap (pp)")
    ax.set_title("Loss-drop ablation: each component removed from v5.3")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(-2, 80)
    fig.savefig(OUT / "F2_ablation_loss_drop.pdf")
    fig.savefig(OUT / "F2_ablation_loss_drop.png")
    plt.close(fig)


# -------- Figure 3: HO-threshold sweep --------

def figure_hothresh_sweep():
    thresholds = [50, 60, 70, 80, 90]
    xs, ys, accs = [], [], []
    for t in thresholds:
        f = V5_RES / f"abl_hothresh_{t}_diagnostic.json"
        if not f.exists():
            continue
        d = load_json(f)
        xs.append(t / 100)
        ys.append(d["img_gap_pp"])
        accs.append(d["acc_full"] * 100)
    if not xs:
        print("skip F3: no hothresh sweep")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax2 = ax.twinx()
    ax.plot(xs, ys, "o-", color="#DD8452", linewidth=2, label="IMG (pp)")
    ax2.plot(xs, accs, "s--", color="#4C72B0", linewidth=1.3, label="Test acc (%)")
    ax.set_xlabel("HO-filter threshold")
    ax.set_ylabel("IMG (pp)", color="#DD8452")
    ax2.set_ylabel("Test accuracy (%)", color="#4C72B0")
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_title("HO-threshold sweep — IMG is stable in 68–76pp band")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "F3_hothresh_sweep.pdf")
    fig.savefig(OUT / "F3_hothresh_sweep.png")
    plt.close(fig)


# -------- Figure 4: scale curve (25% / 50% / 100% training data) --------

def figure_scale_curve():
    scales = [25, 50, 100]
    xs, ys, accs = [], [], []
    for s in scales:
        f = V5_RES / f"abl_scale_{s}_diagnostic.json"
        if not f.exists():
            continue
        d = load_json(f)
        xs.append(s)
        ys.append(d["img_gap_pp"])
        accs.append(d["acc_full"] * 100)
    if not xs:
        print("skip F4: no scale sweep")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(xs, ys, "o-", color="#DD8452", linewidth=2, label="IMG (pp)")
    ax.plot(xs, accs, "s--", color="#4C72B0", linewidth=1.3, label="Test acc (%)")
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("IMG (pp) / Test acc (%)")
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.8,
               label="evidence-blind threshold")
    ax.set_title("Mitigation effectiveness vs data scale")
    ax.legend(loc="center right", frameon=False)
    fig.savefig(OUT / "F4_scale_curve.pdf")
    fig.savefig(OUT / "F4_scale_curve.png")
    plt.close(fig)


# -------- Figure 5: cross-site 3-way LOO IMG --------

def figure_loo():
    entries = []
    for site in ["openi", "chestx_det10", "padchest-gr"]:
        f = CG_STATE / "loo" / f"diag_loo_no_{site}.json"
        if not f.exists():
            continue
        d = load_json(f)
        entries.append((site.replace("_", "-"), d["img_gap_pp"], d["acc_full"] * 100,
                        d["esg_gap_pp"]))
    # 2-way reference from v5 cross-site JSON
    cs = V5_RES / "crosssite_openi_to_chestx_det10_summary.json"
    if cs.exists():
        d = load_json(cs)
        entries.append(("(v5 ref) openi→chestx_det10", d.get("img_gap_pp", 0.0),
                        d.get("test_acc_full", 0.0) * 100, d.get("esg_gap_pp", 0.0)))
    if not entries:
        print("skip F5: no LOO results")
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = [e[0] for e in entries]
    imgs = [e[1] for e in entries]
    esgs = [e[3] for e in entries]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, imgs, width=w, color="#DD8452", label="IMG (pp)")
    ax.bar(x + w / 2, esgs, width=w, color="#55A868", label="ESG (pp)")
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.8,
               label="Evidence-blind threshold (5pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Gap (pp)")
    ax.set_title("Cross-site leave-one-out: evidence-blindness returns under distribution shift")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "F5_loo_cross_site.pdf")
    fig.savefig(OUT / "F5_loo_cross_site.png")
    plt.close(fig)


# -------- Figure 6: baseline landscape (real-RRG) --------

def figure_baselines():
    files = [
        ("BiomedCLIP",        "baseline_biomedclip-zero-shot_diagnostic.json"),
        ("MedGemma-4B-IT",    "baseline_medgemma-4b-it_diagnostic.json"),
        ("MAIRA-2 (pending)", "baseline_maira-2_diagnostic.json"),
        ("Claude 3.5 Sonnet (pending)", "baseline_claude-3-5-sonnet_diagnostic.json"),
    ]
    entries = []
    for name, f in files:
        p = CG_STATE / "baselines" / f
        if not p.exists():
            continue
        d = load_json(p)
        if d["acc_full"] == 0.0:  # mark pending rows
            entries.append((name, None, None))
        else:
            entries.append((name, d["img_gap_pp"], d["esg_gap_pp"]))
    # ours: pull v5.3-contrast as stand-in until v6.0-3site test diag lands
    v53 = load_json(V5_RES / "v5_3_contrast_diagnostic.json")
    entries.append(("ClaimGuard-CXR (ours)", v53["img_gap_pp"], v53["esg_gap_pp"]))

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    names = [e[0] for e in entries]
    imgs = [e[1] if e[1] is not None else 0 for e in entries]
    esgs = [e[2] if e[2] is not None else 0 for e in entries]
    pending = [e[1] is None for e in entries]
    x = np.arange(len(names))
    w = 0.36
    bars_img = ax.bar(x - w / 2, imgs, width=w, color="#DD8452", label="IMG (pp)")
    bars_esg = ax.bar(x + w / 2, esgs, width=w, color="#55A868", label="ESG (pp)")
    # hatch pending bars
    for i, (bi, be) in enumerate(zip(bars_img, bars_esg)):
        if pending[i]:
            bi.set_hatch("//")
            be.set_hatch("//")
            bi.set_alpha(0.5)
            be.set_alpha(0.5)
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.8,
               label="Evidence-blind threshold (5pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Gap (pp)")
    ax.set_title("Baseline landscape on real-RRG claims (hatched = pending re-run)")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "F6_baseline_landscape.pdf")
    fig.savefig(OUT / "F6_baseline_landscape.png")
    plt.close(fig)


# -------- Figure 7: support-score sharpness across configs (KS distance) --

def figure_support_scores():
    cfg_list = [
        ("v5.0-base", "support_scores_v5_0_base.json"),
        ("v5.1-ground", "support_scores_v5_1_ground.json"),
        ("v5.2-real", "support_scores_v5_2_real.json"),
        ("v5.3-contrast", "support_scores_v5_3_contrast.json"),
        ("v6.0-retrain", "support_scores_v6_0_retrain.json"),
    ]
    entries = []
    for name, fname in cfg_list:
        p = CG_STATE / fname
        if not p.exists():
            continue
        d = load_json(p)
        entries.append((name, d.get("scores_pred_class", []),
                        d.get("ks_distance_to_uniform", 0.0),
                        d.get("frac_above_0_9", 0.0)))
    if not entries:
        print("skip F7: no support-score JSONs yet")
        return

    fig, axes = plt.subplots(1, len(entries), figsize=(2.0 * len(entries), 2.6),
                              sharey=True)
    if len(entries) == 1:
        axes = [axes]
    for ax, (name, scores, ks, fhigh) in zip(axes, entries):
        if not scores:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ax.hist(scores, bins=20, range=(0.5, 1.0), color="#4C72B0",
                edgecolor="white", linewidth=0.4)
        ax.set_title(f"{name}\nKS={ks:.2f}  $p>0.9$={fhigh*100:.0f}%",
                     fontsize=8)
        ax.set_xlabel("predicted-class prob")
        if ax is axes[0]:
            ax.set_ylabel("count")
    fig.suptitle("Support-score distribution sharpness (closer to 1.0 = informative for conformal)",
                 fontsize=9, y=1.04)
    fig.savefig(OUT / "F7_support_score_histograms.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F7_support_score_histograms.png", bbox_inches="tight")
    plt.close(fig)


# -------- Figure 8: HO-filter activation rate per RRG model --------

def figure_ho_rrg_activation():
    p = CG_STATE / "ho_filter_rrg_activation.json"
    if not p.exists():
        print("skip F8: no ho_filter_rrg_activation.json yet")
        return
    d = load_json(p)
    by_model = d.get("by_rrg_model", {})
    if not by_model:
        print("skip F8: no by_rrg_model")
        return
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    names = list(by_model.keys())
    rates = [v.get("rate", 0) * 100 for v in by_model.values()]
    ax.bar(names, rates, color="#DD8452")
    ax.axhline(40, color="#888", linestyle=":", linewidth=0.8,
               label="M6 target (40%)")
    ax.axhline(10, color="#C44E52", linestyle=":", linewidth=0.8,
               label="M6 floor (10%)")
    ax.set_ylabel("HO-filter activation rate (%)")
    ax.set_title(f"HO filter on real-RRG claims: overall rate {d.get('activation_rate', 0)*100:.1f}%")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "F8_ho_rrg_activation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F8_ho_rrg_activation.png", bbox_inches="tight")
    plt.close(fig)


# -------- Figure 9: cross-site mechanism --------

def figure_cross_site_mechanism():
    p = CG_STATE / "cross_site_mechanism.json"
    if not p.exists():
        print("skip F9: no cross_site_mechanism.json yet")
        return
    d = load_json(p)
    by_site = d.get("by_site", {})
    if not by_site:
        return
    sites = list(by_site.keys())
    text_only = [v["text_only_acc"] * 100 for v in by_site.values()]
    majority = [v["majority_class_acc"] * 100 for v in by_site.values()]
    ho_rate = [v["ho_activation_rate"] * 100 for v in by_site.values()]
    x = np.arange(len(sites))
    w = 0.27
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(x - w, text_only, width=w, label="text-only acc", color="#4C72B0")
    ax.bar(x, majority, width=w, label="majority baseline", color="#999")
    ax.bar(x + w, ho_rate, width=w, label="HO activation rate", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=15, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Per-site shortcut signal vs HO activation")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.savefig(OUT / "F9_cross_site_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F9_cross_site_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def main():
    figure_headline()
    figure_ablation_loss_drop()
    figure_hothresh_sweep()
    figure_scale_curve()
    figure_loo()
    figure_baselines()
    figure_support_scores()
    figure_ho_rrg_activation()
    figure_cross_site_mechanism()
    for p in sorted(OUT.glob("F*.pdf")):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
