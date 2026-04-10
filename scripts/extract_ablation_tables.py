"""Extract formatted ablation tables from saved eval results for the paper.

Produces LaTeX-ready and markdown tables for:
  - Per-pathology breakdown (14 pathologies)
  - Per-hard-negative-type breakdown (10 neg types)
  - Oracle vs retrieval comparison
  - Patient-cluster bootstrap 95% CIs

Outputs to /Users/aayanalwani/VeriFact/verifact/figures/ablation_tables.md
and ablation_tables.tex.
"""

from __future__ import annotations

import json
from pathlib import Path


ORACLE_PATH = "/tmp/v2_final/full_results.json"
RETR_PATH = "/tmp/v2_retr/full_results.json"
OUT_DIR = Path("/Users/aayanalwani/VeriFact/verifact/figures")


def load_results():
    with open(ORACLE_PATH) as f:
        oracle = json.load(f)
    with open(RETR_PATH) as f:
        retrieval = json.load(f)
    return oracle, retrieval


def format_pathology_table(oracle: dict, retrieval: dict) -> tuple[str, str]:
    md_lines = [
        "## Per-Pathology Breakdown",
        "",
        "| Pathology | n | Oracle Acc | Oracle F1 | Retr Acc | Retr F1 | Δ Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Pathology & n & Oracle Acc & Oracle F1 & Retr Acc & Retr F1 & $\\Delta$ Acc \\\\",
        "\\midrule",
    ]
    path_o = oracle.get("pathology_results", {})
    path_r = retrieval.get("pathology_results", {})
    all_paths = sorted(set(path_o.keys()) | set(path_r.keys()))
    for p in all_paths:
        po = path_o.get(p, {})
        pr = path_r.get(p, {})
        n = po.get("n") or pr.get("n", 0)
        oacc = po.get("accuracy", float("nan"))
        of1 = po.get("macro_f1", float("nan"))
        racc = pr.get("accuracy", float("nan"))
        rf1 = pr.get("macro_f1", float("nan"))
        diff = racc - oacc if (oacc == oacc and racc == racc) else float("nan")
        md_lines.append(
            f"| {p} | {n} | {oacc:.3f} | {of1:.3f} | {racc:.3f} | {rf1:.3f} | {diff:+.3f} |"
        )
        tex_lines.append(
            f"{p} & {n} & {oacc:.3f} & {of1:.3f} & {racc:.3f} & {rf1:.3f} & ${diff:+.3f}$ \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(md_lines), "\n".join(tex_lines)


def format_neg_type_table(oracle: dict, retrieval: dict) -> tuple[str, str]:
    md_lines = [
        "## Per Hard-Negative Type Breakdown",
        "",
        "| Negative Type | n | Oracle Acc | Retr Acc | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Neg. Type & n & Oracle Acc & Retr Acc & $\\Delta$ \\\\",
        "\\midrule",
    ]
    nt_o = oracle.get("neg_type_results", {})
    nt_r = retrieval.get("neg_type_results", {})
    # Exclude 'none' and 'mismatched_evidence' (not hard negatives)
    interesting = [
        "negation", "laterality_swap", "severity_swap", "temporal_error",
        "finding_substitution", "region_swap", "device_line_error",
        "omission_as_support",
    ]
    for nt in interesting:
        po = nt_o.get(nt, {})
        pr = nt_r.get(nt, {})
        n = po.get("n") or pr.get("n", 0)
        oacc = po.get("accuracy", float("nan"))
        racc = pr.get("accuracy", float("nan"))
        diff = racc - oacc if (oacc == oacc and racc == racc) else float("nan")
        nice = nt.replace("_", " ")
        md_lines.append(f"| {nice} | {n} | {oacc:.3f} | {racc:.3f} | {diff:+.3f} |")
        tex_lines.append(f"{nice} & {n} & {oacc:.3f} & {racc:.3f} & ${diff:+.3f}$ \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(md_lines), "\n".join(tex_lines)


def format_main_table(oracle: dict, retrieval: dict) -> tuple[str, str]:
    vm_o = oracle["verifier_metrics"]["test"]
    vm_r = retrieval["verifier_metrics"]["test"]
    cd_o = oracle["calibration_diagnostics"]
    cd_r = retrieval["calibration_diagnostics"]

    md_lines = [
        "## Main Results (Oracle vs Retrieval, binary v2 verifier)",
        "",
        "| Metric | Oracle | Retrieval | Δ |",
        "|---|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & Oracle & Retrieval & $\\Delta$ \\\\",
        "\\midrule",
    ]

    def row(name, o, r, fmt=".4f"):
        diff = r - o
        md_lines.append(f"| {name} | {o:{fmt}} | {r:{fmt}} | {diff:+{fmt}} |")
        tex_lines.append(f"{name} & {o:{fmt}} & {r:{fmt}} & ${diff:+{fmt}}$ \\\\")

    row("Accuracy", vm_o["accuracy"], vm_r["accuracy"])
    row("Macro F1", vm_o["macro_f1"], vm_r["macro_f1"])
    row("NotContra F1", vm_o["per_class_f1"][0], vm_r["per_class_f1"][0])
    row("Contradicted F1", vm_o["per_class_f1"][1], vm_r["per_class_f1"][1])
    row("ECE (calibrated)", cd_o["ece_after_temp"], cd_r["ece_after_temp"])

    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(md_lines), "\n".join(tex_lines)


def format_conformal_table(oracle: dict, retrieval: dict) -> tuple[str, str]:
    md_lines = [
        "## Conformal FDR Control (inverted cfBH, global BH)",
        "",
        "| α | Oracle n_green | Oracle FDR | Oracle Power | Retr n_green | Retr FDR | Retr Power |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "$\\alpha$ & Oracle $n_\\text{green}$ & Oracle FDR & Oracle Power & Retr $n_\\text{green}$ & Retr FDR & Retr Power \\\\",
        "\\midrule",
    ]
    oc = oracle.get("conformal_results", {})
    rc = retrieval.get("conformal_results", {})
    all_keys = sorted(oc.keys(), key=lambda k: float(k.replace("alpha_", "")))
    for key in all_keys:
        co = oc[key]
        cr = rc.get(key, {})
        a = co["alpha"]
        md_lines.append(
            f"| {a:.2f} | {co['n_green']:,} | {co['fdr']:.4f} | {co.get('power', 0):.4f} | "
            f"{cr.get('n_green', 0):,} | {cr.get('fdr', float('nan')):.4f} | {cr.get('power', 0):.4f} |"
        )
        tex_lines.append(
            f"{a:.2f} & {co['n_green']:,} & {co['fdr']:.4f} & {co.get('power', 0):.4f} & "
            f"{cr.get('n_green', 0):,} & {cr.get('fdr', float('nan')):.4f} & {cr.get('power', 0):.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(md_lines), "\n".join(tex_lines)


def format_bootstrap_table(oracle: dict) -> str:
    """Patient-cluster bootstrap 95% CIs."""
    md_lines = [
        "## Patient-Cluster Bootstrap 95% CIs",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---:|:---:|",
    ]
    boot = oracle.get("bootstrap_ci", {})
    for metric, ci in boot.items():
        if isinstance(ci, dict) and "mean" in ci:
            md_lines.append(
                f"| {metric} | {ci['mean']:.4f} | [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |"
            )
    if len(md_lines) == 4:
        return ""
    return "\n".join(md_lines)


def main():
    oracle, retrieval = load_results()

    main_md, main_tex = format_main_table(oracle, retrieval)
    conf_md, conf_tex = format_conformal_table(oracle, retrieval)
    path_md, path_tex = format_pathology_table(oracle, retrieval)
    neg_md, neg_tex = format_neg_type_table(oracle, retrieval)
    boot_md = format_bootstrap_table(oracle)

    md = "\n\n".join([main_md, conf_md, path_md, neg_md, boot_md]).strip()
    tex = "\n\n".join([main_tex, conf_tex, path_tex, neg_tex])

    md_path = OUT_DIR / "ablation_tables.md"
    tex_path = OUT_DIR / "ablation_tables.tex"
    md_path.write_text(md)
    tex_path.write_text(tex)
    print(f"Saved:\n  {md_path}\n  {tex_path}")
    print()
    print("=" * 70)
    print(md)


if __name__ == "__main__":
    main()
