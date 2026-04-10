"""Compile all baseline results into one publication-ready comparison table.

Compares:
  - Rule-based negation detector (local)
  - Untrained RoBERTa-large (random weights, Modal)
  - Trained binary verifier v2 (oracle evidence)
  - Trained binary verifier v2 (retrieval-augmented evidence)
  - Zero-shot LLM judge (if available)

Outputs markdown + LaTeX tables.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_trained_results(path: str) -> dict:
    with open(path) as f:
        r = json.load(f)
    vm = r["verifier_metrics"]["test"]
    cd = r["calibration_diagnostics"]
    cm = vm["confusion_matrix"]
    # Compute contra recall + precision from confusion matrix
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    contra_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    contra_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "accuracy": vm["accuracy"],
        "macro_f1": vm["macro_f1"],
        "f1_notcontra": vm["per_class_f1"][0],
        "f1_contra": vm["per_class_f1"][1],
        "precision_contra": contra_precision,
        "recall_contra": contra_recall,
        "ece": cd["ece_after_temp"],
        "confusion_matrix": cm,
    }


def load_baseline_json(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


def main():
    rows = []

    # Rule-based
    rb = load_baseline_json(
        "/Users/aayanalwani/VeriFact/verifact/figures/baseline_rule_based.json"
    )
    for r in rb:
        if "oracle" in r["name"]:
            rows.append({
                "name": "Rule-based (oracle ev.)",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "f1_contra": r["f1_contra"],
                "precision_contra": r["precision_contra"],
                "recall_contra": r["recall_contra"],
                "auroc": r["auroc"],
                "confusion_matrix": r["confusion_matrix"],
                "ece": None,
            })

    # Untrained RoBERTa
    path = "/tmp/untrained/full_results.json"
    if Path(path).exists():
        t = load_trained_results(path)
        rows.append({
            "name": "Untrained RoBERTa-large",
            "accuracy": t["accuracy"],
            "macro_f1": t["macro_f1"],
            "f1_contra": t["f1_contra"],
            "precision_contra": t["precision_contra"],
            "recall_contra": t["recall_contra"],
            "auroc": None,
            "confusion_matrix": t["confusion_matrix"],
            "ece": t["ece"],
        })

    # Zero-shot LLM
    zs = load_baseline_json(
        "/Users/aayanalwani/VeriFact/verifact/figures/baseline_zeroshot_llm.json"
    )
    for r in zs:
        rows.append({
            "name": f"Zero-shot LLM judge (n={r['n_claims']})",
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "f1_contra": r["f1_contra"],
            "precision_contra": r["precision_contra"],
            "recall_contra": r["recall_contra"],
            "auroc": r.get("auroc"),
            "confusion_matrix": r["confusion_matrix"],
            "ece": None,
        })

    # CheXagent-8b (multimodal VLM baseline)
    chex_path = "/tmp/chexagent/final.json"
    if Path(chex_path).exists():
        with open(chex_path) as f:
            cx = json.load(f)
        rows.append({
            "name": "CheXagent-8b (VLM, image+text)",
            "accuracy": cx["accuracy"],
            "macro_f1": cx["macro_f1"],
            "f1_contra": cx["f1_contra"],
            "precision_contra": cx["precision_contra"],
            "recall_contra": cx["recall_contra"],
            "auroc": cx.get("auroc"),
            "confusion_matrix": cx["confusion_matrix"],
            "ece": None,
        })

    # Trained verifier v2 (oracle)
    t = load_trained_results("/tmp/v2_final/full_results.json")
    rows.append({
        "name": "Trained binary v2 (oracle ev.)",
        "accuracy": t["accuracy"],
        "macro_f1": t["macro_f1"],
        "f1_contra": t["f1_contra"],
        "precision_contra": t["precision_contra"],
        "recall_contra": t["recall_contra"],
        "auroc": 0.9952,  # from local compute
        "confusion_matrix": t["confusion_matrix"],
        "ece": t["ece"],
    })

    # Trained verifier v2 (retrieval)
    t = load_trained_results("/tmp/v2_retr/full_results.json")
    rows.append({
        "name": "Trained binary v2 (retrieved ev.)",
        "accuracy": t["accuracy"],
        "macro_f1": t["macro_f1"],
        "f1_contra": t["f1_contra"],
        "precision_contra": t["precision_contra"],
        "recall_contra": t["recall_contra"],
        "auroc": None,
        "confusion_matrix": t["confusion_matrix"],
        "ece": t["ece"],
    })

    # Markdown
    md = ["# Baseline Comparison (binary hallucination detection)\n"]
    md.append("| Method | Accuracy | Macro F1 | Contra F1 | Contra Prec | Contra Rec | AUROC | ECE |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "—"
        ece = f"{r['ece']:.4f}" if r.get("ece") is not None else "—"
        md.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['f1_contra']:.4f} | {r['precision_contra']:.4f} | "
            f"{r['recall_contra']:.4f} | {auroc} | {ece} |"
        )

    md_text = "\n".join(md)

    # LaTeX
    tex = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Acc & Macro F1 & Contra F1 & Contra P & Contra R & AUROC & ECE \\\\",
        "\\midrule",
    ]
    for r in rows:
        auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "---"
        ece = f"{r['ece']:.4f}" if r.get("ece") is not None else "---"
        name = r['name'].replace("_", "\\_").replace("&", "\\&")
        tex.append(
            f"{name} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & "
            f"{r['f1_contra']:.4f} & {r['precision_contra']:.4f} & "
            f"{r['recall_contra']:.4f} & {auroc} & {ece} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    tex_text = "\n".join(tex)

    out_md = Path("/Users/aayanalwani/VeriFact/verifact/figures/baseline_comparison.md")
    out_tex = Path("/Users/aayanalwani/VeriFact/verifact/figures/baseline_comparison.tex")
    out_md.write_text(md_text)
    out_tex.write_text(tex_text)

    print(md_text)
    print()
    print(f"Saved: {out_md}\nSaved: {out_tex}")


if __name__ == "__main__":
    main()
