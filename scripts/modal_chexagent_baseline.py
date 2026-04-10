"""Zero-shot CheXagent-8b multimodal baseline for ClaimGuard-CXR.

Given a CXR image + claim, asks CheXagent whether the finding is present.
Compares this vision-language approach to our text-only verifier.

Uses OpenI images (public, 3.5GB) stored on Modal volume.

Usage:
    modal run --detach scripts/modal_chexagent_baseline.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-chexagent-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers==4.40.0",  # Pinned — CheXagent-8b requires this version
        "accelerate>=0.27.0",
        "einops>=0.7.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
    )
    .apt_install("wget")
    .run_commands(
        "mkdir -p /openi_images",
        "wget -q -O /tmp/openi.tgz https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
        "tar xzf /tmp/openi.tgz -C /openi_images",
        "rm /tmp/openi.tgz",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/data": vol},
)
def run_chexagent_baseline(
    claims_path: str = "/data/eval_data_openi/test_claims.json",
    images_dir: str = "/openi_images",  # Baked into container image
    output_path: str = "/data/eval_results_chexagent/baseline_results.json",
) -> dict:
    """Run CheXagent-8b VQA baseline on OpenI test claims."""
    import json
    import os
    import re
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, roc_auc_score,
    )
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Load claims
    print(f"Loading claims from {claims_path}")
    with open(claims_path) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} claims")

    # Map patient_id -> image file (OpenI convention: CXR{id}_1_IM-{num}-{num}.png)
    img_dir = Path(images_dir)
    all_images = {}
    if img_dir.exists():
        for p in img_dir.glob("*.png"):
            # Extract CXR ID: CXR1234_1_IM-0001-1001.png -> CXR1234
            m = re.match(r"(CXR\d+)", p.name)
            if m:
                cxr_id = m.group(1)
                if cxr_id not in all_images:
                    all_images[cxr_id] = str(p)
    print(f"Found {len(all_images)} unique CXR images")

    # Load CheXagent-8b
    print("Loading CheXagent-8b...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "StanfordAIMI/CheXagent-8b"
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"CheXagent-8b loaded successfully, type={type(model).__name__}", flush=True)
        use_chexagent = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load CheXagent-8b: {e}", flush=True)
        use_chexagent = False

    # VQA inference
    results = []
    n_no_image = 0
    n_errors = 0

    for i, claim in enumerate(tqdm(claims, desc="CheXagent inference")):
        pid = str(claim["patient_id"])
        claim_text = claim["claim"]
        true_label = 1 if claim["label"] == 1 else 0  # Binary remap

        # Find image
        img_path = all_images.get(pid)
        if not img_path or not Path(img_path).exists():
            n_no_image += 1
            results.append({"true": true_label, "pred": 0, "score": 0.5, "error": "no_image"})
            continue

        if not use_chexagent:
            results.append({"true": true_label, "pred": 0, "score": 0.5, "error": "model_load_failed"})
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            prompt = (
                f"Looking at this chest X-ray, is the following finding present? "
                f"Answer with only YES or NO.\n\n"
                f"Finding: {claim_text}"
            )

            inputs = processor(
                images=img,
                text=prompt,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )

            response = processor.decode(outputs[0], skip_special_tokens=True).strip()

            # Parse YES/NO
            resp_lower = response.lower()
            if "yes" in resp_lower and "no" not in resp_lower:
                pred = 0  # Finding present → not contradicted
                score = 0.2  # Low contra score
            elif "no" in resp_lower:
                pred = 1  # Finding absent → contradicted
                score = 0.8  # High contra score
            else:
                pred = 0  # Default: not contradicted
                score = 0.5

            results.append({
                "true": true_label, "pred": pred,
                "score": score, "error": None,
                "response": response[:100],
            })

        except Exception as e:
            n_errors += 1
            results.append({"true": true_label, "pred": 0, "score": 0.5, "error": str(e)[:100]})

        if (i + 1) % 100 == 0:
            n_ok = sum(1 for r in results if r["error"] is None)
            print(f"  {i+1}/{len(claims)}: {n_ok} OK, {n_no_image} no_image, {n_errors} errors")

    # Compute metrics
    labels = np.array([r["true"] for r in results])
    preds = np.array([r["pred"] for r in results])
    not_contra_scores = 1.0 - np.array([r["score"] for r in results])
    n_ok = sum(1 for r in results if r["error"] is None)
    n_err = len(results) - n_ok

    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_f1 = f1_score(labels, preds, average=None, zero_division=0).tolist()
    prec = precision_score(labels, preds, average=None, zero_division=0).tolist()
    rec = recall_score(labels, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()

    try:
        auroc = roc_auc_score((labels == 0).astype(int), not_contra_scores)
    except ValueError:
        auroc = float("nan")

    print(f"\n=== CheXagent-8b Zero-Shot VQA Baseline ===")
    print(f"  n_claims={len(claims)}, n_with_image={n_ok}, n_no_image={n_no_image}, n_errors={n_errors}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Macro F1:      {mf1:.4f}")
    print(f"  Contra F1:     {per_f1[1]:.4f}")
    print(f"  Contra Prec:   {prec[1]:.4f}")
    print(f"  Contra Recall: {rec[1]:.4f}")
    print(f"  AUROC:         {auroc:.4f}")
    print(f"  Confusion:     {cm}")

    output = {
        "name": "CheXagent-8b (zero-shot VQA, OpenI images)",
        "model": "StanfordAIMI/CheXagent-8b",
        "n_claims": len(claims),
        "n_with_image": n_ok,
        "n_no_image": n_no_image,
        "n_errors": n_errors,
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "f1_notcontra": float(per_f1[0]),
        "f1_contra": float(per_f1[1]),
        "precision_contra": float(prec[1]),
        "recall_contra": float(rec[1]),
        "auroc": float(auroc),
        "confusion_matrix": cm,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    vol.commit()
    return output


@app.local_entrypoint()
def main():
    print("Running CheXagent-8b baseline on OpenI...")
    result = run_chexagent_baseline.remote()
    print(f"\nAccuracy: {result['accuracy']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print(f"Contra F1: {result['f1_contra']:.4f}")
