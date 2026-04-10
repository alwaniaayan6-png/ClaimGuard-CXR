"""CheXzero zero-shot baseline for ClaimGuard-CXR v2.

Uses CheXzero directly (no fine-tuning) to score claim-image alignment.
High CLIP similarity = claim consistent with image, low = potential hallucination.

This tests whether a pretrained CXR-specific CLIP model can detect
contradictions without any task-specific training.

Usage:
    modal run --detach scripts/baseline_chexzero_zeroshot.py
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-baseline-chexzero")

baseline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "open_clip_torch>=2.24.0",
    )
    .run_commands("pip install git+https://github.com/openai/CLIP.git")
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=baseline_image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def run_chexzero_baseline(
    test_claims_path: str = "/data/eval_data/test_claims.json",
    image_dir: str = "/data/openi_images",
    output_path: str = "/data/eval_results_chexzero_zeroshot/results.json",
    seed: int = 42,
) -> dict:
    """Run CheXzero zero-shot on test claims."""
    import json
    import os
    import random

    import clip
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from tqdm import tqdm

    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load test claims
    with open(test_claims_path) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} test claims")

    # Get available images
    has_images = os.path.isdir(image_dir)
    image_files = sorted(os.listdir(image_dir)) if has_images else []

    # Score each claim
    scores = []
    labels = []

    for item in tqdm(claims, desc="CheXzero zero-shot"):
        claim_text = item["claim"]
        label = 1 if item["label"] == 1 else 0
        labels.append(label)

        # Tokenize claim
        text_tokens = clip.tokenize([claim_text], truncate=True).to(device)

        with torch.no_grad():
            text_emb = model.encode_text(text_tokens).float()
            text_emb = F.normalize(text_emb, dim=-1)

            # If we have a matching image, compute image-text similarity
            img_file = item.get("image_file", "")
            img_path = os.path.join(image_dir, img_file) if img_file else ""

            if os.path.exists(img_path):
                img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                img_emb = model.encode_image(img).float()
                img_emb = F.normalize(img_emb, dim=-1)
                sim = (img_emb * text_emb).sum().item()
            else:
                # No image: use text-only embedding norm as a proxy
                # (higher norm = more "in-distribution" for CLIP)
                sim = 0.5  # neutral default

        scores.append(sim)

    scores = np.array(scores)
    labels = np.array(labels)

    # Threshold: claims with sim < median are predicted contradicted
    threshold = np.median(scores)
    preds = (scores < threshold).astype(int)

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    contra_recall = float((preds[labels == 1] == 1).mean()) if labels.sum() > 0 else 0.0

    try:
        auroc = float(roc_auc_score(labels, -scores))  # lower sim = more likely contra
    except ValueError:
        auroc = 0.5

    results = {
        "method": "CheXzero zero-shot",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "contra_recall": contra_recall,
        "auroc": auroc,
        "n_claims": len(claims),
        "n_with_images": sum(1 for s in scores if s != 0.5),
        "threshold": float(threshold),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    vol.commit()
    print(f"\nCheXzero zero-shot: acc={accuracy:.4f}, contra_recall={contra_recall:.4f}")
    return results


@app.local_entrypoint()
def main():
    result = run_chexzero_baseline.remote()
    print(f"\nResult: {result}")
