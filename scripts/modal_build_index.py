"""Modal script for building retrieval indices (FAISS + BM25).

Usage (from local Mac):
    modal run scripts/modal_build_index.py

Builds dense (MedCPT + FAISS) and sparse (BM25) retrieval indices
from 1.2M training passages. Runs on CPU — no GPU needed, but
benefits from Modal's faster network for model downloads.
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-build-index")

index_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "faiss-cpu>=1.7.4",
        "rank-bm25>=0.2.2",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    )
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)


@app.function(
    image=index_image,
    gpu="H100",  # H100 per user policy — T4 encoding took ~25min, H100 would be ~3-4min
    cpu=4,
    memory=32768,  # 32GB RAM — needed for 1.2M passages + embeddings
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def build_indices(
    passages_path: str = "/data/retrieval_passages.json",
    output_dir: str = "/data/indices",
) -> dict:
    """Build FAISS + BM25 indices from pre-extracted passages.

    Returns dict with index paths and statistics.
    """
    import json
    import os
    import pickle

    import faiss
    import numpy as np
    import torch
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load passages
    print(f"Loading passages from {passages_path}...")
    with open(passages_path, "r") as f:
        data = json.load(f)
    passages = data["passages"]
    passage_ids = data["passage_ids"]
    print(f"Loaded {len(passages)} passages")

    # =========================================================
    # 1. BM25 sparse index (fast, ~2 min)
    # =========================================================
    print("\n=== Building BM25 index ===")
    tokenized = [p.lower().split() for p in tqdm(passages, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    bm25_path = os.path.join(output_dir, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "passages": passages,
            "passage_ids": passage_ids,
        }, f)
    print(f"Saved BM25 index to {bm25_path}")
    vol.commit()

    # =========================================================
    # 2. MedCPT dense embeddings (slow, ~30-60 min on CPU)
    # =========================================================
    print("\n=== Building dense FAISS index ===")
    print("Loading MedCPT Article Encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    model.eval()

    batch_size = 256 if device == "cuda" else 128  # GPU can handle larger batches
    all_embeddings = []

    print(f"Encoding {len(passages)} passages (batch_size={batch_size})...")
    for start in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
        batch = passages[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save raw embeddings (for later use)
    emb_path = os.path.join(output_dir, "passage_embeddings.npy")
    np.save(emb_path, embeddings)
    vol.commit()

    # Build FAISS IVF index
    n, dim = embeddings.shape
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # IVF-Flat with inner product (cosine similarity since L2-normalized)
    nlist = min(2048, n // 50)  # Voronoi cells
    print(f"Building IVF-Flat index: {n} vectors, dim={dim}, nlist={nlist}")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    print("Training index...")
    index.train(embeddings)
    print("Adding vectors...")
    index.add(embeddings)
    index.nprobe = 32  # search 32 cells at query time

    faiss_path = os.path.join(output_dir, "faiss_index.index")
    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index to {faiss_path}")

    # Save passage lookup
    lookup_path = os.path.join(output_dir, "passage_lookup.json")
    with open(lookup_path, "w") as f:
        json.dump({"passages": passages, "passage_ids": passage_ids}, f)

    vol.commit()

    print(f"\n=== Index building complete ===")
    print(f"BM25: {bm25_path}")
    print(f"FAISS: {faiss_path} ({n} vectors, dim={dim})")
    print(f"Passages: {lookup_path}")

    return {
        "n_passages": len(passages),
        "embedding_dim": dim,
        "faiss_path": faiss_path,
        "bm25_path": bm25_path,
    }


@app.local_entrypoint()
def main():
    """Entry point for `modal run scripts/modal_build_index.py`."""
    print("Launching index building on Modal...")
    result = build_indices.remote()
    print(f"\nIndex building complete!")
    print(f"Passages indexed: {result['n_passages']}")
    print(f"Embedding dim: {result['embedding_dim']}")
