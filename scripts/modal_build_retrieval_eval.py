"""Build retrieval-augmented eval sets for ClaimGuard-CXR.

For each eval claim, replaces oracle (same-report) evidence with the top-K
passages retrieved from the training-patient FAISS dense index using
MedCPT embeddings.

This produces the **retrieval-augmented** eval set used in the paper's
end-to-end ablation. The oracle eval set (pre-computed evidence from the
same report) is retained as an upper-bound reference in the main table.

Retrieval design:
    * Dense-only (MedCPT + FAISS IVF-Flat, inner-product on L2-normalized vectors)
    * BM25 sparse retrieval skipped in the per-query loop — rank_bm25 requires
      O(corpus) per query (~2s for 1.2M passages), infeasible at 30K claims.
      BM25 index is retained for future batched re-scoring / reranking.
    * Top-K=2 passages concatenated as evidence.

Usage (from local Mac):
    modal run scripts/modal_build_retrieval_eval.py

Reads:
    /data/indices/faiss_index.index
    /data/indices/passage_lookup.json
    /data/eval_data/{calibration,test}_claims.json

Writes:
    /data/eval_data/calibration_claims_retrieved.json
    /data/eval_data/test_claims_retrieved.json
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-retrieval-eval")

image = (
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
    image=image,
    gpu="H100",  # H100 per user policy (job is ~2min on H100, ~2min on T4 — either works, H100 for consistency)
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def build_retrieval_eval(top_k: int = 2) -> dict:
    """Retrieve top-K evidence passages for every eval claim (dense-only).

    Args:
        top_k: Number of retrieved passages per claim to concatenate.

    Returns:
        dict with counts of claims processed per split.
    """
    import json
    import os

    import faiss
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    # ------------------------------------------------------------------
    # Load passages + FAISS index
    # ------------------------------------------------------------------
    print("Loading passage lookup...")
    with open("/data/indices/passage_lookup.json") as f:
        lookup = json.load(f)
    passages = lookup["passages"]
    print(f"Loaded {len(passages)} passages")

    print("Loading FAISS index...")
    faiss_index = faiss.read_index("/data/indices/faiss_index.index")
    faiss_index.nprobe = 64  # search more cells for recall
    print(f"FAISS index: {faiss_index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Load MedCPT query encoder
    # ------------------------------------------------------------------
    print("Loading MedCPT Query Encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    model.eval()

    @torch.no_grad()
    def encode_queries(texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode claims into 768-d L2-normalized vectors."""
        embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embs.append(pooled.cpu().float().numpy())
        return np.concatenate(embs, axis=0)

    # ------------------------------------------------------------------
    # Dense-only batched retrieval
    # ------------------------------------------------------------------
    def batch_retrieve(query_embs: np.ndarray) -> list[str]:
        """Batch-search FAISS and return concatenated top-K passages per query."""
        D, I = faiss_index.search(query_embs.astype(np.float32), top_k)
        evidence_strs: list[str] = []
        for row in I:
            ps = [passages[int(doc_id)] for doc_id in row if doc_id >= 0]
            evidence_strs.append(" ".join(ps))
        return evidence_strs

    # ------------------------------------------------------------------
    # Process eval splits
    # ------------------------------------------------------------------
    results = {}
    for split_name in ["calibration", "test"]:
        in_path = f"/data/eval_data/{split_name}_claims.json"
        out_path = f"/data/eval_data/{split_name}_claims_retrieved.json"

        if not os.path.exists(in_path):
            print(f"WARN: {in_path} not found, skipping")
            continue

        print(f"\n=== Processing {split_name} ===")
        with open(in_path) as f:
            claims = json.load(f)
        print(f"Loaded {len(claims)} claims")

        # Batch-encode all claim texts first
        claim_texts = [c["claim"] for c in claims]
        print("Encoding queries...")
        query_embs = encode_queries(claim_texts)

        # Batched retrieve (FAISS handles batched search efficiently)
        print("Retrieving evidence (batched FAISS search)...")
        retrieved_evidence = batch_retrieve(query_embs)

        for claim, retrieved in zip(claims, retrieved_evidence):
            claim["oracle_evidence"] = claim.get("evidence", "")
            claim["evidence"] = retrieved  # replace with retrieved
            claim["retrieval_source"] = "dense_medcpt_faiss_ivf"

        with open(out_path, "w") as f:
            json.dump(claims, f, indent=2)
        print(f"Saved to {out_path}")
        results[split_name] = len(claims)

    vol.commit()
    return results


@app.local_entrypoint()
def main():
    print("Launching retrieval-augmented eval data build on Modal...")
    result = build_retrieval_eval.remote()
    print(f"\nDone: {result}")
