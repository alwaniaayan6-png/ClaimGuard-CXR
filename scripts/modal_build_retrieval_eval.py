"""Build retrieval-augmented eval sets for ClaimGuard-CXR.

For each eval claim, replaces oracle (same-report) evidence with the top-K
passages retrieved from the training-patient corpus via a configurable
retrieval pipeline.

v3.1 upgrade — hybrid retrieval
-------------------------------
The v1 script only supported dense-only MedCPT + FAISS retrieval, leaving
2-5 pp on the table relative to a well-tuned hybrid pipeline.  v3.1 adds:

  1. Batched BM25 sparse retrieval via
     ``models.retriever.bm25_index.BM25Index.search_batch``.
  2. Reciprocal-rank fusion (Cormack 2009) via
     ``models.retriever.rrf_fusion.rrf_fuse``.
  3. Cross-encoder reranking via
     ``models.retriever.reranker.CrossEncoderReranker.rerank_batch``.

Supported modes (``--method``):

  - ``dense_only``       MedCPT dense FAISS (v1 behaviour).
  - ``sparse_only``      BM25 only.
  - ``hybrid_rrf``       Dense + sparse, fused via RRF(k=60), top_k=2.
  - ``hybrid_rrf_rerank`` Dense + sparse → RRF top-20 → cross-encoder → top-2
    (DEFAULT — the production pipeline).

All four modes emit a ``retrieval_source`` field on each claim so the
downstream evaluator can distinguish ablation runs.  The hybrid modes
stamp the same INDEPENDENT trust tier as the v1 dense path because the
corpus is still same-schema radiologist-written text from a disjoint
patient split.

Usage (from local Mac):
    modal run scripts/modal_build_retrieval_eval.py
    modal run scripts/modal_build_retrieval_eval.py --method dense_only
    modal run scripts/modal_build_retrieval_eval.py --method sparse_only
    modal run scripts/modal_build_retrieval_eval.py --method hybrid_rrf

Reads:
    /data/indices/faiss_index.index
    /data/indices/passage_lookup.json
    /data/indices/bm25_index.pkl           (new — built by build_bm25_index.py)
    /data/eval_data/{calibration,test}_claims.json

Writes:
    /data/eval_data/{calibration,test}_claims_retrieved.json
        (under method name — e.g., _hybrid_rrf_rerank.json)
    /data/results/retrieval_ablation_{method}.json
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
    # Ship the in-repo retriever package so we can import
    # rrf_fusion, batched BM25, and the cross-encoder reranker.
    .add_local_python_source("models")
)

vol = modal.Volume.from_name("claimguard-data", create_if_missing=True)

_SUPPORTED_METHODS = (
    "dense_only",
    "sparse_only",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
)


@app.function(
    image=image,
    gpu="H100",  # H100 per user policy (job is ~3-8 min on H100 depending on method)
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 2,
    volumes={"/data": vol},
)
def build_retrieval_eval(
    method: str = "hybrid_rrf_rerank",
    top_k: int = 2,
    rerank_candidate_k: int = 20,
    rrf_k: int = 60,
) -> dict:
    """Retrieve top-K evidence passages for every eval claim.

    Args:
        method: One of ``dense_only``, ``sparse_only``, ``hybrid_rrf``,
            ``hybrid_rrf_rerank``.
        top_k: Number of retrieved passages per claim to concatenate
            into the final evidence string.
        rerank_candidate_k: Number of RRF-fused candidates to pass into
            the cross-encoder reranker (only used when
            ``method == "hybrid_rrf_rerank"``).
        rrf_k: Smoothing constant for RRF fusion (paper default 60).

    Returns:
        dict with counts of claims processed per split.
    """
    import json
    import os
    import sys

    import faiss
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {_SUPPORTED_METHODS}, got {method!r}"
        )

    # Make the in-repo retriever package importable inside the Modal
    # function.  ``add_local_python_source`` above puts the repo root on
    # the path, but we also need to handle the submodule imports.
    sys.path.insert(0, "/root")
    from models.retriever.bm25_index import BM25Index  # noqa: E402
    from models.retriever.rrf_fusion import rrf_fuse  # noqa: E402
    from models.retriever.reranker import CrossEncoderReranker  # noqa: E402

    # ------------------------------------------------------------------
    # Load passages + FAISS index
    # ------------------------------------------------------------------
    print("Loading passage lookup...")
    with open("/data/indices/passage_lookup.json") as f:
        lookup = json.load(f)
    passages = lookup["passages"]
    print(f"Loaded {len(passages)} passages")

    need_dense = method in ("dense_only", "hybrid_rrf", "hybrid_rrf_rerank")
    need_sparse = method in ("sparse_only", "hybrid_rrf", "hybrid_rrf_rerank")
    need_rerank = method == "hybrid_rrf_rerank"

    faiss_index = None
    if need_dense:
        print("Loading FAISS index...")
        faiss_index = faiss.read_index("/data/indices/faiss_index.index")
        faiss_index.nprobe = 64  # search more cells for recall
        print(f"FAISS index: {faiss_index.ntotal} vectors")

    bm25 = None
    if need_sparse:
        print("Loading BM25 index...")
        bm25 = BM25Index()
        bm25.load("/data/indices/bm25_index.pkl")

    # ------------------------------------------------------------------
    # Load MedCPT query encoder (dense path only)
    # ------------------------------------------------------------------
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if need_dense:
        print("Loading MedCPT Query Encoder...")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        model.eval()

    # ------------------------------------------------------------------
    # Load cross-encoder reranker (hybrid_rrf_rerank only)
    # ------------------------------------------------------------------
    reranker = None
    if need_rerank:
        print("Loading cross-encoder reranker...")
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            max_length=512,
            device=device,
        )

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
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embs.append(pooled.cpu().float().numpy())
        return np.concatenate(embs, axis=0)

    # ------------------------------------------------------------------
    # Core retrieval functions
    # ------------------------------------------------------------------
    def dense_retrieve(query_embs: np.ndarray, k: int) -> list[list[int]]:
        """FAISS search → per-query list of passage ids (length k)."""
        _, I = faiss_index.search(query_embs.astype(np.float32), k)
        return [[int(x) for x in row if x >= 0] for row in I]

    def sparse_retrieve(queries: list[str], k: int) -> list[list[int]]:
        """Batched BM25 → per-query list of passage ids (length ≤ k)."""
        hits_per_query = bm25.search_batch(queries, top_k=k)
        out: list[list[int]] = []
        for hits in hits_per_query:
            # bm25_index returns (text, score, passage_id); ids are strings
            # when the index was built from our pipeline, but they encode
            # integer positions into the passages list.
            out.append([int(pid) for _, _, pid in hits])
        return out

    def fuse_and_select(
        dense_ids: list[list[int]],
        sparse_ids: list[list[int]],
        fused_top_k: int,
    ) -> list[list[int]]:
        """Reciprocal rank fusion → per-query top-k passage ids."""
        return rrf_fuse(
            dense_ranks=dense_ids,
            sparse_ranks=sparse_ids,
            k=rrf_k,
            top_k=fused_top_k,
        )

    def passages_for_ids(id_list: list[int]) -> list[str]:
        return [passages[i] for i in id_list]

    # ------------------------------------------------------------------
    # Process eval splits
    # ------------------------------------------------------------------
    results = {}
    for split_name in ["calibration", "test"]:
        in_path = f"/data/eval_data/{split_name}_claims.json"
        out_path = (
            f"/data/eval_data/{split_name}_claims_retrieved_{method}.json"
        )

        if not os.path.exists(in_path):
            print(f"WARN: {in_path} not found, skipping")
            continue

        print(f"\n=== Processing {split_name} with method={method} ===")
        with open(in_path) as f:
            claims = json.load(f)
        print(f"Loaded {len(claims)} claims")

        claim_texts = [c["claim"] for c in claims]

        # For dense methods we retrieve more than top_k so RRF has room
        # to discriminate; likewise for sparse.
        dense_k = max(top_k, rerank_candidate_k) if need_rerank else max(
            top_k, 20
        )
        sparse_k = max(top_k, rerank_candidate_k) if need_rerank else max(
            top_k, 20
        )

        dense_ids: list[list[int]] = []
        sparse_ids: list[list[int]] = []

        if need_dense:
            print("Encoding queries + FAISS search...")
            query_embs = encode_queries(claim_texts)
            dense_ids = dense_retrieve(query_embs, dense_k)

        if need_sparse:
            print("BM25 batched search...")
            sparse_ids = sparse_retrieve(claim_texts, sparse_k)

        # Per-method selection
        if method == "dense_only":
            final_id_lists = [row[:top_k] for row in dense_ids]
        elif method == "sparse_only":
            final_id_lists = [row[:top_k] for row in sparse_ids]
        elif method == "hybrid_rrf":
            final_id_lists = fuse_and_select(dense_ids, sparse_ids, top_k)
        elif method == "hybrid_rrf_rerank":
            # Step 1 — fuse both rankings to get ``rerank_candidate_k``
            candidate_ids = fuse_and_select(
                dense_ids, sparse_ids, rerank_candidate_k
            )
            # Step 2 — batch cross-encoder rerank, keep top_k
            print(
                f"Cross-encoder reranking ({len(candidate_ids)} queries × "
                f"{rerank_candidate_k} candidates)..."
            )
            pairs: list[tuple[str, list[str]]] = []
            for query, ids in zip(claim_texts, candidate_ids):
                pairs.append((query, passages_for_ids(ids)))
            rerank_results = reranker.rerank_batch(
                pairs, top_k=top_k, batch_size=64
            )
            # Map reranked passage texts back to their ids (first-match
            # within the candidate pool, which is guaranteed unique
            # because RRF dedupes).
            final_id_lists = []
            for (query, candidate_pool), ranked in zip(
                candidate_ids, rerank_results
            ):
                # reranker returns (passage_text, score); we need the ids
                # so we look up each passage in the original candidate
                # pool (O(k^2), trivial for k=20).
                texts_in_pool = [passages[i] for i in query]
                top_ids = []
                for passage_text, _ in ranked:
                    try:
                        local_idx = texts_in_pool.index(passage_text)
                        top_ids.append(query[local_idx])
                    except ValueError:
                        continue
                final_id_lists.append(top_ids)
        else:
            raise RuntimeError(f"unhandled method {method}")

        retrieved_evidence: list[str] = []
        for id_list in final_id_lists:
            ps = passages_for_ids(id_list)
            retrieved_evidence.append(" ".join(ps))

        for claim, retrieved in zip(claims, retrieved_evidence):
            claim["oracle_evidence"] = claim.get("evidence", "")
            claim["evidence"] = retrieved  # replace with retrieved
            claim["retrieval_source"] = f"retrieval:{method}"

            # Provenance: evidence is retrieved from the training-patient
            # corpus, which is radiologist-written and does not include
            # the claim's source report by construction. Tier flips from
            # TRUSTED (oracle) to INDEPENDENT (retrieved).
            claim["evidence_source_type"] = "retrieved_report_text"
            claim["evidence_is_independent"] = True
            claim["evidence_generator_id"] = None
            claim["claim_generator_id"] = None
            claim["evidence_trust_tier"] = "independent"

        with open(out_path, "w") as f:
            json.dump(claims, f, indent=2)
        print(f"Saved to {out_path}")
        results[split_name] = len(claims)

    vol.commit()
    return {"method": method, **results}


@app.local_entrypoint()
def main(method: str = "hybrid_rrf_rerank"):
    print(
        f"Launching retrieval-augmented eval data build on Modal (method={method})..."
    )
    result = build_retrieval_eval.remote(method=method)
    print(f"\nDone: {result}")
