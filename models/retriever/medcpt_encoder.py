"""
MedCPT-based dense retriever for evidence passage retrieval in ClaimGuard-CXR.

Provides:
- MedCPTRetriever  — dual-encoder dense retrieval via NCBI MedCPT models.
- HybridRetriever  — fuses dense (FAISS) and sparse (BM25) results with RRF,
                     then optionally reranks the top passages with a
                     cross-encoder.
- build_faiss_index — builds an IVF-PQ FAISS index from a passage embedding
                      matrix for fast approximate nearest-neighbour search.
- build_bm25_index  — builds a BM25Okapi index from a list of raw passage
                      strings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import faiss  # faiss-cpu
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required for HybridRetriever / build_faiss_index. "
        "Install via: pip install faiss-cpu"
    ) from exc

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise ImportError(
        "rank_bm25 is required for build_bm25_index. "
        "Install via: pip install rank-bm25"
    ) from exc


# ---------------------------------------------------------------------------
# Dense encoder
# ---------------------------------------------------------------------------

class MedCPTRetriever:
    """Dual-encoder retriever using the NCBI MedCPT model family.

    MedCPT ships two separate fine-tuned encoders — one for short queries
    (claims) and one for longer article passages.  This class wraps both so
    that query and passage embeddings live in a shared representation space.

    Args:
        query_model_id: HuggingFace model ID for the query encoder.
        article_model_id: HuggingFace model ID for the passage encoder.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
            Defaults to CUDA if available, else CPU.
        batch_size: Number of texts encoded per forward pass.
        max_length: Tokeniser truncation length in tokens.
    """

    def __init__(
        self,
        query_model_id: str = "ncbi/MedCPT-Query-Encoder",
        article_model_id: str = "ncbi/MedCPT-Article-Encoder",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
        self.query_model = AutoModel.from_pretrained(query_model_id).to(self.device)
        self.query_model.eval()

        self.article_tokenizer = AutoTokenizer.from_pretrained(article_model_id)
        self.article_model = AutoModel.from_pretrained(article_model_id).to(self.device)
        self.article_model.eval()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool token embeddings, ignoring padding tokens.

        Args:
            last_hidden_state: Shape (B, L, H).
            attention_mask: Shape (B, L), 1 for real tokens 0 for padding.

        Returns:
            Pooled embeddings of shape (B, H).
        """
        mask = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)        # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        return summed / counts

    def _encode(
        self,
        texts: list[str],
        tokenizer: Any,
        model: Any,
        desc: str,
    ) -> np.ndarray:
        """Batch-encode texts, returning L2-normalised embeddings.

        Args:
            texts: Raw text strings to encode.
            tokenizer: HuggingFace tokenizer.
            model: HuggingFace model with ``last_hidden_state`` output.
            desc: Label for the tqdm progress bar.

        Returns:
            Float32 numpy array of shape (n, 768).
        """
        all_embeddings: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc=desc,
            unit="batch",
        ):
            batch_texts = texts[start : start + self.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)

            pooled = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            # L2 normalise so that dot-product == cosine similarity
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_embeddings.append(pooled.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode claim/query strings with the MedCPT query encoder.

        Args:
            texts: List of query strings (typically short clinical claims).

        Returns:
            L2-normalised embeddings of shape (n, 768), dtype float32.
        """
        return self._encode(
            texts,
            self.query_tokenizer,
            self.query_model,
            desc="Encoding queries",
        )

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode article/passage strings with the MedCPT article encoder.

        Args:
            texts: List of passage strings (PubMed abstracts, report
                sentences, etc.).

        Returns:
            L2-normalised embeddings of shape (n, 768), dtype float32.
        """
        return self._encode(
            texts,
            self.article_tokenizer,
            self.article_model,
            desc="Encoding passages",
        )


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combines dense FAISS retrieval with BM25 sparse retrieval via RRF.

    Reciprocal Rank Fusion (RRF) with k=60 merges ranked lists from both
    retrievers without requiring score calibration.  An optional cross-encoder
    reranker can then rescore the fused top-k set.

    Args:
        dense_retriever: An initialised :class:`MedCPTRetriever`.
        bm25_index: A ``rank_bm25.BM25Okapi`` index over the passage corpus.
        faiss_index: A FAISS index with passage embeddings already added.
        passages: The raw passage strings corresponding to index positions.
        rrf_k: The RRF constant k (default 60, per the original paper).
    """

    _RRF_K: int = 60

    def __init__(
        self,
        dense_retriever: MedCPTRetriever,
        bm25_index: BM25Okapi,
        faiss_index: faiss.Index,
        passages: list[str],
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25 = bm25_index
        self.faiss_index = faiss_index
        self.passages = passages
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dense_retrieve(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Return (doc_id, distance) pairs from FAISS.

        Args:
            query_embedding: Shape (1, 768).
            top_k: Number of nearest neighbours to retrieve.

        Returns:
            List of (doc_id, distance) sorted by ascending distance
            (closer = more relevant).
        """
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return list(zip(indices[0].tolist(), distances[0].tolist()))

    def _sparse_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Return (doc_id, bm25_score) pairs from BM25Okapi.

        Args:
            query: Raw query string.
            top_k: Number of top passages to return.

        Returns:
            List of (doc_id, score) sorted by descending BM25 score.
        """
        tokens = query.lower().split()
        scores: np.ndarray = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        """Compute Reciprocal Rank Fusion score for a single rank.

        Args:
            rank: 1-based rank position (1 = best).
            k: RRF constant (default 60).

        Returns:
            RRF score 1 / (k + rank).
        """
        return 1.0 / (k + rank)

    def _fuse(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Fuse dense and sparse ranked lists with RRF.

        Args:
            dense_results: List of (doc_id, score) from FAISS, ordered best
                to worst.
            sparse_results: List of (doc_id, score) from BM25, ordered best
                to worst.

        Returns:
            Combined list of (doc_id, rrf_score) sorted by descending
            RRF score.
        """
        rrf_scores: dict[int, float] = {}

        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(
                rank, self.rrf_k
            )

        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(
                rank, self.rrf_k
            )

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Retrieve evidence passages for a single query using dense + sparse fusion.

        Encodes the query with MedCPT, retrieves top-k candidates from both
        FAISS and BM25, then fuses the ranked lists with RRF.

        Args:
            query: The claim or question string to retrieve evidence for.
            top_k: Number of passages to return after fusion (default 20).

        Returns:
            List of (passage_text, rrf_score) tuples sorted by descending
            score, length ``top_k``.
        """
        query_embedding = self.dense_retriever.encode_queries([query])  # (1, 768)

        dense_results = self._dense_retrieve(query_embedding, top_k=top_k)
        sparse_results = self._sparse_retrieve(query, top_k=top_k)

        fused = self._fuse(dense_results, sparse_results)[:top_k]

        return [(self.passages[doc_id], score) for doc_id, score in fused]

    def rerank(
        self,
        query: str,
        passages: list[str],
        reranker_model: Any,
        top_n: int = 2,
    ) -> list[tuple[str, float]]:
        """Rerank candidate passages with a cross-encoder and return the top-n.

        The cross-encoder receives (query, passage) pairs and returns a
        relevance logit for each pair.  This method selects the top-n
        highest-scoring passages.

        Args:
            query: The query/claim string.
            passages: Candidate passage strings to rerank (typically the
                output of :meth:`retrieve`).
            reranker_model: A cross-encoder that exposes a
                ``predict(pairs: list[tuple[str, str]]) -> np.ndarray``
                interface (e.g. a ``sentence_transformers.CrossEncoder``).
            top_n: Number of passages to return after reranking (default 2).

        Returns:
            List of (passage_text, score) tuples of length ``top_n``, sorted
            by descending cross-encoder score.
        """
        if not passages:
            return []

        pairs = [(query, passage) for passage in passages]
        scores: np.ndarray = np.asarray(reranker_model.predict(pairs))

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(passages[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexIVFPQ:
    """Build an IVF-PQ FAISS index from a passage embedding matrix.

    Uses ``IndexIVFPQ`` with nlist=1024 Voronoi cells and m=64 PQ
    sub-quantisers for a good trade-off between recall and memory.  The index
    is trained on the full embedding set before vectors are added.

    Args:
        embeddings: Float32 numpy array of shape (n, 768) containing
            L2-normalised passage embeddings from :class:`MedCPTRetriever`.

    Returns:
        A trained and populated ``faiss.IndexIVFPQ`` ready for ``.search()``.

    Raises:
        ValueError: If ``embeddings`` has fewer rows than ``nlist`` (1024),
            which would make IVF training degenerate.
    """
    n, dim = embeddings.shape
    nlist = 1024
    m = 64        # number of PQ sub-quantisers (dim must be divisible by m)
    nbits = 8     # bits per sub-quantiser code

    if n < nlist:
        raise ValueError(
            f"Need at least {nlist} passages to train IVF index, got {n}. "
            "Either add more passages or reduce nlist."
        )
    if dim % m != 0:
        raise ValueError(
            f"Embedding dimension ({dim}) must be divisible by m ({m}) for PQ."
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    quantizer = faiss.IndexFlatIP(dim)          # inner-product coarse quantiser
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    index.train(embeddings)
    index.add(embeddings)
    return index


def build_bm25_index(passages: list[str]) -> BM25Okapi:
    """Build a BM25Okapi sparse index from a list of raw passage strings.

    Tokenises by lowercasing and whitespace-splitting.  For production use
    consider replacing with a proper tokeniser (e.g. NLTK WordPunct).

    Args:
        passages: List of raw passage strings forming the retrieval corpus.

    Returns:
        A ``rank_bm25.BM25Okapi`` instance ready for ``.get_scores()``.
    """
    tokenised = [p.lower().split() for p in passages]
    return BM25Okapi(tokenised)
