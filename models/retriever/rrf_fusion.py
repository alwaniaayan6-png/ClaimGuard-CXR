"""Reciprocal Rank Fusion for hybrid (dense + sparse) retrieval.

Given per-query ranked lists from two (or more) retrievers, combine them
into a single fused ranking using the RRF formula from Cormack et al.
(2009, "Reciprocal Rank Fusion outperforms Condorcet and Individual
Rank Learning Methods"):

    score(doc) = Σ_{r ∈ retrievers}   1 / (k + rank_r(doc))

where ``k`` is a smoothing constant (paper-recommended 60) and
``rank_r(doc)`` is the 1-indexed rank of ``doc`` under retriever ``r``.
Documents not in a retriever's top-K contribute 0 from that retriever.

This module is retriever-agnostic — the inputs are integer doc-id lists
so the caller can plug in BM25, dense FAISS, cross-encoder, or
anything else that produces a ranking.

Design notes
------------
* RRF is famously robust across retriever-score scales because it
  only uses ranks, not raw scores.  Numerically, it beats linear
  interpolation of scores whenever the two retrievers differ in their
  score distributions (always true for BM25 vs. dense cosine).
* k=60 is the paper default.  Larger k makes the fusion flatter
  (closer to a uniform committee vote); smaller k makes top results
  dominate more.  We expose ``k`` as a parameter for ablations.
* We return the *fused ranking* (list of doc-ids sorted by fused
  score, descending) so callers can truncate to any top-K after the
  fact.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def rrf_fuse_single(
    *ranked_lists: Sequence[int],
    k: int = 60,
    top_k: int | None = None,
) -> list[int]:
    """Fuse multiple single-query ranked lists into one.

    Args:
        *ranked_lists: Two or more sequences of doc-ids, each ordered
            by descending relevance under its respective retriever.
            Doc-ids are assumed to be unique within a single list.
        k: Smoothing constant in the RRF formula ``1 / (k + rank)``.
            Paper default is 60; smaller values make top results
            dominate more, larger values flatten.
        top_k: If given, truncate the output to this many doc-ids.
            If None, return all doc-ids that appeared in any input.

    Returns:
        A list of doc-ids sorted by descending fused score.  Ties are
        broken by first appearance order (stable sort).

    Raises:
        ValueError: If no ranked lists are provided.
    """
    if not ranked_lists:
        raise ValueError("rrf_fuse_single requires at least one ranked list")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    counter = 0
    for ranked in ranked_lists:
        for rank_0, doc_id in enumerate(ranked):
            # rank_0 is 0-indexed; RRF uses 1-indexed rank
            rank_1 = rank_0 + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_1)
            if doc_id not in first_seen:
                first_seen[doc_id] = counter
                counter += 1

    # Sort by (-score, first_seen) for stable descending order
    fused = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )
    doc_ids = [doc_id for doc_id, _score in fused]
    if top_k is not None:
        doc_ids = doc_ids[:top_k]
    return doc_ids


def rrf_fuse_batch(
    ranked_lists_per_retriever: Sequence[Sequence[Sequence[int]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[list[int]]:
    """Batched RRF: fuse per-query rankings from N retrievers.

    Args:
        ranked_lists_per_retriever: Shape ``[n_retrievers, n_queries]``.
            Each entry is a ranked list of doc-ids for one query from
            one retriever.  All retrievers must have exactly the same
            number of queries (length-``n_queries`` outer list).
        k: RRF smoothing constant.
        top_k: Optional per-query truncation.

    Returns:
        List of length ``n_queries``; entry ``i`` is the fused ranking
        for query ``i``.

    Raises:
        ValueError: If the retrievers disagree on ``n_queries``.
    """
    if not ranked_lists_per_retriever:
        raise ValueError("rrf_fuse_batch requires at least one retriever's rankings")

    n_retrievers = len(ranked_lists_per_retriever)
    n_queries = len(ranked_lists_per_retriever[0])
    for r_idx, retriever_rankings in enumerate(ranked_lists_per_retriever):
        if len(retriever_rankings) != n_queries:
            raise ValueError(
                f"Retriever {r_idx} has {len(retriever_rankings)} queries, "
                f"expected {n_queries}"
            )

    fused_per_query: list[list[int]] = []
    for q_idx in range(n_queries):
        per_retriever_for_query = [
            ranked_lists_per_retriever[r_idx][q_idx]
            for r_idx in range(n_retrievers)
        ]
        fused_per_query.append(
            rrf_fuse_single(*per_retriever_for_query, k=k, top_k=top_k)
        )
    return fused_per_query


def rrf_fuse(
    dense_ranks: Sequence[Sequence[int]],
    sparse_ranks: Sequence[Sequence[int]],
    k: int = 60,
    top_k: int | None = None,
) -> list[list[int]]:
    """Convenience two-retriever wrapper (dense + sparse).

    Thin wrapper over ``rrf_fuse_batch`` for the common dense+sparse
    case used by the ClaimGuard-CXR evidence retriever.

    Args:
        dense_ranks: Per-query ranked doc-id lists from the dense
            retriever (e.g., MedCPT FAISS).
        sparse_ranks: Per-query ranked doc-id lists from the sparse
            retriever (e.g., BM25).
        k: RRF smoothing constant.
        top_k: Optional per-query truncation.

    Returns:
        Per-query fused ranked lists.
    """
    if len(dense_ranks) != len(sparse_ranks):
        raise ValueError(
            f"dense_ranks has {len(dense_ranks)} queries but sparse_ranks "
            f"has {len(sparse_ranks)}"
        )
    return rrf_fuse_batch(
        [dense_ranks, sparse_ranks],
        k=k,
        top_k=top_k,
    )
