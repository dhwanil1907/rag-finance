"""Public interface: filter → vector search → rerank → return top chunks."""

from __future__ import annotations

from retrieval.query import MetadataFilter, RetrievedChunk, VectorRetriever
from retrieval.rerank import rerank

_retriever: VectorRetriever | None = None


def _get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever


def retrieve(
    query: str,
    filters: MetadataFilter | None = None,
    vector_top_k: int = 20,
    rerank_top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline:
      1. Metadata pre-filter + vector search (top vector_top_k)
      2. Cross-encoder rerank (top rerank_top_k)

    Returns chunks ordered by rerank score descending.
    Each chunk has: .text, .score (vector distance), .rerank_score, .metadata
    """
    retriever = _get_retriever()
    candidates = retriever.search(query, filters=filters, top_k=vector_top_k)
    return rerank(query, candidates, top_k=rerank_top_k)
