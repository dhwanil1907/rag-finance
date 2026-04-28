"""Corrective RAG: detect weak retrievals and retry with a reformulated query."""

from __future__ import annotations

import logging

from retrieval.query import MetadataFilter, RetrievedChunk, VectorRetriever
from retrieval.rerank import rerank

logger = logging.getLogger(__name__)

# Cross-encoder (ms-marco-MiniLM-L-6-v2) outputs raw logits.
# Below this threshold the best chunk is likely irrelevant — trigger correction.
RERANK_SCORE_THRESHOLD = -2.0


def _best_rerank_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return float("-inf")
    return max(c.rerank_score for c in chunks if c.rerank_score is not None)


def _reformulate(query: str) -> str:
    """Ask Gemini to rephrase the query for better financial document retrieval."""
    from generation.gemini import generate

    prompt = (
        "You are helping improve a search query over SEC financial filings (10-Ks, 10-Qs).\n"
        "Rewrite the query below to be more specific and likely to match text found in a financial document.\n"
        "Return ONLY the rewritten query — no explanation, no punctuation changes beyond the query itself.\n\n"
        f"Original query: {query}\n\n"
        "Rewritten query:"
    )
    try:
        return generate(prompt).strip()
    except Exception as exc:
        logger.warning("Query reformulation failed (%s); using original query.", exc)
        return query


def corrective_retrieve(
    query: str,
    retriever: VectorRetriever,
    filters: MetadataFilter | None = None,
    vector_top_k: int = 20,
    rerank_top_k: int = 5,
) -> tuple[list[RetrievedChunk], str | None]:
    """
    Full corrective retrieval pipeline:
      1. Run standard retrieve (vector search + rerank)
      2. If best rerank score < RERANK_SCORE_THRESHOLD, reformulate the query and retry
      3. Return whichever attempt scored higher, plus the reformulated query if used

    Returns:
        (chunks, reformulated_query) — reformulated_query is None if no correction was needed
    """
    candidates = retriever.search(query, filters=filters, top_k=vector_top_k)
    chunks = rerank(query, candidates, top_k=rerank_top_k)
    score = _best_rerank_score(chunks)

    if score >= RERANK_SCORE_THRESHOLD:
        return chunks, None

    logger.info(
        "Weak retrieval (best rerank score=%.2f < %.2f); reformulating query.",
        score,
        RERANK_SCORE_THRESHOLD,
    )

    reformulated = _reformulate(query)
    if reformulated == query:
        return chunks, None

    logger.info("Retrying with reformulated query: %r", reformulated)
    retry_candidates = retriever.search(reformulated, filters=filters, top_k=vector_top_k)
    retry_chunks = rerank(reformulated, retry_candidates, top_k=rerank_top_k)
    retry_score = _best_rerank_score(retry_chunks)

    if retry_score > score:
        logger.info(
            "Corrected retrieval improved score: %.2f → %.2f", score, retry_score
        )
        return retry_chunks, reformulated

    logger.info(
        "Reformulated query did not improve results (%.2f vs %.2f); keeping original.",
        retry_score,
        score,
    )
    return chunks, None
