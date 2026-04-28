"""Cross-encoder reranking over retrieved chunks."""

from __future__ import annotations

from retrieval.query import RetrievedChunk

_model = None  # lazy-loaded


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Score each chunk with a cross-encoder, return top_k sorted by score desc."""
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = score

    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)  # type: ignore[arg-type]
    return ranked[:top_k]
