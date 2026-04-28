"""Single-string RAG prompt for Gemini."""

from __future__ import annotations

from retrieval.query import RetrievedChunk

_SYSTEM = (
    "You are a financial document assistant. Answer only from the numbered sources below. "
    "Cite sources using bracket numbers like [1], [2], etc. If the answer is not contained "
    "in the sources, say so explicitly. Do not hallucinate numbers, dates, or facts."
)


def build_rag_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build the full prompt: system rules, numbered source blocks, and the user question.
    Each chunk is formatted as:
      [n] Company: ... | Doc: ... | Date: ... | Page: ...
      <chunk text>
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk.metadata or {}
        header = (
            f"[{i}] Company: {m.get('company', '')} | Doc: {m.get('doc_type', '')} | "
            f"Date: {m.get('filing_date', '')} | Page: {m.get('page_number', '')}"
        )
        blocks.append(f"{header}\n{chunk.text.strip()}")

    context = "\n\n".join(blocks)
    q = query.strip()
    return f"{_SYSTEM}\n\n{context}\n\nQuestion: {q}\n\nAnswer:"
