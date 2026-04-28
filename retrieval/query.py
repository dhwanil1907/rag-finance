"""Vector search with metadata pre-filtering."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from ingest.config import CHROMA_DIR, GOOGLE_EMBEDDING_MODEL

COLLECTION_NAME = "financial_docs"


@dataclass
class MetadataFilter:
    """All fields are optional; only non-None values are applied as filters."""

    company: str | None = None
    ticker: str | None = None
    doc_type: str | None = None          # "10-K" | "10-Q" | "earnings_transcript"
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    sector: str | None = None
    filing_date_from: str | None = None  # YYYY-MM-DD inclusive
    filing_date_to: str | None = None    # YYYY-MM-DD inclusive


@dataclass
class RetrievedChunk:
    text: str
    score: float                          # cosine distance (lower = more similar)
    metadata: dict[str, Any]
    rerank_score: float | None = None     # filled in by reranker


def _build_where_clause(f: MetadataFilter) -> dict[str, Any] | None:
    """Translate MetadataFilter into a ChromaDB $and where clause."""
    conditions: list[dict[str, Any]] = []

    if f.company:
        conditions.append({"company": {"$eq": f.company}})
    if f.ticker:
        conditions.append({"ticker": {"$eq": f.ticker}})
    if f.doc_type:
        conditions.append({"doc_type": {"$eq": f.doc_type}})
    if f.fiscal_year is not None:
        conditions.append({"fiscal_year": {"$eq": f.fiscal_year}})
    if f.fiscal_quarter is not None:
        conditions.append({"fiscal_quarter": {"$eq": f.fiscal_quarter}})
    if f.sector:
        conditions.append({"sector": {"$eq": f.sector}})
    if f.filing_date_from:
        conditions.append({"filing_date": {"$gte": f.filing_date_from}})
    if f.filing_date_to:
        conditions.append({"filing_date": {"$lte": f.filing_date_to}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _embed_query(text: str, client: genai.Client) -> list[float]:
    result = client.models.embed_content(
        model=GOOGLE_EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


class VectorRetriever:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set.")
        self._genai = genai.Client(api_key=api_key)

        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    def search(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Embed query, apply metadata filters, return top_k chunks."""
        query_vector = _embed_query(query, self._genai)
        where = _build_where_clause(filters) if filters else None

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            chunks.append(RetrievedChunk(text=doc, score=dist, metadata=meta))

        return chunks
