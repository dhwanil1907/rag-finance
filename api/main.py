"""FastAPI: health, documents, and RAG query."""

from __future__ import annotations

import logging
from collections import Counter
from contextlib import asynccontextmanager
from typing import Annotated, Any

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from generation.gemini import generate
from generation.prompt import build_rag_prompt
from ingest.config import CHROMA_DIR
from retrieval.correct import corrective_retrieve
from retrieval.query import COLLECTION_NAME, MetadataFilter, VectorRetriever
from retrieval.rerank import rerank

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CHROMA_PAGE_LIMIT = 50_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    app.state.retriever = VectorRetriever()
    yield


def get_retriever(request: Request) -> VectorRetriever:
    return request.app.state.retriever


RetrieverDep = Annotated[VectorRetriever, Depends(get_retriever)]


app = FastAPI(title="Financial RAG API", version="0.1.0", lifespan=lifespan)


class QueryFilters(BaseModel):
    company: str | None = None
    ticker: str | None = None
    doc_type: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    sector: str | None = None
    filing_date_from: str | None = None
    filing_date_to: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: QueryFilters | None = None
    vector_top_k: int = Field(20, ge=1, le=100)
    rerank_top_k: int = Field(5, ge=1, le=50)


class SourceItem(BaseModel):
    company: str
    ticker: str
    doc_type: str
    fiscal_year: int
    fiscal_quarter: int | None = None
    filing_date: str
    sector: str
    source_file: str
    page_number: int
    vector_score: float
    rerank_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    chunk_count: int
    reformulated_query: str | None = None


class DocumentRow(BaseModel):
    company: str | None = None
    ticker: str | None = None
    doc_type: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    filing_date: str | None = None
    sector: str | None = None
    source_file: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    document_count: int


def _filters_to_metadata(qf: QueryFilters | None) -> MetadataFilter | None:
    if qf is None:
        return None
    data = qf.model_dump(exclude_none=True)
    if not data:
        return None
    return MetadataFilter(**data)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fiscal_quarter_api(meta: dict[str, Any]) -> int | None:
    fq = _coerce_int(meta.get("fiscal_quarter"), -1)
    return None if fq < 0 else fq


def _chunk_to_source_item(chunk) -> SourceItem:
    meta = chunk.metadata or {}
    fq_api = _fiscal_quarter_api(meta)
    rs = chunk.rerank_score if chunk.rerank_score is not None else 0.0
    return SourceItem(
        company=str(meta.get("company", "")),
        ticker=str(meta.get("ticker", "")),
        doc_type=str(meta.get("doc_type", "")),
        fiscal_year=_coerce_int(meta.get("fiscal_year"), 0),
        fiscal_quarter=fq_api,
        filing_date=str(meta.get("filing_date", "")),
        sector=str(meta.get("sector", "")),
        source_file=str(meta.get("source_file", "")),
        page_number=_coerce_int(meta.get("page_number"), 0),
        vector_score=float(chunk.score),
        rerank_score=float(rs),
    )


def _aggregate_documents(collection) -> tuple[list[DocumentRow], int]:
    raw = collection.get(include=["metadatas"], limit=_CHROMA_PAGE_LIMIT)
    metas = raw.get("metadatas") or []
    counts: Counter[str] = Counter()
    sample_meta: dict[str, dict[str, Any]] = {}
    for meta in metas:
        if not meta:
            continue
        sf = meta.get("source_file")
        if not sf:
            continue
        sf = str(sf)
        counts[sf] += 1
        if sf not in sample_meta:
            sample_meta[sf] = dict(meta)

    rows: list[DocumentRow] = []
    for sf in sorted(counts.keys()):
        meta = sample_meta[sf]
        fy = _coerce_int(meta.get("fiscal_year"), 0)
        rows.append(
            DocumentRow(
                source_file=sf,
                company=meta.get("company"),
                ticker=meta.get("ticker"),
                doc_type=meta.get("doc_type"),
                fiscal_year=fy if fy else None,
                filing_date=meta.get("filing_date"),
                sector=meta.get("sector"),
                fiscal_quarter=_fiscal_quarter_api(meta),
                chunk_count=counts[sf],
            )
        )
    document_count = len(counts)
    return rows, document_count


@app.get("/health", response_model=HealthResponse)
def health(retriever: RetrieverDep) -> HealthResponse:
    _, document_count = _aggregate_documents(retriever.collection)
    return HealthResponse(
        status="ok",
        collection=COLLECTION_NAME,
        document_count=document_count,
    )


@app.get("/documents", response_model=list[DocumentRow])
def list_documents(retriever: RetrieverDep) -> list[DocumentRow]:
    rows, _ = _aggregate_documents(retriever.collection)
    return rows


@app.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest, retriever: RetrieverDep) -> QueryResponse:
    filters = _filters_to_metadata(body.filters)
    try:
        chunks, reformulated_query = corrective_retrieve(
            body.query,
            retriever,
            filters=filters,
            vector_top_k=body.vector_top_k,
            rerank_top_k=body.rerank_top_k,
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return QueryResponse(
            answer="No chunks matched the query and filters. Try broader filters or ingest documents.",
            sources=[],
            chunk_count=0,
        )

    sources = [_chunk_to_source_item(c) for c in chunks]
    prompt = build_rag_prompt(body.query, chunks)

    try:
        answer = generate(prompt)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunk_count=len(sources),
        reformulated_query=reformulated_query,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
