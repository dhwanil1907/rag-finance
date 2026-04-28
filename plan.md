# RAG with Metadata Filtering — Project Plan

*Last reviewed: April 2026.*

## Overview

Document Q&A system over financial documents (10-Ks, earnings call transcripts) using retrieval-augmented generation with structured metadata filtering. Pairs thematically with FinSight as a resume cluster.

**Status:** Phases 1–4 are implemented end-to-end (ingest → Chroma → retrieval/rerank → Gemini → FastAPI → Streamlit). **Corpus:** one Apple 10-K is configured (`data/raw/AAPL`, `ingest/config.py`) and embedded into `chroma_db/`.

---

## What's left to do

**Ship checklist (from roadmap)**  
Add more PDFs under `data/raw/`, one `DOCUMENT_CONFIGS` entry per file in `ingest/config.py`, then run `python -m ingest.embed`. Aim for **2–3+ filings** (mixed 10-K / 10-Q / transcript is fine) so filters and retrieval are exercised across documents. **Manually evaluate retrieval**: 10–15 representative queries (filtered and unfiltered), note bad chunks or misses. **Record a short screen demo** for portfolio/resume. **Optional:** deploy (e.g. Hugging Face Spaces or Railway).

**Corpus depth (original target)**  
Grow toward **5–10 companies** and **2–3 years** of filings each (SEC PDFs via EDGAR; transcripts manual or pre-downloaded). There is no automated EDGAR downloader in this repo yet—ingest remains config-driven per file.

**Stretch features (see Features → Stretch)**  
Multi-document comparison queries, PDF table extraction, Gemini multimodal over figures/charts, streaming answers (FastAPI + SSE).

---

## Goals

- Answer natural language questions grounded in specific financial documents
- Filter retrieval by company, date range, fiscal quarter, and sector
- Score and re-rank retrieved chunks before feeding to LLM
- Use Gemini API (free tier) as the generation backend

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | `gemini-embedding-001` (Google) | Free via Gemini API |
| Vector Store | ChromaDB | Local, persistent, supports metadata filtering natively |
| LLM | Gemini 1.5 Flash | Free tier, multimodal capable |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace) | Local, lightweight |
| Document Parsing | `pdfplumber` + `langchain` text splitters | Handles multi-column financial PDFs |
| API Layer | FastAPI | Clean REST interface for queries |
| Frontend | Streamlit | Filter sidebar + Q&A (`ui/app.py`) |
| Language | Python 3.11+ | |
| Env/Secrets | `python-dotenv` | |

---

## Features

### Core
- [x] Ingest PDF financial documents (10-Ks, earnings transcripts)
- [x] Chunk documents with metadata attached (company, ticker, doc type, date, sector, fiscal year)
- [x] Embed chunks and store in ChromaDB with metadata
- [x] Query endpoint: natural language → filtered vector search → ranked chunks → Gemini answer
- [x] Metadata filtering: filter by company, date range, sector, doc type before vector search
- [x] Retrieval scoring: return top-k with vector distance / scores in API responses
- [x] Re-ranking layer: cross-encoder reranks top-k before generation
- [x] Source attribution: API returns `sources` (company, filing, page, scores) alongside the answer

### Stretch
- [ ] Multi-document comparison (e.g. "compare Apple vs Microsoft R&D spend in 2023")
- [ ] Table extraction from PDFs (structured financial data)
- [ ] Gemini multimodal: answer questions about charts/graphs in filings
- [ ] Streaming responses via FastAPI + SSE

---

## Data Sources

- SEC EDGAR full-text search API (free, no key needed): `https://efts.sec.gov/LATEST/search-index?q=...`
- Direct PDF links from EDGAR for 10-Ks and 10-Qs
- Earnings call transcripts: Motley Fool, Seeking Alpha (manual scrape or pre-downloaded)
- Target corpus: 5–10 companies, 2–3 years of filings each (manageable for local ChromaDB)

---

## Document Schema (Metadata per Chunk)

Logical shape (ingest config + Chroma). Stored `fiscal_quarter` uses integer **-1** for annual / no quarter (API returns `null` for that case).

```python
{
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "doc_type": "10-K",          # 10-K | 10-Q | earnings_transcript
  "fiscal_year": 2023,
  "fiscal_quarter": None,       # Q1–Q4 or None for annual (stored as -1 in Chroma)
  "filing_date": "2023-11-03",
  "sector": "Technology",
  "source_file": "aapl_10k_2023.pdf",
  "page_number": 42,
  "chunk_index": 7
}
```

---

## Architecture

```
User Query
    │
    ▼
[Filters — API or Streamlit UI]
    │  company, doc type, sector, fiscal year, filing dates
    ▼
[ChromaDB Metadata Pre-filter]
    │  narrows candidate set before ANN search
    ▼
[Vector Search — top-k=20]
    │
    ▼
[Cross-Encoder Reranker — top-k=5]
    │
    ▼
[Gemini 1.5 Flash — RAG prompt + citations]
    │
    ▼
[Answer + Sources JSON]  →  Streamlit / clients
```

---

## Roadmap

### Phase 1 — Ingestion Pipeline (Days 1–2)
- [x] Set up ChromaDB with persistent local storage (`./chroma_db`)
- [x] PDF parser with `pdfplumber`, chunking with `RecursiveCharacterTextSplitter` (512 / 64)
- [x] Metadata extraction per document (manual `ingest/config.py` per PDF)
- [x] End-to-end ingest verified on one filing (Apple 10-K: `data/raw/AAPL` + `python -m ingest.embed`)
- [ ] Expand to 2–3+ 10-Ks (or mixed 10-Q / transcript) for a meaningful multi-document smoke corpus

### Phase 2 — Retrieval + Reranking (Days 3–4)
- [x] Query module: metadata filters → vector search (`retrieval/query.py`, `retrieve.py`)
- [x] Integrate cross-encoder reranker (`retrieval/rerank.py`)
- [ ] Evaluate retrieval quality manually (spot-check 10–15 queries on your corpus)
- [x] Chunk size and overlap tuned to 512 / 64 in code

### Phase 3 — Generation + API (Days 5–6)
- [x] RAG prompt template with citations (`generation/prompt.py`)
- [x] FastAPI routes: `POST /query`, `GET /documents`, `GET /health` (`api/main.py`)
- [x] Gemini API integration with retry/backoff (`generation/gemini.py`)
- [x] Structured response: `{ answer, sources, chunk_count }` (+ per-source `vector_score`, `rerank_score`)

### Phase 4 — Polish + Demo (Day 7)
- [x] Streamlit UI with filter sidebar (`ui/app.py`)
- [x] README with architecture diagram, setup, example queries
- [ ] Record short demo (for portfolio/resume)
- [ ] Deploy optional: Hugging Face Spaces or Railway (free tier)

---

## Key Design Decisions

**ChromaDB over FAISS**: ChromaDB supports native metadata filtering before the ANN search, which is the core value-add here. FAISS requires post-filtering (less efficient on large corpora).

**Reranking**: Bi-encoder retrieval (vector search) optimizes for recall; cross-encoder reranking optimizes precision. Both are needed for finance Q&A where a slightly off chunk can produce hallucinated numbers.

**Metadata-first filtering**: Financial queries are almost always scoped (specific company, specific period). Filtering before vector search dramatically reduces noise and improves answer accuracy.

**Gemini Flash over GPT**: Free tier, sufficient context window (1M tokens), Anthropic-independent for resume diversity.

---

## Project Structure

```
rag-finance/
├── data/
│   ├── raw/          # downloaded PDFs
│   └── processed/    # reserved / future use
├── ingest/
│   ├── parse.py      # PDF → chunks
│   ├── embed.py      # chunks → ChromaDB
│   └── config.py     # doc metadata config
├── retrieval/
│   ├── query.py      # metadata filter + vector search
│   ├── rerank.py     # cross-encoder reranking
│   └── retrieve.py   # filter → search → rerank (pipeline entry)
├── generation/
│   ├── prompt.py     # RAG prompt string
│   └── gemini.py     # Gemini API client
├── api/
│   └── main.py       # FastAPI app
├── ui/
│   └── app.py        # Streamlit frontend
├── chroma_db/        # local persistent store (runtime)
├── .env.example
├── requirements.txt
├── README.md
└── plan.md
```

---

## Resume / Interview Angle

- "Built a document Q&A pipeline over SEC filings combining vector search with structured metadata filtering — closer to production RAG than toy demos"
- "Added a reranking layer that improved answer precision by reducing noise from semantically similar but date/company-mismatched chunks"
- Pairs with FinSight: "I've applied both structured SQL-style analysis and unstructured LLM retrieval to financial data"
