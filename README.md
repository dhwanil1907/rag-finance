# Financial Document RAG

## Overview

This project is a retrieval-augmented Q&A system over financial PDFs (10-K, 10-Q, earnings transcripts). Chunks are embedded into ChromaDB with rich metadata (company, ticker, fiscal period, filing date, sector). Unlike a minimal RAG demo, retrieval **narrows the corpus with metadata filters before** vector search, then **re-ranks** candidates with a cross-encoder before **Gemini** generates an answer with citations. A FastAPI service exposes `/query`, and a Streamlit UI drives filters and question input.

## Architecture

```
  PDF (data/raw/)
       |
       v
  parse (pdfplumber + split)
       |
       v
  embed (gemini-embedding-001)
       |
       v
  ChromaDB (persistent, metadata per chunk)
       ^
       |
  POST /query  ---- metadata filters + embedding ----+
       |                                             |
       v                                             |
  vector top-k  -->  cross-encoder rerank top-k      |
       |                                             |
       v                                             |
  Gemini 1.5 Flash (RAG prompt + citations)  <-------+
       |
       v
  JSON answer + sources  -->  Streamlit UI
```

## Tech Stack

| Layer | Tool | Why |
| --- | --- | --- |
| Parsing | pdfplumber | Text extraction from filings |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | 1024 / 128 overlap |
| Embeddings | Google `gemini-embedding-001` | Gemini API; API key only |
| Vector DB | ChromaDB (local) | Metadata filters + persistence |
| Retrieval | Custom `VectorRetriever` | Filter → query by embedding |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, small, improves precision |
| Generation | Gemini 1.5 Flash | Fast, long-context answers |
| API | FastAPI | Typed request/response, easy to script |
| UI | Streamlit | Quick portfolio demo |

## Setup

**Prerequisites:** Python 3.11+, a [Google AI Studio](https://aistudio.google.com/) API key (free tier is enough for development).

```bash
cd rag-finance
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Set GOOGLE_API_KEY in .env
```

Add PDFs under `data/raw/` and add matching rows to `ingest/config.py`. Then ingest and run the stack:

```bash
python -m ingest.embed
# Terminal A:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Terminal B:
streamlit run ui/app.py
```

Optional: point the UI at another host with `RAG_API_BASE=http://localhost:8000` (default).

## Example Queries

Use the sidebar to scope filings, then try:

1. **Company + year:** Select *Apple Inc.* and fiscal year *2023*, ask: *What does management cite as the top competitive risks in the annual report?*
2. **Form type:** Select *10-Q*, ask: *How did liquidity and capital resources change versus the prior quarter?*
3. **Sector:** Select *Technology* and a *10-K*, ask: *Summarize revenue recognition policies described in the filing.*
4. **Date range:** Set filing *From / To* around a known earnings season, ask: *What forward guidance did the company give?*
5. **Broad + text:** Clear filters, ask: *Compare debt maturity profile language across the retrieved chunks* (relies on semantic retrieval across all ingested names).

## Project Structure

```
rag-finance/
├── api/
│   └── main.py           # FastAPI: /health, /documents, /query
├── data/
│   ├── raw/              # PDFs
│   └── processed/        # reserved
├── generation/
│   ├── gemini.py
│   └── prompt.py
├── ingest/
│   ├── config.py
│   ├── embed.py
│   └── parse.py
├── retrieval/
│   ├── query.py
│   ├── rerank.py
│   └── retrieve.py
├── ui/
│   └── app.py            # Streamlit
├── chroma_db/            # created at runtime (Chroma persistence)
├── .env.example
├── requirements.txt
└── README.md
```
# rag-finance
