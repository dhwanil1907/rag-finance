# Concepts & Workflow — RAG Finance

## Core Concepts

### 1. RAG (Retrieval-Augmented Generation)
Instead of asking an LLM to answer from memory (which leads to hallucinations), RAG first retrieves relevant chunks of text from a document store, then passes those chunks as context to the LLM. The LLM answers only from what it's given.

**Why it matters here:** Financial documents have specific numbers, dates, and figures. A general LLM would guess or hallucinate them. RAG grounds the answer in the actual filing.

---

### 2. Chunking
A 77-page 10-K can't be fed to an LLM in one shot — context windows have limits and retrieval over a full document is noisy. So the PDF is split into smaller pieces called chunks.

- **Chunk size:** 1024 characters
- **Overlap:** 128 characters (so context isn't cut off at a chunk boundary)
- **Tool used:** `RecursiveCharacterTextSplitter` from LangChain — tries to split on paragraphs, then sentences, then words, preserving natural language boundaries

Each chunk inherits metadata: company, ticker, doc type, fiscal year, filing date, page number, chunk index.

---

### 3. Embeddings
Text chunks are converted into vectors (lists of numbers) that capture semantic meaning. Chunks with similar meaning end up close together in vector space.

- **Model:** `gemini-embedding-001` (Google AI Studio, free tier)
- **Task type:** `RETRIEVAL_DOCUMENT` for ingest, `RETRIEVAL_QUERY` for search
- **Output:** One 3072-dimension vector per chunk

At query time, the user's question is also embedded using the same model (`RETRIEVAL_QUERY` task type), so the question vector can be compared against the stored chunk vectors.

---

### 4. Vector Store (ChromaDB)
A database that stores chunks along with their vectors and metadata. Supports approximate nearest-neighbor (ANN) search — find the top-k chunks whose vectors are closest to the query vector.

- **Index type:** HNSW (Hierarchical Navigable Small World) — graph-based ANN index, fast at scale
- **Similarity metric:** Cosine distance (lower = more similar)
- **Storage:** Local persistent files in `chroma_db/`
- **Key feature:** Native metadata filtering — filter by company, ticker, doc type, fiscal year *before* the vector search, not after. This is more efficient and precise.

---

### 5. Metadata Filtering
Financial queries are almost always scoped: "Apple's revenue in 2024", not "revenue across all companies ever." Metadata filters narrow the candidate set before vector search runs.

Filters supported: `company`, `ticker`, `doc_type`, `fiscal_year`, `fiscal_quarter`, `sector`, `filing_date` (range).

ChromaDB translates these into a `$and` where clause that pre-filters chunks before ANN search.

---

### 6. Reranking (Cross-Encoder)
Vector search (bi-encoder) is fast but imprecise — it finds semantically similar chunks but can miss nuances like whether a chunk is actually *about* the question vs. just using similar words.

A **cross-encoder** takes the query and each retrieved chunk *together* as a pair and scores their relevance. Much more accurate, but slower — so it's only run on the top 20 chunks from vector search, and then the top 5 are kept.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local HuggingFace, no API needed)
- **Input:** (query, chunk) pairs
- **Output:** A relevance score per pair, chunks re-sorted by score

---

### 7. Generation (Gemini 2.5 Flash)
The top 5 reranked chunks are formatted into a prompt with source headers and passed to Gemini. The model is instructed to:
- Answer only from the numbered sources
- Cite using bracket numbers [1], [2]
- Say explicitly if the answer isn't in the sources

**Temperature:** 0.2 (low randomness — financial answers should be factual, not creative)

---

### 8. Source Attribution
Every answer comes with `sources` — the exact chunks used, including company, doc type, filing date, page number, vector distance score, and rerank score. This makes the answer auditable.

---

## Full Workflow

```
PDF file (data/raw/AAPL)
        │
        ▼
[1. Parse] — pdfplumber extracts text page by page
        │
        ▼
[2. Chunk] — RecursiveCharacterTextSplitter splits into ~1024-char pieces
             each chunk gets metadata: company, ticker, doc_type, fiscal_year,
             filing_date, sector, page_number, chunk_index
        │
        ▼
[3. Embed] — Gemini embedding-001 converts each chunk to a 3072-dim vector
             batched in groups of 20, with 25s pauses (free tier rate limit)
        │
        ▼
[4. Store] — ChromaDB upserts vectors + text + metadata into local chroma_db/
             HNSW index built automatically
        │
        │   (ingest complete — done once per document)
        │
        ▼
[5. Query] — User submits a natural language question + optional filters
             (via Streamlit UI → FastAPI POST /query)
        │
        ▼
[6. Embed query] — Same Gemini model embeds the question (RETRIEVAL_QUERY task)
        │
        ▼
[7. Filter + Search] — ChromaDB applies metadata filters, then ANN search
                       returns top 20 chunks by cosine distance
        │
        ▼
[8. Rerank] — cross-encoder/ms-marco-MiniLM-L-6-v2 scores each (query, chunk)
              pair, keeps top 5 by relevance score
        │
        ▼
[9. Generate] — Top 5 chunks formatted into a RAG prompt with source headers
                Gemini 2.5 Flash generates an answer with citations
        │
        ▼
[10. Response] — FastAPI returns { answer, sources, chunk_count }
                 Streamlit displays the answer + expandable sources panel
```

---

## Why Each Layer Exists

| Layer | Problem it solves |
|---|---|
| Chunking | LLMs have context limits; retrieval over full docs is noisy |
| Embeddings | Enables semantic search — find relevant chunks even if exact words don't match |
| Metadata filtering | Scopes search to the right company/period before vector search runs |
| Vector search | Fast approximate retrieval over thousands of chunks |
| Reranking | Improves precision — vector search optimizes recall, cross-encoder optimizes relevance |
| RAG prompt | Grounds the LLM answer in real document text, preventing hallucination |
| Source attribution | Makes answers auditable — user can verify which page/chunk the answer came from |
