"""PDF → text chunks with inherited and per-chunk metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.config import DocumentConfig

logger = logging.getLogger(__name__)

# Larger chunks → fewer embed calls / faster ingest; smaller → finer retrieval.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


class ParsedChunk(TypedDict):
    text: str
    metadata: dict[str, Any]


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def parse_pdf(pdf_path: Path, doc_config: DocumentConfig) -> list[ParsedChunk]:
    """
    Extract text page-by-page, split each page with RecursiveCharacterTextSplitter,
    and attach metadata (config fields + source_file, page_number, chunk_index).
    """
    source_file = doc_config["filename"]
    fiscal_quarter = doc_config.get("fiscal_quarter")
    # Chroma metadata has no null for ints; -1 means "no quarter" (e.g. 10-K).
    quarter_value = -1 if fiscal_quarter is None else int(fiscal_quarter)

    base: dict[str, Any] = {
        "company": doc_config["company"],
        "ticker": doc_config["ticker"],
        "doc_type": doc_config["doc_type"],
        "fiscal_year": doc_config["fiscal_year"],
        "fiscal_quarter": quarter_value,
        "filing_date": doc_config["filing_date"],
        "sector": doc_config["sector"],
        "source_file": source_file,
    }

    chunks: list[ParsedChunk] = []
    chunk_index = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                logger.debug("Skipping empty page %s in %s", page_number, source_file)
                continue
            for piece in _splitter.split_text(page_text):
                meta = {
                    **base,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
                chunks.append({"text": piece, "metadata": meta})
                chunk_index += 1

    logger.info("Parsed %s → %s chunks", source_file, len(chunks))
    return chunks
