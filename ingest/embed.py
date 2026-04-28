"""Embed parsed chunks and upsert into ChromaDB (persistent local)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from ingest.config import (
    CHROMA_DIR,
    DOCUMENT_CONFIGS,
    GOOGLE_EMBEDDING_MODEL,
    RAW_DIR,
    DocumentConfig,
)
from ingest.parse import parse_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "financial_docs"
# Free tier: daily cap on embedding requests (e.g. 1000/day for gemini-embedding-001); each
# text in a batch typically counts as one request. RPM limits also apply — small batches + pause helps.
EMBED_BATCH_SIZE = 20
RATE_LIMIT_SLEEP_SEC = 25
EMBED_MAX_RETRIES = 8
QUOTA_RETRY_BASE_SEC = 40

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            logger.error("GOOGLE_API_KEY is not set (load .env or export it).")
            sys.exit(1)
        _client = genai.Client(api_key=key)
    return _client


def _is_daily_embedding_quota_exhausted(exc: BaseException) -> bool:
    """Gemini free tier: daily embed request limit hit (retries won't help until quota resets)."""
    msg = str(exc).lower()
    return (
        "embed_content_free_tier" in msg
        or "embedcontentrequestsperday" in msg.replace("_", "")
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed via Gemini embedding model; returns one vector per input string."""
    if not texts:
        return []
    client = _get_client()
    last_err: BaseException | None = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=GOOGLE_EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in result.embeddings]
        except genai_errors.APIError as exc:
            last_err = exc
            if _is_daily_embedding_quota_exhausted(exc):
                logger.error(
                    "Gemini embedding daily quota exhausted (free tier ~1000 requests/day). "
                    "Wait until it resets, use another API key/project, or enable billing. %s",
                    exc,
                )
                raise RuntimeError(
                    "Daily Gemini embedding quota exceeded; see log. Retrying today will not help."
                ) from exc
            if exc.code != 429 and exc.code != 503:
                raise
            wait = QUOTA_RETRY_BASE_SEC + attempt * 10
            logger.warning(
                "Embedding API throttled (%s); sleeping %ss then retry %s/%s",
                exc.code,
                wait,
                attempt + 1,
                EMBED_MAX_RETRIES,
            )
            time.sleep(wait)
        except Exception as exc:
            if _is_daily_embedding_quota_exhausted(exc):
                logger.error(
                    "Gemini embedding daily quota exhausted (free tier ~1000 requests/day). "
                    "Wait until it resets, use another API key/project, or enable billing. %s",
                    exc,
                )
                raise RuntimeError(
                    "Daily Gemini embedding quota exceeded; see log. Retrying today will not help."
                ) from exc
            msg = str(exc).lower()
            if "429" not in msg and "quota" not in msg and "resource exhausted" not in msg:
                raise
            last_err = exc
            wait = QUOTA_RETRY_BASE_SEC + attempt * 10
            logger.warning(
                "Embedding quota error (%s); sleeping %ss then retry %s/%s",
                type(exc).__name__,
                wait,
                attempt + 1,
                EMBED_MAX_RETRIES,
            )
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {EMBED_MAX_RETRIES} attempts") from last_err


def _source_file_already_ingested(collection: Collection, source_file: str) -> bool:
    existing = collection.get(
        where={"source_file": source_file},
        limit=1,
        include=[],
    )
    return bool(existing["ids"])


def _chunk_ids(source_file: str, chunk_indices: list[int]) -> list[str]:
    return [f"{source_file}::{idx}" for idx in chunk_indices]


def ingest_document(
    collection: Collection,
    pdf_path: Path,
    doc_config: DocumentConfig,
) -> int:
    """Parse, embed, upsert one PDF. Returns number of chunks written."""
    chunks = parse_pdf(pdf_path, doc_config)
    if not chunks:
        logger.warning("No chunks produced for %s", doc_config["filename"])
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    indices = [int(m["chunk_index"]) for m in metadatas]
    ids = _chunk_ids(doc_config["filename"], indices)

    all_embeddings: list[list[float]] = []
    batches = list(range(0, len(texts), EMBED_BATCH_SIZE))
    for i, start in enumerate(batches):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        logger.info(
            "Embedding %s chunks %s–%s / %s",
            doc_config["filename"],
            start,
            min(start + len(batch), len(texts)) - 1,
            len(texts),
        )
        all_embeddings.extend(embed_texts(batch))
        # Rate limit: free tier = 100 requests/min. Each text = 1 request.
        # Sleep between batches (not after the last one).
        if i < len(batches) - 1:
            logger.info("Rate limit pause: sleeping %ss…", RATE_LIMIT_SLEEP_SEC)
            time.sleep(RATE_LIMIT_SLEEP_SEC)

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def run_ingestion(*, force: bool = False) -> None:
    load_dotenv()
    _get_client()  # validates API key early

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Removed collection %r (force re-ingest)", COLLECTION_NAME)
        except Exception as exc:
            logger.warning("Could not delete collection (ok if missing): %s", exc)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for doc_config in DOCUMENT_CONFIGS:
        filename = doc_config["filename"]
        pdf_path = RAW_DIR / filename

        if not pdf_path.is_file():
            logger.warning("Skipping %s: file not found under data/raw/", filename)
            continue

        if _source_file_already_ingested(collection, filename):
            logger.info("Skipping %s: already present in ChromaDB", filename)
            continue

        logger.info("Ingesting %s", filename)
        n = ingest_document(collection, pdf_path, doc_config)
        logger.info("Upserted %s chunks for %s", n, filename)

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed PDFs from data/raw into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the vector collection first (fixes broken HNSW / sqlite mismatch).",
    )
    args = parser.parse_args()
    run_ingestion(force=args.force)


if __name__ == "__main__":
    main()
