"""Per-PDF metadata: add one dict per file in data/raw/."""

from pathlib import Path
from typing import Literal, NotRequired, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Gemini API (google-generativeai): text-embedding-004 was removed from v1beta; use current model.
GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"

DocType = Literal["10-K", "10-Q", "earnings_transcript"]


class DocumentConfig(TypedDict):
    """Maps a PDF filename to filing metadata (manual source of truth)."""

    filename: str
    company: str
    ticker: str
    doc_type: DocType
    fiscal_year: int
    filing_date: str  # YYYY-MM-DD
    sector: str
    fiscal_quarter: NotRequired[int | None]


# One entry per PDF placed in data/raw/. Edit filenames and fields to match your files.
DOCUMENT_CONFIGS: list[DocumentConfig] = [
    {
        "filename": "AAPL",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "doc_type": "10-K",
        "fiscal_year": 2024,
        "fiscal_quarter": None,
        "filing_date": "2024-11-01",
        "sector": "Technology",
    },
]


def config_by_filename() -> dict[str, DocumentConfig]:
    return {cfg["filename"]: cfg for cfg in DOCUMENT_CONFIGS}
