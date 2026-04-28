"""Gemini text generation with retries (model configurable; default is current Flash)."""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# 1.5-* IDs are retired on the Gemini API; override with GEMINI_GENERATION_MODEL if needed.
GEMINI_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
_BACKOFF_SEC = (1.0, 2.0, 4.0)
_MAX_ATTEMPTS = 3

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "503" in msg or "resource exhausted" in msg or "deadline" in msg


def generate(prompt: str) -> str:
    """
    Call Gemini (default Flash) with the full prompt string.
    Up to 3 attempts with exponential backoff on transient errors.
    """
    client = _get_client()

    last_error: BaseException | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = response.text
            if text and text.strip():
                return text.strip()
            raise RuntimeError("Model returned an empty response.")
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc):
                raise RuntimeError(
                    f"Gemini request failed (non-retryable): {type(exc).__name__}: {exc}"
                ) from exc
            if attempt == _MAX_ATTEMPTS - 1:
                break
            delay = _BACKOFF_SEC[attempt]
            logger.warning(
                "Gemini transient error (%s: %s); sleeping %.1fs before retry %s/%s",
                type(exc).__name__,
                exc,
                delay,
                attempt + 2,
                _MAX_ATTEMPTS,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Gemini failed after {_MAX_ATTEMPTS} attempts (last error: "
        f"{type(last_error).__name__}: {last_error})"
    ) from last_error
