"""
llm.py
------
Natural-language explanation generation via Google Gemini.

Reads the API key from (in order):
    1. Explicit api_key argument
    2. GEMINI_API_KEY environment variable
    3. GOOGLE_API_KEY environment variable

Functions:
    generate_explanation(prompt, ...) - Call Gemini with the SHAP prompt
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from google import genai

from logging_utils import get_logger

logger = get_logger("llm")

ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
SRC_ENV = Path(__file__).resolve().parent / ".env"

DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = (
    "gemini-3.6-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
)
RETRYABLE_TOKENS = (
    "NOT_FOUND",
    "404",
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "high demand",
    "try again",
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Reload on every call so a newly added .env is picked up
    # without restarting Streamlit.
    load_dotenv(ROOT_ENV, override=True)
    load_dotenv(SRC_ENV, override=False)


def _resolve_api_key(api_key=None):
    _load_env()
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if key:
        key = key.strip().strip('"').strip("'")
    if not key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in your environment (or a .env file), or pass api_key=..."
        )
    return key


def _model_candidates(preferred: str | None) -> list[str]:
    models = []
    if preferred:
        models.append(preferred)
    for name in FALLBACK_MODELS:
        if name not in models:
            models.append(name)
    return models


def generate_explanation(prompt, model=DEFAULT_MODEL, api_key=None,
                         retries: int = 3):
    """
    Generate an applicant-facing rejection explanation with Gemini.

    Retries overloaded models, then walks a fallback list. Raises only
    if every candidate fails.

    Args:
        prompt (str):  Output of build_explanation_prompt(...) or grounded variant
        model (str):   Gemini model id
        api_key (str): Optional override; otherwise uses env vars
        retries (int): Attempts per model for 429/503-style failures

    Returns:
        str: Natural-language explanation text
    """
    client = genai.Client(api_key=_resolve_api_key(api_key))
    last_error = None
    for model_id in _model_candidates(model):
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                logger.info(
                    "Generated explanation (%d chars) with %s", len(text), model_id
                )
                return text
            except Exception as e:
                last_error = e
                msg = str(e)
                missing = "NOT_FOUND" in msg or "404" in msg
                retryable = any(token in msg for token in RETRYABLE_TOKENS)
                if missing:
                    logger.warning("Gemini model %s not found", model_id)
                    break
                if retryable and attempt < retries:
                    wait = 1.5 * attempt
                    logger.warning(
                        "Gemini %s busy (attempt %d/%d); retrying in %.1fs",
                        model_id, attempt, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                if retryable:
                    logger.warning("Gemini model %s unavailable: %s", model_id, e)
                    break
                raise
    raise RuntimeError(f"Gemini explanation failed: {last_error}") from last_error
