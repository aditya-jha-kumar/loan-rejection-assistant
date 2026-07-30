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

import os
from pathlib import Path

from google import genai

# Optional .env support (project root or src/)
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

DEFAULT_MODEL = "gemini-3.5-flash"


def _resolve_api_key(api_key=None):
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in your environment (or a .env file), or pass api_key=..."
        )
    return key


def generate_explanation(prompt, model=DEFAULT_MODEL, api_key=None):
    """
    Generate an applicant-facing rejection explanation with Gemini.

    Args:
        prompt (str):  Output of build_explanation_prompt(...)
        model (str):   Gemini model id (default: gemini-2.5-flash)
        api_key (str): Optional override; otherwise uses env vars

    Returns:
        str: Natural-language explanation text
    """
    client = genai.Client(api_key=_resolve_api_key(api_key))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text
