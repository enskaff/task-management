"""Utilities for interacting with the Google Gemini API."""
from __future__ import annotations

import os
from typing import Optional

import google.generativeai as genai

_MODEL_NAME = "models/gemini-2.0-flash-lite"
_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
_model: Optional[genai.GenerativeModel] = None
_initialization_error: Optional[Exception] = None

if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        _model = genai.GenerativeModel(model_name=_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - defensive logging
        _initialization_error = exc


def generate_response(prompt: str) -> str:
    """Generate a text response from the Gemini model.

    Args:
        prompt: The user prompt to send to Gemini.

    Returns:
        The generated text response from Gemini.

    Raises:
        ValueError: If ``prompt`` is empty.
        EnvironmentError: If the API key is missing.
        RuntimeError: If the Gemini client fails to initialize or the request fails.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if _initialization_error is not None:
        raise RuntimeError("Failed to initialize Gemini client") from _initialization_error

    if _model is None:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please configure the environment variable before using the LLM."
        )

    try:
        response = _model.generate_content(prompt)
    except Exception as exc:  # pragma: no cover - actual API errors are external
        raise RuntimeError("Gemini API request failed") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini API returned an empty response.")

    return text
