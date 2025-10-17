"""Utilities for interacting with the Google Gemini API."""
from __future__ import annotations

import logging
import os
from typing import Optional

import google.generativeai as genai

from src.pmo_agent import memory

logger = logging.getLogger(__name__)

_MODEL_NAME = "models/gemini-2.0-flash-lite"
_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
_model: Optional[genai.GenerativeModel] = None
_initialization_error: Optional[Exception] = None
_MAX_PROMPT_CHARS = 20_000
_CONTEXT_SNIPPET_LENGTH = 1_000

if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        _model = genai.GenerativeModel(model_name=_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - defensive logging
        _initialization_error = exc


def _apply_memory_context(prompt: str) -> str:
    """Inject memory items into the prompt while respecting size limits."""

    context_items = memory.get_context_items()
    if not context_items:
        return prompt

    # Reserve space for headers and the user prompt.
    header = "### CONTEXT\n"
    separator = "\n---\n### USER PROMPT\n"
    remaining_budget = _MAX_PROMPT_CHARS - len(header) - len(separator) - len(prompt)
    if remaining_budget <= 0:
        return prompt

    lines = ["### CONTEXT"]
    used = 0
    for item in context_items:
        snippet = item["content"][:_CONTEXT_SNIPPET_LENGTH]
        line = f"{item['label']}: {snippet}"
        line_length = len(line) + 1  # Account for newline when joining.

        if used + line_length > remaining_budget:
            # Try to trim the snippet to fit into the remaining space.
            available = max(remaining_budget - used - len(item["label"]) - 2, 0)
            if available <= 0:
                break
            trimmed_line = f"{item['label']}: {snippet[:available]}"
            if trimmed_line.strip():
                lines.append(trimmed_line)
            break

        lines.append(line)
        used += line_length

    if len(lines) == 1:  # No context added.
        return prompt

    context_block = "\n".join(lines)
    logger.info("Including %s memory items in LLM prompt (final length=%s)", len(lines) - 1, len(context_block) + len(separator) + len(prompt))
    return f"{context_block}{separator}{prompt}"


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

    final_prompt = _apply_memory_context(prompt)

    try:
        response = _model.generate_content(final_prompt)
    except Exception as exc:  # pragma: no cover - actual API errors are external
        raise RuntimeError("Gemini API request failed") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini API returned an empty response.")

    return text
