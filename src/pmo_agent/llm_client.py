"""Utilities for interacting with the Google Gemini API."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import google.generativeai as genai

from src.pmo_agent import memory

logger = logging.getLogger(__name__)

_MODEL_NAME = "models/gemini-2.0-flash-lite"
_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
_model: Optional[genai.GenerativeModel] = None
_initialization_error: Optional[Exception] = None
_MAX_PROMPT_CHARS = 20_000

if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        _model = genai.GenerativeModel(model_name=_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - defensive logging
        _initialization_error = exc


def generate_response(prompt: str, session_id: str | None = None) -> str:
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

    doc_items = memory.get_context_items()
    final_prompt = memory.build_context_with_memory(
        session_id=session_id,
        user_prompt=prompt,
        doc_items=doc_items,
        total_limit=_MAX_PROMPT_CHARS,
    )
    logger.debug("Final prompt length after memory merge: %s", len(final_prompt))

    try:
        response = _model.generate_content(final_prompt)
    except Exception as exc:  # pragma: no cover - actual API errors are external
        raise RuntimeError("Gemini API request failed") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini API returned an empty response.")

    return text


def chat_complete(system_prompt: str, messages: List[dict]) -> str:
    """Run a short chat completion with the Gemini model."""

    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt must be provided.")

    if _initialization_error is not None:
        raise RuntimeError("Failed to initialize Gemini client") from _initialization_error

    if _model is None:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please configure the environment variable before using the LLM."
        )

    sanitized_messages: List[dict] = []
    for message in messages or []:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            logger.debug("Skipping malformed message entry: %s", message)
            continue
        sanitized_messages.append({"role": role, "content": content})

    conversation_lines = [f"System: {system_prompt.strip()}"]
    for entry in sanitized_messages:
        prefix = "User" if entry["role"] == "user" else "Assistant"
        conversation_lines.append(f"{prefix}: {entry['content']}")

    conversation_text = "\n\n".join(conversation_lines)
    if len(conversation_text) > _MAX_PROMPT_CHARS:
        logger.debug("Chat prompt exceeds limit; trimming to %s characters", _MAX_PROMPT_CHARS)
        conversation_text = (
            conversation_lines[0]
            + "\n\n"
            + conversation_text[-(_MAX_PROMPT_CHARS - len(conversation_lines[0]) - 2):]
        )

    try:
        response = _model.generate_content(conversation_text)
    except Exception as exc:  # pragma: no cover - actual API errors are external
        raise RuntimeError("Gemini API request failed") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini API returned an empty response.")

    return text
