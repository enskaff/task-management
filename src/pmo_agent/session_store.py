"""Session-scoped context and chat history storage."""
from __future__ import annotations

import logging
from collections import deque
from threading import Lock
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 12
_MAX_CONTEXT_CHARS = 20_000
_MAX_MESSAGE_CHARS = 4_000

_context_store: Dict[str, str] = {}
_history_store: Dict[str, Deque[dict]] = {}
_lock = Lock()


def _normalize_text(text: str, limit: int | None = None) -> str:
    trimmed = (text or "").strip()
    if limit is not None and len(trimmed) > limit:
        logger.debug("Trimming text to %s characters", limit)
        return trimmed[:limit]
    return trimmed


def set_context(session_id: str, text: str) -> None:
    """Store the latest uploaded context for the session."""

    if not session_id:
        raise ValueError("Session ID is required to store context.")

    cleaned = _normalize_text(text, _MAX_CONTEXT_CHARS)
    with _lock:
        _context_store[session_id] = cleaned
    logger.info("Updated context for session %s (chars=%s)", session_id, len(cleaned))


def get_context(session_id: str) -> str | None:
    """Retrieve the stored context for the session, if any."""

    if not session_id:
        return None

    with _lock:
        return _context_store.get(session_id)


def _append_message(session_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("Role must be 'user' or 'assistant'.")

    if not session_id:
        raise ValueError("Session ID is required to append chat messages.")

    cleaned = _normalize_text(content, _MAX_MESSAGE_CHARS)
    if not cleaned:
        logger.debug("Ignoring empty %s message for session %s", role, session_id)
        return

    with _lock:
        history = _history_store.setdefault(session_id, deque(maxlen=_MAX_HISTORY_MESSAGES))
        history.append({"role": role, "content": cleaned})
    logger.info("Stored %s message for session %s", role, session_id)


def append_user(session_id: str, content: str) -> None:
    """Record a user message."""

    _append_message(session_id, "user", content)


def append_assistant(session_id: str, content: str) -> None:
    """Record an assistant message."""

    _append_message(session_id, "assistant", content)


def get_history(session_id: str, max_msgs: int = _MAX_HISTORY_MESSAGES) -> List[dict]:
    """Return the most recent chat messages for the session."""

    if not session_id:
        return []

    with _lock:
        history = list(_history_store.get(session_id, ()))

    if max_msgs < len(history):
        history = history[-max_msgs:]
    return history
