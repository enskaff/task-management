"""In-memory context store for the PMO agent."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Iterable, List, Sequence

logger = logging.getLogger(__name__)


_MAX_ITEMS = 20
_MAX_CONTENT_LENGTH = 10_000
_PREVIEW_LENGTH = 160

_CHAT_MESSAGE_LIMIT = 40
_CHAT_STORAGE_LIMIT = 100
_CHAT_MAX_CHARS = 12_000
_CHAT_MESSAGE_CHAR_LIMIT = 4_000
_DOC_CONTEXT_CHAR_LIMIT = 6_000
_DOC_SNIPPET_LENGTH = 600
_TOTAL_PROMPT_LIMIT = 20_000


@dataclass
class _MemoryItem:
    label: str
    content: str

    @property
    def preview(self) -> str:
        """Return a short preview of the stored content."""
        return (self.content[:_PREVIEW_LENGTH] + "…") if len(self.content) > _PREVIEW_LENGTH else self.content


@dataclass
class _MemoryStore:
    _items: List[_MemoryItem] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def add_text(self, label: str, content: str) -> None:
        """Add text to the in-memory store."""
        if not label:
            raise ValueError("Label must be provided for memory entries.")
        if content is None:
            raise ValueError("Content must be provided for memory entries.")

        trimmed_content = content.strip()
        if not trimmed_content:
            raise ValueError("Content must be a non-empty string.")

        if len(trimmed_content) > _MAX_CONTENT_LENGTH:
            logger.debug("Trimming content for label %s to %s characters", label, _MAX_CONTENT_LENGTH)
            trimmed_content = trimmed_content[:_MAX_CONTENT_LENGTH]

        item = _MemoryItem(label=label, content=trimmed_content)

        with self._lock:
            self._items.insert(0, item)
            if len(self._items) > _MAX_ITEMS:
                dropped = self._items.pop()
                logger.info("Memory capacity reached. Dropping oldest item with label %s", dropped.label)

        logger.info("Added memory item with label %s (chars=%s)", label, len(trimmed_content))

    def reset(self) -> None:
        """Reset the memory store."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
        logger.info("Memory reset. Removed %s items", count)

    def list_items(self) -> List[Dict[str, str]]:
        """Return stored items with label and preview for UI/API responses."""
        with self._lock:
            snapshot = list(self._items)

        return [
            {"label": item.label, "preview": item.preview}
            for item in snapshot
        ]

    def get_context_items(self) -> List[Dict[str, str]]:
        """Return items for constructing LLM context (newest first)."""
        with self._lock:
            snapshot = list(self._items)
        return [
            {"label": item.label, "content": item.content}
            for item in snapshot
        ]


memory_store = _MemoryStore()
_chat_lock = Lock()
_chat_history: Dict[str, List[Dict[str, str]]] = {}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_text(label: str, content: str) -> None:
    """Convenience wrapper to add memory text."""
    memory_store.add_text(label, content)


def reset() -> None:
    """Convenience wrapper to reset the memory store."""
    memory_store.reset()


def list_items() -> List[Dict[str, str]]:
    """Convenience wrapper to list memory entries."""
    return memory_store.list_items()


def get_context_items() -> List[Dict[str, str]]:
    """Convenience wrapper to retrieve context items for LLM usage."""
    return memory_store.get_context_items()


def append_chat(session_id: str, role: str, content: str) -> None:
    """Store a chat message for a session."""

    if not session_id:
        raise ValueError("Session ID is required for chat history.")
    if role not in {"user", "assistant", "system"}:
        raise ValueError("Role must be 'user', 'assistant', or 'system'.")
    if content is None:
        raise ValueError("Content must be provided.")

    trimmed = content.strip()
    if not trimmed:
        logger.debug("Ignoring empty chat message for session %s", session_id)
        return

    if len(trimmed) > _CHAT_MESSAGE_CHAR_LIMIT:
        logger.debug(
            "Trimming chat message for session %s role %s to %s characters",
            session_id,
            role,
            _CHAT_MESSAGE_CHAR_LIMIT,
        )
        trimmed = trimmed[:_CHAT_MESSAGE_CHAR_LIMIT]

    entry = {"role": role, "content": trimmed, "ts": _utc_iso()}

    with _chat_lock:
        history = _chat_history.setdefault(session_id, [])
        history.append(entry)
        if len(history) > _CHAT_STORAGE_LIMIT:
            dropped = len(history) - _CHAT_STORAGE_LIMIT
            del history[0:dropped]
            logger.info(
                "Trimmed %s oldest chat messages for session %s to enforce limits",
                dropped,
                session_id,
            )

    logger.info("Stored chat message for session %s with role %s", session_id, role)


def get_chat(session_id: str, limit_chars: int = _CHAT_MAX_CHARS, max_messages: int = _CHAT_MESSAGE_LIMIT) -> List[Dict[str, str]]:
    """Return the most recent chat messages for a session within limits."""

    if not session_id:
        return []

    with _chat_lock:
        history = list(_chat_history.get(session_id, []))

    if not history:
        return []

    collected: List[Dict[str, str]] = []
    used_chars = 0

    for message in reversed(history):  # Start from the newest
        if len(collected) >= max_messages:
            break

        available = limit_chars - used_chars
        if available <= 0:
            break

        content = message["content"]
        if len(content) > available:
            content = content[-available:]

        collected.append({
            "role": message["role"],
            "content": content,
            "ts": message["ts"],
        })
        used_chars += len(content)

    collected.reverse()
    return collected


def reset_chat(session_id: str) -> None:
    """Remove chat history for a session."""

    if not session_id:
        return

    with _chat_lock:
        existed = session_id in _chat_history
        _chat_history.pop(session_id, None)

    if existed:
        logger.info("Reset chat history for session %s", session_id)


def _build_chat_context(messages: Sequence[Dict[str, str]]) -> str:
    if not messages:
        return ""

    lines = ["### CONTEXT: PRIOR CHAT"]
    for message in reversed(messages):  # Latest first for readability
        lines.append(f"- role: {message['role']} -> {message['content']}")
    return "\n".join(lines)


def _build_doc_context(doc_items: Iterable[Dict[str, str]], budget: int) -> str:
    header = "### CONTEXT: NOTES & DOCS"
    if budget <= len(header):
        return ""

    lines = [header]
    remaining = budget - len(header) - 1  # Reserve space for newline after the header

    for item in doc_items:
        if remaining <= 0:
            break

        label = item.get("label", "doc")
        content = item.get("content", "")
        if not content:
            continue

        snippet_limit = min(_DOC_SNIPPET_LENGTH, remaining)
        snippet = content[:snippet_limit]
        line = f"{label}: {snippet}"
        line_length = len(line)

        if line_length > remaining:
            available = remaining - len(label) - 2
            if available <= 0:
                break
            snippet = snippet[:available]
            if not snippet:
                break
            line = f"{label}: {snippet}"
            line_length = len(line)

        lines.append(line)
        remaining -= line_length + 1  # account for newline

    return "\n".join(lines) if len(lines) > 1 else ""


def build_context_with_memory(
    session_id: str | None,
    user_prompt: str,
    doc_items: Sequence[Dict[str, str]] | None = None,
    total_limit: int = _TOTAL_PROMPT_LIMIT,
) -> str:
    """Compose a prompt including chat history and document memory."""

    if user_prompt is None:
        raise ValueError("User prompt must be provided.")

    doc_items = list(doc_items or get_context_items())
    prompt_section = f"### USER PROMPT\n{user_prompt}"
    divider = "---"

    available_for_context = total_limit - len(prompt_section) - len(divider) - 2
    if available_for_context <= 0:
        logger.debug("Prompt length exceeds total limit; returning prompt section only")
        return prompt_section

    chat_char_budget = max(available_for_context - 64, 0)
    chat_messages = get_chat(session_id or "", limit_chars=min(_CHAT_MAX_CHARS, chat_char_budget)) if chat_char_budget > 0 else []
    chat_block = _build_chat_context(chat_messages)
    context_sections: List[str] = []

    if chat_block:
        chat_length = len(chat_block)
        if chat_length <= available_for_context:
            context_sections.append(chat_block)
            available_for_context -= chat_length + 2
        else:
            logger.debug("Chat context exceeds budget even after trimming; dropping chat block")

    if available_for_context > 0 and doc_items:
        doc_budget = min(_DOC_CONTEXT_CHAR_LIMIT, available_for_context)
        doc_block = _build_doc_context(doc_items, doc_budget)
        if doc_block:
            context_sections.append(doc_block)

    if context_sections:
        context_text = "\n\n".join(context_sections)
        final_prompt = f"{context_text}\n\n{divider}\n{prompt_section}"
        chat_included = chat_block != ""
        docs_included = any(section.startswith("### CONTEXT: NOTES & DOCS") for section in context_sections)
        logger.info(
            "Built composite prompt (chat=%s, docs=%s, total_len=%s)",
            chat_included,
            docs_included,
            len(final_prompt),
        )
        return final_prompt

    logger.debug("No context available; returning prompt section only")
    return prompt_section
