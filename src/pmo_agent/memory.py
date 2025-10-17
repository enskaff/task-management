"""In-memory context store for the PMO agent."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Dict

logger = logging.getLogger(__name__)


_MAX_ITEMS = 20
_MAX_CONTENT_LENGTH = 10_000
_PREVIEW_LENGTH = 160


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
