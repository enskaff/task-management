"""Prompt utilities for the PMO agent."""
from __future__ import annotations

import os

DEFAULT_SYSTEM_PROMPT = (
    "You are a focused project management office (PMO) assistant. "
    "Provide concise, structured updates, track tasks, and surface risks without inventing facts. "
    "Use bullet lists and short paragraphs when helpful, and ask clarifying questions if context is missing."
)


def get_system_prompt() -> str:
    """Return the configured system prompt, defaulting to the PMO persona."""

    override = os.environ.get("SYSTEM_PROMPT")
    if override and override.strip():
        return override.strip()
    return DEFAULT_SYSTEM_PROMPT
