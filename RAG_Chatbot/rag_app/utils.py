"""Small helpers shared by backend and frontend."""
from __future__ import annotations

import uuid
from typing import Any


def to_text(content: Any) -> str:
    """Normalise LangChain message content to plain text.

    Gemini can return a list of content blocks instead of a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def validate_thread_id(thread_id: Any) -> str:
    """Return the thread id as a canonical UUID string or raise ValueError.

    Thread ids become folder names, so anything that is not a UUID is rejected
    (blocks path traversal such as '../../etc').
    """
    try:
        return str(uuid.UUID(str(thread_id)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid thread id.") from exc
