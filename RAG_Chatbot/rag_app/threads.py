"""Conversation thread helpers: listing, titles, history, deletion."""
from __future__ import annotations

import sqlite3
from typing import Optional

from .db import meta_conn, meta_lock
from .documents import remove_thread_documents
from .graph import chatbot, checkpointer
from .utils import to_text, validate_thread_id


def retrieve_all_threads() -> list[str]:
    """All thread ids, most recently active first."""
    try:
        with meta_lock:
            rows = meta_conn.execute(
                "SELECT thread_id FROM checkpoints "
                "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def _messages(thread_id: str) -> list:
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    if state and state.values:
        return state.values.get("messages", [])
    return []


def get_thread_title(thread_id: str, max_len: int = 28) -> Optional[str]:
    """First user message, shortened. None if the thread has no messages yet."""
    for msg in _messages(thread_id):
        if msg.type == "human":
            text = to_text(msg.content).strip()
            if text:
                return text if len(text) <= max_len else text[:max_len] + "…"
    return None


def load_history(thread_id: str) -> list[dict]:
    """Rebuild the chat transcript (with tool traces) from the checkpoint."""
    history: list[dict] = []
    pending_tools: list[dict] = []
    for msg in _messages(thread_id):
        if msg.type == "human":
            history.append({"role": "user", "content": to_text(msg.content)})
            pending_tools = []
        elif msg.type == "tool":
            pending_tools.append({"name": msg.name or "tool", "content": to_text(msg.content)})
        elif msg.type == "ai":
            if getattr(msg, "tool_calls", None):
                continue
            text = to_text(msg.content)
            if text:
                history.append({"role": "assistant", "content": text, "tools": pending_tools})
                pending_tools = []
    return history


def delete_thread(thread_id: str) -> None:
    """Remove a conversation and its PDFs."""
    thread_id = validate_thread_id(thread_id)
    checkpointer.delete_thread(thread_id)  # needs a recent langgraph-checkpoint-sqlite
    remove_thread_documents(thread_id)
