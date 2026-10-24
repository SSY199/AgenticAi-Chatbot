"""High-level API the frontend talks to (streaming)."""
from __future__ import annotations

from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .graph import chatbot
from .utils import to_text


def stream_answer(user_input: str, thread_id: str) -> Iterator[dict]:
    """Run one chat turn and yield UI events.

    Event shapes:
      {"type": "text", "content": str}
      {"type": "tool_start", "name": str}
      {"type": "tool_end", "name": str, "content": str}
      {"type": "error", "content": str}
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chatbot_turn",
    }
    try:
        for chunk, meta in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                yield {
                    "type": "tool_end",
                    "name": chunk.name or "tool",
                    "content": to_text(chunk.content),
                }
                continue

            if isinstance(chunk, AIMessage) and meta.get("langgraph_node") == "chat_node":
                names = [
                    tc["name"]
                    for tc in (getattr(chunk, "tool_call_chunks", None) or [])
                    if tc.get("name")
                ] or [tc["name"] for tc in (chunk.tool_calls or []) if tc.get("name")]
                for name in names:
                    yield {"type": "tool_start", "name": name}

                text = to_text(chunk.content)
                if text:
                    yield {"type": "text", "content": text}
    except Exception as exc:  # surface errors in the UI instead of crashing
        yield {"type": "error", "content": str(exc)}
