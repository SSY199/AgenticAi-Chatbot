"""Main chat area: history, suggestions and streaming a new answer."""
from __future__ import annotations

import streamlit as st

from rag_app.documents import list_documents
from rag_app.service import stream_answer

from .state import get_history

SUGGESTIONS = [
    "Summarize this document",
    "What are the key topics covered?",
    "List the important definitions",
]


def _clip(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def _render_tools(tools: list[dict]) -> None:
    if not tools:
        return
    with st.expander(f"🔧 Used {len(tools)} tool call(s)"):
        for t in tools:
            st.markdown(f"**{t['name']}**")
            st.code(_clip(t["content"]), language="json")


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_tools(msg.get("tools", []))
        st.markdown(msg["content"])


def _run_turn(prompt: str, thread_id: str, history: list[dict]) -> None:
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tools_used: list[dict] = []
    state = {"box": None, "error": False}

    def set_status(label: str) -> None:
        if state["box"] is None:
            state["box"] = st.status(label, expanded=True)
        else:
            state["box"].update(label=label, state="running", expanded=True)

    def token_stream():
        for event in stream_answer(prompt, thread_id):
            kind = event["type"]
            if kind == "tool_start":
                set_status(f"🔧 Calling **{event['name']}**…")
            elif kind == "tool_end":
                tools_used.append({"name": event["name"], "content": event["content"]})
                set_status(f"🔧 Finished **{event['name']}**")
                state["box"].code(_clip(event["content"]), language="json")
            elif kind == "text":
                yield event["content"]
            elif kind == "error":
                state["error"] = True
                st.error(f"Something went wrong: {event['content']}")

    with st.chat_message("assistant"):
        answer = st.write_stream(token_stream())
        if state["box"] is not None:
            state["box"].update(label="✅ Tools finished", state="complete", expanded=False)

    answer = answer if isinstance(answer, str) else ""
    if answer or tools_used:
        history.append({"role": "assistant", "content": answer, "tools": tools_used})
    if not state["error"]:
        st.rerun()  # refresh sidebar titles; history re-renders with tool traces


def render_chat() -> None:
    ss = st.session_state
    thread_id = ss.thread_id
    history = get_history(thread_id)

    for msg in history:
        _render_message(msg)

    # Suggested prompts once a PDF is indexed and the chat is still empty.
    if not history and list_documents(thread_id):
        st.caption("Try asking:")
        cols = st.columns(len(SUGGESTIONS))
        for i, text in enumerate(SUGGESTIONS):
            if cols[i].button(text, key=f"suggest_{i}", use_container_width=True):
                ss.pending_prompt = text
                st.rerun()

    prompt = st.chat_input("Type your message here...") or ss.pop("pending_prompt", None)
    if prompt:
        _run_turn(prompt, thread_id, history)
