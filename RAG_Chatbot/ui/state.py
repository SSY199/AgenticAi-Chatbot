"""Session-state helpers: threads, cached titles and histories."""
from __future__ import annotations

import uuid

import streamlit as st

from rag_app.threads import delete_thread, load_history, retrieve_all_threads


def init_session() -> None:
    ss = st.session_state
    ss.setdefault("titles", {})
    ss.setdefault("histories", {})
    ss.setdefault("processed_uploads", set())
    if "thread_ids" not in ss:
        ss.thread_ids = retrieve_all_threads()
    if "thread_id" not in ss:
        new_thread()


def get_history(thread_id: str) -> list[dict]:
    histories = st.session_state.histories
    if thread_id not in histories:
        histories[thread_id] = load_history(thread_id)
    return histories[thread_id]


def new_thread() -> None:
    """Start a fresh chat (re-uses the current one if it is still empty)."""
    ss = st.session_state
    current = ss.get("thread_id")
    if current and not get_history(current):
        return
    tid = str(uuid.uuid4())
    ss.thread_ids.insert(0, tid)
    ss.thread_id = tid


def switch_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id


def remove_thread(thread_id: str) -> None:
    ss = st.session_state
    try:
        delete_thread(thread_id)
    except Exception as exc:
        st.sidebar.error(f"Could not delete chat: {exc}")
        return
    if thread_id in ss.thread_ids:
        ss.thread_ids.remove(thread_id)
    ss.histories.pop(thread_id, None)
    ss.titles.pop(thread_id, None)

    if ss.thread_id == thread_id:
        if ss.thread_ids:
            ss.thread_id = ss.thread_ids[0]
        else:
            del ss["thread_id"]
            new_thread()
