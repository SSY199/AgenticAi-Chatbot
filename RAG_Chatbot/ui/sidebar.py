"""Sidebar: chat list and PDF panel."""
from __future__ import annotations

import time

import streamlit as st

from rag_app import config
from rag_app.documents import (
    IngestError,
    ingest_pdf,
    list_documents,
    remove_thread_documents,
)
from rag_app.threads import get_thread_title

from .state import new_thread, remove_thread, switch_thread


def _title_for(thread_id: str) -> str:
    titles = st.session_state.titles
    title = titles.get(thread_id) or get_thread_title(thread_id)
    if title:
        titles[thread_id] = title  # only cache real titles
    return title or "New chat"


def _render_threads() -> None:
    st.sidebar.header("Chats")
    if st.sidebar.button("➕ New chat", use_container_width=True):
        new_thread()
        st.rerun()

    current = st.session_state.thread_id
    for tid in list(st.session_state.thread_ids):
        col_title, col_delete = st.sidebar.columns([5, 1])
        if col_title.button(
            _title_for(tid),
            key=f"thread_{tid}",
            use_container_width=True,
            type="primary" if tid == current else "secondary",
        ):
            switch_thread(tid)
            st.rerun()
        if col_delete.button("🗑", key=f"del_{tid}", help="Delete this chat"):
            remove_thread(tid)
            st.rerun()

    with st.sidebar.expander("Debug"):
        st.code(current)


def _render_documents() -> None:
    ss = st.session_state
    tid = ss.thread_id

    st.sidebar.divider()
    st.sidebar.header("📄 Documents")

    docs = list_documents(tid)
    for d in docs:
        expires = time.strftime("%d %b, %H:%M", time.localtime(d["updated_at"] + config.TTL_SECONDS))
        st.sidebar.success(
            f"**{d['filename']}**\n\n{d['pages']} pages · {d['chunks']} chunks\n\nExpires {expires}"
        )
    if docs:
        if st.sidebar.button("Remove all PDFs", use_container_width=True):
            remove_thread_documents(tid)
            ss.processed_uploads = {k for k in ss.processed_uploads if not k.startswith(f"{tid}:")}
            st.rerun()
    else:
        st.sidebar.info("No PDF uploaded for this chat yet.")

    if len(docs) >= config.MAX_FILES_PER_THREAD:
        st.sidebar.caption(f"Limit reached ({config.MAX_FILES_PER_THREAD} PDFs per chat).")
        return

    files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{tid}",  # one uploader per chat
    )

    changed = False
    for f in files or []:
        marker = f"{tid}:{getattr(f, 'file_id', f.name)}"
        if marker in ss.processed_uploads:
            continue
        ss.processed_uploads.add(marker)
        try:
            with st.sidebar.status(f"Indexing {f.name}…"):
                ingest_pdf(f.getvalue(), tid, f.name)
            changed = True
        except IngestError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.sidebar.error(f"Unexpected error while indexing {f.name}: {exc}")
    if changed:
        st.rerun()


def render_sidebar() -> None:
    _render_threads()
    _render_documents()
