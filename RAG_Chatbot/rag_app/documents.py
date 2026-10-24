"""PDF ingestion, per-thread FAISS indexes, metadata and TTL cleanup."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .db import meta_conn, meta_lock
from .llm import embeddings
from .utils import validate_thread_id


class IngestError(ValueError):
    """Raised for user-facing ingestion problems (limits, bad PDFs...)."""


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

_store_cache: dict[str, FAISS] = {}
_cache_lock = threading.Lock()
_last_cleanup = 0.0


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------
def list_documents(thread_id: str) -> list[dict[str, Any]]:
    with meta_lock:
        rows = meta_conn.execute(
            "SELECT filename, pages, chunks, updated_at FROM documents "
            "WHERE thread_id = ? ORDER BY updated_at",
            (str(thread_id),),
        ).fetchall()
    return [
        {"filename": r[0], "pages": r[1], "chunks": r[2], "updated_at": r[3]}
        for r in rows
    ]


def thread_has_document(thread_id: str) -> bool:
    return bool(list_documents(thread_id))


def _touch(thread_id: str) -> None:
    """Reset the TTL countdown for a thread that is being used."""
    with meta_lock, meta_conn:
        meta_conn.execute(
            "UPDATE documents SET updated_at = ? WHERE thread_id = ?",
            (time.time(), thread_id),
        )


def _thread_dir(thread_id: str) -> Path:
    return config.VECTOR_STORE_DIR / validate_thread_id(thread_id)


# --------------------------------------------------------------------------
# Index storage
# --------------------------------------------------------------------------
def _save_atomically(store: FAISS, target: Path) -> None:
    """Write to a temp folder first so a failure never leaves a half-written index."""
    tmp_dir = config.VECTOR_STORE_DIR / f".tmp-{uuid.uuid4().hex}"
    try:
        store.save_local(str(tmp_dir))
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp_dir, target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_store(thread_id: str) -> FAISS | None:
    thread_id = validate_thread_id(thread_id)
    with _cache_lock:
        cached = _store_cache.get(thread_id)
    if cached is not None:
        return cached

    target = _thread_dir(thread_id)
    if not target.exists():
        return None
    try:
        # Safe: these index files are created by this app, never user-supplied.
        store = FAISS.load_local(
            str(target), embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        return None
    with _cache_lock:
        _store_cache[thread_id] = store
    return store


def _invalidate(thread_id: str) -> None:
    with _cache_lock:
        _store_cache.pop(thread_id, None)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict[str, Any]:
    """Index a PDF into the thread's FAISS store (merging with existing files)."""
    if not file_bytes:
        raise IngestError("The uploaded file is empty.")

    thread_id = validate_thread_id(thread_id)
    filename = os.path.basename(filename or "document.pdf")
    cleanup_expired()

    existing = list_documents(thread_id)
    if any(d["filename"] == filename for d in existing):
        raise IngestError(f"'{filename}' is already indexed in this chat.")
    if len(existing) >= config.MAX_FILES_PER_THREAD:
        raise IngestError(
            f"Upload limit reached: max {config.MAX_FILES_PER_THREAD} PDFs per chat."
        )
    used_chunks = sum(d["chunks"] for d in existing)

    # --- load -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages: list[Document] = PyPDFLoader(tmp_path).load()
    except Exception as exc:
        raise IngestError("Could not read this file. Is it a valid PDF?") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    total_pages = len(pages)
    if total_pages > config.MAX_PAGES_PER_FILE:
        raise IngestError(f"PDF has {total_pages} pages; max is {config.MAX_PAGES_PER_FILE}.")

    # Clean metadata: real filename + 1-based page number.
    for page in pages:
        page.metadata = {
            "source": filename,
            "page": int(page.metadata.get("page", 0)) + 1,
        }
    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise IngestError("No extractable text found (scanned PDFs need OCR first).")

    # --- split ------------------------------------------------------------
    chunks = _splitter.split_documents(pages)
    if used_chunks + len(chunks) > config.MAX_CHUNKS_PER_THREAD:
        raise IngestError(
            f"Chunk limit exceeded ({config.MAX_CHUNKS_PER_THREAD} per chat)."
        )

    # --- embed + persist ----------------------------------------------------
    new_store = FAISS.from_documents(chunks, embeddings)
    target = _thread_dir(thread_id)
    if target.exists():
        store = FAISS.load_local(
            str(target), embeddings, allow_dangerous_deserialization=True
        )
        store.merge_from(new_store)
    else:
        store = new_store
    _save_atomically(store, target)
    _invalidate(thread_id)

    with meta_lock, meta_conn:
        meta_conn.execute(
            "INSERT OR REPLACE INTO documents (thread_id, filename, pages, chunks, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, filename, total_pages, len(chunks), time.time()),
        )
    _touch(thread_id)

    return {"filename": filename, "pages": total_pages, "chunks": len(chunks)}


def search_documents(thread_id: str, query: str) -> list[Document]:
    """MMR search over the thread's PDFs. Returns [] if nothing is indexed."""
    store = _load_store(thread_id)
    if store is None:
        return []
    _touch(validate_thread_id(thread_id))
    return store.max_marginal_relevance_search(
        query, k=config.RETRIEVER_K, fetch_k=config.RETRIEVER_FETCH_K
    )


def remove_thread_documents(thread_id: str) -> None:
    thread_id = validate_thread_id(thread_id)
    with meta_lock, meta_conn:
        meta_conn.execute("DELETE FROM documents WHERE thread_id = ?", (thread_id,))
    shutil.rmtree(_thread_dir(thread_id), ignore_errors=True)
    _invalidate(thread_id)


def cleanup_expired(force: bool = False) -> None:
    """Delete indexes unused for longer than the TTL (throttled)."""
    global _last_cleanup
    now = time.time()
    if not force and now - _last_cleanup < config.CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup = now

    cutoff = now - config.TTL_SECONDS
    with meta_lock:
        rows = meta_conn.execute(
            "SELECT thread_id FROM documents GROUP BY thread_id HAVING MAX(updated_at) < ?",
            (cutoff,),
        ).fetchall()
    for (tid,) in rows:
        remove_thread_documents(tid)
