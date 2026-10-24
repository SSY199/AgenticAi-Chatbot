"""SQLite connections.

Two separate connections to the same WAL-mode database file:
  * checkpoint_conn -> used only by LangGraph's SqliteSaver
  * meta_conn       -> document metadata + thread queries (guarded by meta_lock)
"""
from __future__ import annotations

import sqlite3
import threading

from .config import DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


checkpoint_conn = _connect()
meta_conn = _connect()
meta_lock = threading.Lock()

with meta_lock, meta_conn:
    meta_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            thread_id  TEXT NOT NULL,
            filename   TEXT NOT NULL,
            pages      INTEGER NOT NULL,
            chunks     INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (thread_id, filename)
        )
        """
    )
