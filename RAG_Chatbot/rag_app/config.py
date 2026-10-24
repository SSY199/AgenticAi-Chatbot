"""Central configuration: paths, model names, limits, API keys."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "faiss_stores"
DB_PATH = DATA_DIR / "chatbot.db"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Models ----------------------------------------------------------------
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2  # low = more reliable tool arguments and grounded answers
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- API keys (all read from .env) -----------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
# TAVILY_API_KEY is picked up automatically by langchain-tavily from the env.

# ---- RAG / ingestion -------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 20  # candidate pool for MMR re-ranking

# ---- Guard limits ----------------------------------------------------------
MAX_FILES_PER_THREAD = 3
MAX_PAGES_PER_FILE = 100
MAX_CHUNKS_PER_THREAD = 5000  # cumulative across all files in a thread
TTL_SECONDS = 24 * 60 * 60  # an index expires after 24h without use
CLEANUP_INTERVAL_SECONDS = 10 * 60  # how often expired indexes are swept

# ---- Conversation ----------------------------------------------------------
MAX_HISTORY_MESSAGES = 30  # messages sent to the LLM each turn
