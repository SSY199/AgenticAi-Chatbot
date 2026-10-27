"""Central configuration: paths, model names, limits, API keys."""
from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "faiss_stores"
DB_PATH = DATA_DIR / "chatbot.db"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load Environment Variables (Local Dev Fallback) -----------------------
# This explicitly searches your RAG_Chatbot folder for the .env file
local_env = BASE_DIR / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env)
else:
    load_dotenv()

# Helper function to check environment variables first, then Streamlit Cloud Secrets
def get_secret(key: str) -> str | None:
    return os.getenv(key) or st.secrets.get(key)

# ---- Models ----------------------------------------------------------------
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2  # low = more reliable tool arguments and grounded answers
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- API keys (Checks .env locally, Secrets on Streamlit Cloud) ------------
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ALPHAVANTAGE_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# Sync Tavily to os.environ so LangChain can discover it automatically on the cloud
if "TAVILY_API_KEY" not in os.environ and "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

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
