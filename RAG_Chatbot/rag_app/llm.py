"""LLM and embedding model instances."""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=config.LLM_TEMPERATURE,
)

# Small open-source model that runs locally.
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
