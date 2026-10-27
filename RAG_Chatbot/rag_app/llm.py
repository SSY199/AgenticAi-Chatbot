"""LLM and embedding model instances."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

print("LLM_MODEL:", repr(config.LLM_MODEL))
print("GEMINI_API_KEY configured:", bool(config.GEMINI_API_KEY))
print("LLM_TEMPERATURE:", repr(config.LLM_TEMPERATURE))
print("TEMPERATURE TYPE:", type(config.LLM_TEMPERATURE))

llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=config.LLM_TEMPERATURE,
)

# Small open-source model that runs locally.
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
