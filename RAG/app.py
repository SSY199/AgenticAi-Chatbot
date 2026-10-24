"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="LangGraph RAG Chatbot", page_icon="💬", layout="wide")

from rag_app.documents import cleanup_expired  # noqa: E402
from ui.chat import render_chat  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.state import init_session  # noqa: E402

cleanup_expired()  # throttled; removes PDFs unused for 24h
init_session()
render_sidebar()

st.title("💬 LangGraph RAG Chatbot")
render_chat()
