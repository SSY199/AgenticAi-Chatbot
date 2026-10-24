# LangGraph RAG Chatbot

Streamlit chat app with a LangGraph agent (Gemini), per-chat PDF RAG (FAISS),
web search, stock prices and a calculator.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
streamlit run app.py
```

## Layout
```
app.py            Streamlit entry point
rag_app/          Backend (no Streamlit imports)
  config.py       paths, models, limits
  db.py           SQLite connections + documents table
  llm.py          LLM + embeddings
  documents.py    PDF ingestion, FAISS, TTL cleanup
  tools.py        search, stock, calculator, rag_tool
  graph.py        LangGraph workflow
  service.py      stream_answer() event stream for the UI
  threads.py      thread list, titles, history, delete
  utils.py        to_text(), thread-id validation
ui/               Frontend
  state.py        session state helpers
  sidebar.py      chat list + PDF panel
  chat.py         chat area + streaming
data/             created at runtime (SQLite DB + FAISS indexes)
```
