from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# --- Updated Imports for Gemini and HuggingFace ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -------------------
# 1. LLM + embeddings (Updated to Gemini & HuggingFace)
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

# Uses an industry-standard, lightweight, open-source model locally
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -------------------
# 2. SQLite Database & Storage Setup
# -------------------
DB_PATH = "chatbot.db"
VECTOR_STORE_DIR = "./faiss_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Guard limits to prevent memory exhaustion and DoS
MAX_FILES_PER_THREAD = 3
MAX_PAGES_PER_FILE = 100
MAX_CHUNKS_PER_THREAD = 5000
TTL_SECONDS = 86400  # 24 Hours time-to-live

# Establish Connection and initialize document structure table
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)

with conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            thread_id TEXT PRIMARY KEY,
            filename TEXT,
            documents_count INTEGER,
            chunks_count INTEGER,
            updated_at REAL
        )
    """)


def cleanup_expired_threads():
    """Deletes physical FAISS assets and DB meta records older than the TTL limit."""
    cutoff_time = time.time() - TTL_SECONDS
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM document_metadata WHERE updated_at < ?", (cutoff_time,))
    expired_threads = cursor.fetchall()
    
    for (expired_id,) in expired_threads:
        thread_dir = os.path.join(VECTOR_STORE_DIR, expired_id)
        if os.path.exists(thread_dir):
            shutil.rmtree(thread_dir)
        
        cursor.execute("DELETE FROM document_metadata WHERE thread_id = ?", (expired_id,))
    conn.commit()


def _get_retriever(thread_id: Optional[str]):
    """Lazy loads the FAISS index from disk when needed and updates its TTL."""
    if not thread_id:
        return None
        
    # Check for memory leaks before parsing indices
    cleanup_expired_threads()
    
    thread_dir = os.path.join(VECTOR_STORE_DIR, str(thread_id))
    if os.path.exists(thread_dir):
        try:
            # Refresh timestamp to reset TTL countdown on active usage
            with conn:
                conn.execute(
                    "UPDATE document_metadata SET updated_at = ? WHERE thread_id = ?",
                    (time.time(), str(thread_id))
                )
            
            vector_store = FAISS.load_local(
                thread_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        except Exception:
            return None
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Builds a disk-persisted FAISS retriever for the uploaded PDF constrained by strict limits.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
        
    thread_id_str = str(thread_id)
    cleanup_expired_threads()

    # Enforce file limit checkpoint per thread
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM document_metadata WHERE thread_id = ?", (thread_id_str,))
    if cursor.fetchone()[0] >= MAX_FILES_PER_THREAD:
        raise ValueError(f"Upload limit reached. Maximum {MAX_FILES_PER_THREAD} documents allowed per thread.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Enforce page limit safety checkpoint
        if len(docs) > MAX_PAGES_PER_FILE:
            raise ValueError(f"PDF length exceeds safety constraints. Maximum {MAX_PAGES_PER_FILE} pages allowed.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        
        # Enforce total chunk safety constraint
        if len(chunks) > MAX_CHUNKS_PER_THREAD:
            raise ValueError(f"Document complexity too high. Total chunks exceed limit of {MAX_CHUNKS_PER_THREAD}.")

        # Persist index assets safely to a disk folder named after the target thread ID
        thread_dir = os.path.join(VECTOR_STORE_DIR, thread_id_str)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(thread_dir)

        resolved_name = filename or os.path.basename(temp_path)
        
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_metadata (thread_id, filename, documents_count, chunks_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id_str, resolved_name, len(docs), len(chunks), time.time())
            )

        return {
            "filename": resolved_name,
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = TavilySearch(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY"),
    engine_id=os.getenv("TAVILY_ENGINE_ID"),
    topic="general",
    search_type="web",
    search_depth="advanced"
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM document_metadata WHERE thread_id = ?",
        (str(thread_id),),
    )

    return cursor.fetchone() is not None


def thread_document_metadata(thread_id: str) -> dict:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT filename, documents_count, chunks_count
        FROM document_metadata
        WHERE thread_id = ?
        """,
        (str(thread_id),),
    )

    row = cursor.fetchone()

    if row:
        return {
            "filename": row[0],
            "documents": row[1],
            "chunks": row[2],
        }

    return {}