"""Tools the LLM can call."""
from __future__ import annotations

import operator
from typing import Literal

import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from .config import ALPHAVANTAGE_API_KEY
from .documents import list_documents, search_documents

search_tool = TavilySearch(max_results=3, topic="general", search_depth="advanced")

_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


@tool
def calculator(
    first_num: float, second_num: float, operation: Literal["add", "sub", "mul", "div"]
) -> dict:
    """Perform a basic arithmetic operation on two numbers."""
    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed"}
    result = _OPS[operation](first_num, second_num)
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a ticker symbol such as 'AAPL' or 'TSLA'."""
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "ALPHAVANTAGE_API_KEY is not configured."}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHAVANTAGE_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"error": f"Stock lookup failed: {exc}"}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Search the PDF(s) uploaded in this chat and return the most relevant passages."""
    # thread_id comes from the graph config, never from the LLM.
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if not thread_id or not list_documents(thread_id):
        return {"error": "No document indexed for this chat. Ask the user to upload a PDF."}

    docs = search_documents(thread_id, query)
    return {
        "query": query,
        "results": [
            {
                "text": d.page_content,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
            }
            for d in docs
        ],
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
