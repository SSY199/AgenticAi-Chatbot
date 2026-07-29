from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import BaseTool

from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os

from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import random
import sqlite3
import os


load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
  
def run_async(coro):
    return _submit_async(coro).result()
   
def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop"""
    return _submit_async(coro)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

search_tool = TavilySearch(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY"),
    engine_id=os.getenv("TAVILY_ENGINE_ID"),
    topic="general",
    search_type="web",
    search_depth="advanced"
)


client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": ["C:/Users/sahil/LangGraph/mcp-math-server/main.py"],
        },
        # "expense": {
        #     "transport": "streamable_http",
        #     "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        # }
    }
)

@tool
def calculator(first_num: float, seconf_num: float, operation: str) -> dict:
    """A simple calculator tool that performs basic arithmetic operations."""
    
    try:
        if operation == "add":
            result = first_num + seconf_num
        elif operation == "subtract":
            result = first_num - seconf_num
        elif operation == "multiply":
            result = first_num * seconf_num
        elif operation == "divide":
            if seconf_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / seconf_num
        else:
            return {"error": f"Invalid operation: {operation}. Supported operations are add, subtract, multiply, divide."}
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
      
      
# pip install yfinance
import yfinance as yf

@tool
def get_stock_price(ticker: str) -> dict:
    """Fetches the current stock price for a given ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        # Get the most recent price
        price = stock.history(period="1d")['Close'].iloc[-1]
        return {"ticker": ticker.upper(), "price": float(price), "currency": "USD"}
    except Exception as e:
        return {"error": str(e)}  
      
      
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        return []
      
mcp_tools = load_mcp_tools()
      
      
tools = [*mcp_tools, get_stock_price, search_tool]
llm_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
async def chat_node(state: ChatState): 
    """LLM node that may answer a question or call a tool based on the input messages."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}
  
tool_node = ToolNode(tools) if tools else None

#Checkpointer

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer()) 

# Define Graph State
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node") 
else :
    graph.add_edge("chat_node", END)
    
chatbot = graph.compile(checkpointer=checkpointer)

#7. Helper

async def _alist_threads():
  all_threads = set()
  async for checkpoint in checkpointer.alist(None):
      all_threads.add(checkpoint.config['configurable']['thread_id'])
  return list(all_threads)

def retrieve_all_thread():
        
    return run_async(_alist_threads())

