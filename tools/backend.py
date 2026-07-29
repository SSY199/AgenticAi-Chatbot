from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import random
import sqlite3
import os


load_dotenv()

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
      
      
tools = [calculator, get_stock_price, search_tool]
llm_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
def chat_node(state: ChatState): 
    """LLM node that may answer a question or call a tool based on the input messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
  
tool_node = ToolNode(tools)

#Checkpointer

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# Define Graph State
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")             


chatbot = graph.compile(checkpointer=checkpointer)

#7. Helper

def retrieve_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)

