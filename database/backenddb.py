from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Load API key
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)


# Define Graph State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Node
def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
  

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)


# checkpointer
checkpointer = SqliteSaver(conn=conn)

# Build Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)