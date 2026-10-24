"""LangGraph workflow: chat node <-> tool node, with SQLite checkpointing."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import MAX_HISTORY_MESSAGES
from .db import checkpoint_conn
from .documents import list_documents
from .llm import llm
from .tools import tools

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _system_prompt(thread_id: str | None) -> SystemMessage:
    docs = list_documents(thread_id) if thread_id else []
    if docs:
        names = ", ".join(d["filename"] for d in docs)
        doc_rule = (
            f"The user has uploaded: {names}. For questions about these documents, "
            "call `rag_tool` and cite the file name and page number of what you use. "
            "If the passages do not answer the question, say so instead of guessing."
        )
    else:
        doc_rule = (
            "No PDF is uploaded in this chat. If the user asks about a document, "
            "ask them to upload one."
        )
    return SystemMessage(
        content=(
            "You are a helpful assistant. You can also use web search, stock price "
            "and calculator tools when helpful. " + doc_rule
        )
    )


def chat_node(state: ChatState, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    history = trim_messages(
        state["messages"],
        max_tokens=MAX_HISTORY_MESSAGES,
        strategy="last",
        token_counter=len,  # count messages, not tokens
        start_on="human",
        include_system=False,
    )
    response = llm_with_tools.invoke([_system_prompt(thread_id), *history], config=config)
    return {"messages": [response]}


checkpointer = SqliteSaver(conn=checkpoint_conn)
checkpointer.setup()

_graph = StateGraph(ChatState)
_graph.add_node("chat_node", chat_node)
_graph.add_node("tools", ToolNode(tools))
_graph.add_edge(START, "chat_node")
_graph.add_conditional_edges("chat_node", tools_condition)
_graph.add_edge("tools", "chat_node")

chatbot = _graph.compile(checkpointer=checkpointer)
