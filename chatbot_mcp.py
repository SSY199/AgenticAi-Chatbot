import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": ["C:/Users/sahil/LangGraph/mcp-math-server/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


# Define graph State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Build the state graph workflow
async def build_graph():
  
    tools = await client.get_tools()
    print(f"Tools retrieved from MCP server: {tools}")
    llm_with_tools = model.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Define standard tool workflow node
    tool_node = ToolNode(tools)

    # Initialize graph framework with defined schema
    graph = StateGraph(ChatState)

    # Register workflow nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Set up operational logic routes
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    # Compile structure into an executable agent
    chatbot = graph.compile()
    return chatbot
  
  
async def main():
    chatbot = await build_graph()

    # Query input structure wrapped in the defined state schema
    inputs = {"messages": [HumanMessage(content="Add an expense of 500 dollars for office supplies")]}
    result = await chatbot.ainvoke(inputs)

    # Safely print out the final response in the message chain array
    print("\n--- Final Output ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())