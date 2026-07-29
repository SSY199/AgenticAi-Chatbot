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

# Load environment variables
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)


# Define tools as async functions for clean asynchronous node execution
@tool
async def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """A simple calculator tool that performs basic arithmetic operations.

    Supported operations: add, subtract, multiply, divide.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {
                "error": f"Invalid operation: {operation}. Supported operations are add, subtract, multiply, divide."
            }

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


tools = [calculator]
llm_with_tools = model.bind_tools(tools)


# Define graph State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Build the state graph workflow
def build_graph():

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


# Execute the main application cycle
async def main():
    chatbot = build_graph()

    # Query input structure wrapped in the defined state schema
    inputs = {"messages": [HumanMessage(content="What is 5 + 3?")]}
    result = await chatbot.ainvoke(inputs)

    # Safely print out the final response in the message chain array
    print("\n--- Final Output ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
