from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  google_api_key=os.getenv("GEMINI_API_KEY"),
  temperature=0.7,
)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """A simple calculator tool that performs basic arithmetic operations."""
    
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
            return {"error": f"Invalid operation: {operation}. Supported operations are add, subtract, multiply, divide."}
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
      

tools = [calculator]


llm_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState): 
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
  
  
tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

result = chatbot.invoke({"messages": [HumanMessage(content="What is 5 + 3?")]})

print(result["messages"][0].content)
