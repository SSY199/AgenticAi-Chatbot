import uuid
import streamlit as st

# Updated imports to match the new backend_rag.py
from backend_rag import (
    chatbot, 
    retrieve_all_threads, 
    ingest_pdf, 
    thread_has_document, 
    thread_document_metadata
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ---------- Utility Functions ----------

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

    if thread_id not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][thread_id] = []

def reset_conversation():
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    add_thread(new_thread_id)

def get_thread_title(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    
    # Fetches the first user message from LangGraph checkpoint to use as a title.
    if state and "messages" in state.values and state.values["messages"]:
        for msg in state.values["messages"]:
            if msg.type == "human" and getattr(msg, "content", None):
                content = msg.content
                return content[:25] + "..." if len(content) > 25 else content

    str_tid = str(thread_id)
    return f"✨ New Chat ({str_tid[:8]})" if len(str_tid) > 10 else f"✨ New Chat ({str_tid})"


# ---------- Session State ----------

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

# Ensure current thread structures exist
add_thread(st.session_state["thread_id"])


# ---------- Sidebar: Navigation ----------

st.sidebar.title("Navigation")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_conversation()
    st.rerun()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"]:
    title = get_thread_title(tid)
    label = f"✅ {title}" if tid == st.session_state["thread_id"] else f" {title}"

    if st.sidebar.button(label, key=f"btn_{tid}", use_container_width=True):
        st.session_state["thread_id"] = tid
        add_thread(tid)
        st.rerun()

st.sidebar.caption(f"Current Thread ID:\n{st.session_state['thread_id']}")


# ---------- Sidebar: PDF Upload ----------
st.sidebar.divider()
st.sidebar.header("📄 Document Context")

# 1. Show current document status
if thread_has_document(st.session_state["thread_id"]):
    meta = thread_document_metadata(st.session_state["thread_id"])
    st.sidebar.success(f"**Active PDF:** {meta.get('filename')}\n\n*({meta.get('documents')} pages, {meta.get('chunks')} chunks)*")
else:
    st.sidebar.info("No PDF uploaded for this chat yet.")

# 2. Upload mechanism
uploaded_file = st.sidebar.file_uploader("Upload a PDF for RAG", type=["pdf"])

if uploaded_file:
    current_meta = thread_document_metadata(st.session_state["thread_id"])
    
    # Only ingest if this file hasn't been indexed for this thread yet
    if current_meta.get("filename") != uploaded_file.name:
        with st.sidebar.status("Ingesting PDF...") as status:
            file_bytes = uploaded_file.read()
            summary = ingest_pdf(file_bytes, st.session_state["thread_id"], uploaded_file.name)
            status.update(label=f"Successfully ingested {summary['filename']}!", state="complete", expanded=False)
            st.rerun()


# ---------- Current Conversation ----------

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"],
    },
    "metadata": {
        "thread_id": st.session_state["thread_id"],
    },
    "run_name": "chatbot_turn",
}

current_history = st.session_state["chat_histories"][st.session_state["thread_id"]]

# Populate session state history from LangGraph checkpointer if empty
if not current_history:
    state = chatbot.get_state(CONFIG)
    if state and "messages" in state.values:
        for msg in state.values["messages"]:
            # 1. Identify User Messages
            if msg.type == "human":
                current_history.append({"role": "user", "content": msg.content})
            
            # 2. Identify Final Assistant Messages (Skip tool calls and tool results)
            elif msg.type == "ai":
                if getattr(msg, "tool_calls", None):
                    continue
                if msg.content:
                    current_history.append({"role": "assistant", "content": msg.content})


for message in current_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------- User Input ----------

# ---------- User Input ----------

user_input = st.chat_input("Type your message here...")

if user_input:
    current_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # Store both the UI box and tool name in a dictionary to bypass scope limitations
        status_holder = {
            "box": None,
            "active_tool_name": "tool"
        }

        def response_stream():
            try:
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # Handle Tool Calls safely during streaming chunks
                    has_tool_calls = getattr(message_chunk, "tool_calls", None)
                    is_tool_msg = isinstance(message_chunk, ToolMessage)

                    if has_tool_calls or is_tool_msg:
                        if has_tool_calls:
                            # Safely extract tool name if present in the chunk
                            if message_chunk.tool_calls and isinstance(message_chunk.tool_calls, list) and "name" in message_chunk.tool_calls[0]:
                                status_holder["active_tool_name"] = message_chunk.tool_calls[0]["name"]
                            elif isinstance(message_chunk.tool_calls, dict) and "name" in message_chunk.tool_calls:
                                status_holder["active_tool_name"] = message_chunk.tool_calls["name"]
                                
                            msg = f"🔧 Calling **{status_holder['active_tool_name']}**..."
                        else:
                            tool_name = getattr(message_chunk, "name", status_holder["active_tool_name"])
                            msg = f"🔧 Finished **{tool_name}**"

                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(msg, expanded=True)
                        else:
                            status_holder["box"].update(label=msg, state="running", expanded=True)
                        
                        if is_tool_msg:
                            status_holder["box"].code(message_chunk.content)

                    # Handle AI Text
                    if isinstance(message_chunk, AIMessage) and getattr(message_chunk, "content", None):
                        yield message_chunk.content

            except Exception as e:
                if status_holder["box"] is not None:
                    status_holder["box"].error(f"Error: {str(e)}")
                else:
                    st.error(f"Error: {str(e)}")

        ai_message = st.write_stream(response_stream())
        
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Completed tools",
                expanded=False,
                state="complete",
            )

    current_history.append({"role": "assistant", "content": ai_message})
    st.rerun()

