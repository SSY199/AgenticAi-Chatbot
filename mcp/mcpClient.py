import uuid
import queue
import streamlit as st
from mcpServer import chatbot, retrieve_all_thread, submit_async_task
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
    """Fetches the first user message from LangGraph checkpoint to use as a title."""

    if state and "messages" in state.values and state.values["messages"]:
        # Find the first message sent by a human user
        for msg in state.values["messages"]:
            if msg.type == "human" and getattr(msg, "content", None):
                content = msg.content
                # Truncate to keep the sidebar visually clean
                return content[:25] + "..." if len(content) > 25 else content

    # Fallback title if there are no messages in the checkpoint database yet
    str_tid = str(thread_id)
    return f"✨ New Chat ({str_tid[:8]})" if len(str_tid) > 10 else f"✨ New Chat ({str_tid})"


# ---------- Session State ----------

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_thread()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

# Ensure current thread structures exist
add_thread(st.session_state["thread_id"])


# ---------- Sidebar ----------

st.sidebar.title("Conversation History")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_conversation()
    st.rerun()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"]:
    # 1. Fetch conversational title or human-readable fallback dynamically
    title = get_thread_title(tid)

    # 2. Add visual indicator for the active thread
    label = f"✅ {title}" if tid == st.session_state["thread_id"] else f" {title}"

    if st.sidebar.button(label, key=f"btn_{tid}", use_container_width=True):
        st.session_state["thread_id"] = tid
        # Safeguard: Initialize state structures if switching to an older DB thread
        add_thread(tid)
        st.rerun()

st.sidebar.text(f"Current Thread ID:\n{st.session_state['thread_id']}")


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
            role = "user" if msg.type == "human" else "assistant"
            current_history.append({"role": role, "content": msg.content})

for message in current_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------- User Input ----------

user_input = st.chat_input("Type your message here...")

if user_input:
    current_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # Use a dictionary to hold the st.status container so it can be updated inside the generator
        status_holder = {"box": None}

        def response_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            # Fire off the async task in your background loop
            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Show tool calls lazily (only if tools are actually triggered)
                if getattr(message_chunk, "tool_calls", None) or isinstance(message_chunk, ToolMessage):
                    
                    # Extract tool name based on the chunk type
                    if getattr(message_chunk, "tool_calls", None):
                        tool_name = message_chunk.tool_calls[0]["name"]
                        msg = f"🔧 Calling **{tool_name}**..."
                    else:
                        tool_name = getattr(message_chunk, "name", "tool")
                        msg = f"🔧 Finished **{tool_name}**"

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(msg, expanded=True)
                    else:
                        status_holder["box"].update(
                            label=msg,
                            state="running",
                            expanded=True,
                        )
                        # Optional: output raw tool responses in the expander
                        if isinstance(message_chunk, ToolMessage):
                            status_holder["box"].code(message_chunk.content)

                # Stream only AI text
                if isinstance(message_chunk, AIMessage) and getattr(message_chunk, "content", None):
                    yield message_chunk.content

        # Stream the chunks yielded from the queue
        ai_message = st.write_stream(response_stream())
        
        # Once complete, collapse and mark the status box as finished if it was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Completed",
                expanded=False,
                state="complete",
            )

    # Save to history and rerun
    current_history.append({"role": "assistant", "content": ai_message})
    st.rerun()