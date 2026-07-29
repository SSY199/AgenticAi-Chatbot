import uuid
import streamlit as st
from backend import chatbot, retrieve_all_thread
from langchain_core.messages import HumanMessage

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
    label = f" {title}" if tid == st.session_state["thread_id"] else f" {title}"

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
      
      with st.status("Agent is working...", expanded=True) as status:
          def response_stream():
              for message_chunk, metadata in chatbot.stream(
                  {"messages": [HumanMessage(content=user_input)]},
                  config=CONFIG,
                  stream_mode="messages",
              ):

                  # Show tool calls in the status container
                  if getattr(message_chunk, "tool_calls", None):
                      for tool in message_chunk.tool_calls:
                          status.write(f"Calling **{tool['name']}**")

                  # Show tool outputs
                  elif message_chunk.type == "tool":
                      status.write(f"{message_chunk.name}")
                      status.code(message_chunk.content)

                  # Stream only AI text
                  elif getattr(message_chunk, "content", None):
                      yield message_chunk.content

          ai_message = st.write_stream(response_stream())
          
          status.update(
              label="✅ Completed",
              expanded=False,
              state="complete",
          )

    current_history.append({"role": "assistant", "content": ai_message})

    # Trigger a rerun to refresh sidebar titles right after the first message stream ends
    st.rerun()
