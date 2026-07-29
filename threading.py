import streamlit as st
from chatbot import chatbot
from langchain_core.messages import HumanMessage
import uuid


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


# ---------- Session State ----------

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

add_thread(st.session_state["thread_id"])


# ---------- Sidebar ----------

st.sidebar.title("Conversation History")

if st.sidebar.button("New Chat"):
    reset_conversation()
    st.rerun()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"]:
    if st.sidebar.button(tid, key=tid):
        st.session_state["thread_id"] = tid
        st.rerun()

st.sidebar.text(f"Current Thread:\n{st.session_state['thread_id']}")


# ---------- Current Conversation ----------

current_history = st.session_state["chat_histories"][
    st.session_state["thread_id"]
]

for message in current_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------- User Input ----------

user_input = st.chat_input("Type your message here...")

if user_input:
    current_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )
            if getattr(message_chunk, "content", None)
        )

    current_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )