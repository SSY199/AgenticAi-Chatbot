# 🤖 Agentic AI Chatbot

An intelligent AI chatbot built using **LangChain**, **LangGraph**, and **Streamlit** that supports context-aware conversations with persistent multi-thread memory. The chatbot leverages Google's **Gemini LLM** and integrates **LangSmith** for tracing, debugging, and monitoring.

---

## 🚀 Features

- 💬 AI-powered conversational assistant
- 🧠 Persistent chat memory across sessions
- 🧵 Multi-thread conversation support
- ⚡ Real-time response generation
- 🔄 LangGraph-based workflow orchestration
- 📊 LangSmith integration for tracing and debugging
- 🎨 Clean and interactive Streamlit interface
- 🔌 Modular architecture for future RAG and tool integrations

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **LangGraph**
- **Google Gemini API**
- **LangSmith**
- **SQLite**

---

## 📂 Project Structure

```
AgenticAi-Chatbot/
│── frontend.py          # Streamlit UI
│── chatbot.py           # LangGraph chatbot workflow
│── stream.py            # Streaming responses
│── threading.py         # Thread management
│── database/            # Database utilities
│── tools/               # Tool implementations
│── sequential/          # Sequential workflows
│── parallel/            # Parallel workflows
│── chatbot.db           # SQLite database
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/SSY199/AgenticAi-Chatbot.git
cd AgenticAi-Chatbot
```

### Create a virtual environment

Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AgenticAIChatbot
```

---

## ▶️ Run the Application

```bash
streamlit run frontend.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 🧠 Architecture

```
User
   │
   ▼
Streamlit UI
   │
   ▼
LangGraph Workflow
   │
   ├── Conversation Memory
   ├── Thread Management
   ├── LLM (Gemini)
   └── LangSmith Tracing
   │
   ▼
AI Response
```

---

## 📸 Screenshots

Add screenshots here.

Example:

```
assets/home.png
assets/chat.png
```

---

## 🎯 Future Improvements

- 📄 PDF Chat (RAG)
- 🌐 Web Search Integration
- 🛠 Tool Calling
- 🧩 MCP Server Support
- 🎤 Voice Assistant
- 📁 File Upload Support
- 🔐 User Authentication
- ☁️ Cloud Deployment

---

## 👨‍💻 Author

**Sahil Yadav**

- GitHub: https://github.com/SSY199
- Portfolio: https://sahil199-portfolio.vercel.app

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.