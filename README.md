# AI-Powered RAG Chatbot

An intelligent conversational AI chatbot built with **LangGraph, LangChain, Google Gemini, Streamlit, FAISS, and SQLite**.

The chatbot supports **persistent conversations, PDF-based Retrieval-Augmented Generation (RAG), web search, stock price lookup, calculator tools, and real-time streaming responses**.

## Features

### 💬 Persistent Conversations

* Create and switch between multiple chat threads
* Persistent conversation history using **SQLite**
* LangGraph checkpointing for state management
* Automatically restores previous conversations
* Thread titles generated from the first user message

### 📄 PDF-Based RAG

Upload a PDF to provide document-specific context to the chatbot.

The chatbot:

* Extracts text using `PyPDFLoader`
* Splits documents into chunks using `RecursiveCharacterTextSplitter`
* Generates embeddings using HuggingFace embeddings
* Stores vector embeddings using **FAISS**
* Retrieves relevant document chunks during conversations
* Maintains document context separately for each chat thread

### 🔎 Web Search

The chatbot can search the web when additional or up-to-date information is required using the **Tavily Search API**.

### 📈 Stock Price Lookup

Fetch stock information using the Alpha Vantage API.

Example:

```text
What is the current stock price of AAPL?
```

### 🧮 Calculator Tool

Perform basic arithmetic operations:

* Addition
* Subtraction
* Multiplication
* Division

Example:

```text
Calculate 245 * 67
```

### 🔧 Tool Calling

The chatbot uses **LangGraph tool calling** to decide when external tools are required.

Available tools:

```text
User Query
    ↓
Gemini LLM
    ↓
Does the query require a tool?
    ↓
┌──────────┬───────────┬─────────────┬────────────┐
│   RAG    │ Web Search│ Stock Price │ Calculator │
└──────────┴───────────┴─────────────┴────────────┘
    ↓
Tool Result
    ↓
Gemini LLM
    ↓
Final Response
```

The UI also displays the current tool execution status.

### ⚡ Streaming Responses

AI responses are streamed in real time using Streamlit's `st.write_stream()`.

This provides a more interactive experience instead of waiting for the complete response.

### 🗂️ Per-Thread Document Context

Each conversation has its own document context.

```text
Thread A
 ├── Conversation History
 └── PDF / FAISS Vector Store

Thread B
 ├── Conversation History
 └── PDF / FAISS Vector Store
```

This prevents documents from one conversation from being used in another conversation.

### 🛡️ Basic Safety Limits

The application includes limits to help prevent excessive resource usage:

* Maximum PDF pages per file: `100`
* Maximum chunks per thread: `5000`
* Document/vector-store TTL: `24 hours`

Expired vector stores and document metadata are automatically cleaned up.

## Tech Stack

| Technology                  | Usage                                 |
| --------------------------- | ------------------------------------- |
| **Python**                  | Core programming language             |
| **Streamlit**               | Chatbot frontend                      |
| **LangChain**               | LLM and tool integration              |
| **LangGraph**               | Agent workflow and conversation state |
| **Google Gemini 2.5 Flash** | Large Language Model                  |
| **HuggingFace**             | Text embeddings                       |
| **FAISS**                   | Vector similarity search              |
| **SQLite**                  | Checkpoint and metadata storage       |
| **PyPDFLoader**             | PDF document loading                  |
| **Tavily**                  | Web search                            |
| **Alpha Vantage**           | Stock data                            |

## Architecture

```text
                         ┌─────────────────┐
                         │  Streamlit UI   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    LangGraph    │
                         │     Agent       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Gemini 2.5 Flash│
                         └────────┬────────┘
                                  │
                     Tool Required?
                   ╱      │      │      ╲
                  ▼       ▼      ▼       ▼
                RAG      Web    Stock  Calculator
                 │        │       │       │
                 ▼        ▼       ▼       ▼
              FAISS    Tavily  Alpha     Python
              Search           Vantage
                 │
                 ▼
            PDF Context
```

## RAG Pipeline

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Extract Pages
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
FAISS Vector Store
    ↓
Persisted Per Thread
    ↓
User Question
    ↓
Similarity Search
    ↓
Top Relevant Chunks
    ↓
Gemini LLM
    ↓
Final Answer
```

## Project Structure

```text
.
├── app.py
├── backend_rag.py
├── chatbot.db
├── faiss_stores/
├── .env
├── requirements.txt
└── README.md
```

### `app.py`

Contains the Streamlit frontend.

Responsibilities:

* Chat UI
* Conversation navigation
* Thread management
* PDF upload
* Chat history rendering
* Streaming responses
* Tool execution status

### `backend_rag.py`

Contains the chatbot backend.

Responsibilities:

* Gemini LLM initialization
* HuggingFace embeddings
* FAISS vector store
* PDF ingestion
* RAG retrieval
* LangGraph workflow
* Tool definitions
* SQLite persistence
* Thread management

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```text
.env
venv/
__pycache__/
chatbot.db
faiss_stores/
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available locally in your browser.

## How It Works

### Normal Conversation

```text
User
  ↓
LangGraph
  ↓
Gemini
  ↓
Response
```

### PDF Question

```text
User: "Summarize chapter 2"
              ↓
        Gemini detects
        document query
              ↓
           RAG Tool
              ↓
       FAISS Similarity Search
              ↓
     Relevant PDF Chunks
              ↓
           Gemini
              ↓
        Final Response
```

### Tool-Based Question

```text
User: "What is 245 × 67?"
              ↓
           Gemini
              ↓
       Calculator Tool
              ↓
          16415
              ↓
       Gemini Response
```

## Example Queries

### General Chat

```text
Explain what Retrieval-Augmented Generation is.
```

### PDF RAG

```text
Summarize the uploaded document.
```

```text
What are the main concepts discussed in chapter 3?
```

```text
Explain the uploaded PDF in simple terms.
```

### Web Search

```text
What are the latest developments in generative AI?
```

### Stock Tool

```text
Get the stock price of TSLA.
```

### Calculator

```text
Calculate (245 * 67) / 5.
```

## LangGraph Workflow

The chatbot follows this workflow:

```text
START
  │
  ▼
chat_node
  │
  ├── No Tool Call ──────────────► END
  │
  └── Tool Call
         │
         ▼
      ToolNode
         │
         ▼
      chat_node
         │
         ▼
        END
```

The `chat_node` sends the conversation to Gemini.

If Gemini requests a tool, LangGraph routes execution to the `ToolNode`. After receiving the tool result, the response is sent back to the LLM to generate the final answer.

## Current Limitations

The project currently uses:

* Vector similarity search instead of hybrid search
* Basic document metadata management
* No user authentication
* No semantic long-term memory
* No RAG page/source citations in final responses
* No document deletion interface
* No conversation deletion or renaming
* No automated evaluation pipeline

## Future Improvements

### Retrieval Improvements

* [ ] Hybrid search using BM25 + vector search
* [ ] Reranking retrieved documents
* [ ] Page-level source citations
* [ ] Multi-document retrieval
* [ ] Document selection and deletion

### Memory

* [ ] Long-term user memory
* [ ] Automatic memory extraction
* [ ] Conversation summarization
* [ ] Semantic memory retrieval

### Chat Features

* [ ] Rename conversations
* [ ] Delete conversations
* [ ] Search conversations
* [ ] Pin important conversations
* [ ] AI-generated conversation titles

### AI Engineering

* [ ] Query routing node
* [ ] Tool usage evaluation
* [ ] RAG evaluation
* [ ] Answer faithfulness evaluation
* [ ] Latency monitoring
* [ ] Token and cost tracking
* [ ] User feedback system

### Security

* [ ] User authentication
* [ ] Secure API key management
* [ ] Rate limiting
* [ ] Improved prompt injection protection
* [ ] Tool permission controls

## Future Architecture

```text
                         ┌──────────────┐
                         │ Streamlit UI │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │    Auth      │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  LangGraph   │
                         │ Query Router │
                         └──────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              Chat Node       RAG           Tools
                               │         ╱    │    ╲
                               ▼       Web  Stock  Calc
                         Hybrid Search
                               │
                         ┌─────┴─────┐
                         ▼           ▼
                       BM25       FAISS
                         │           │
                         └─────┬─────┘
                               ▼
                            Reranker
                               │
                               ▼
                          Source Citations
                               │
                               ▼
                          Final Response

                    ┌──────────┴──────────┐
                    ▼                     ▼
             Short-Term Memory      Long-Term Memory
             SQLite Checkpoints     User Memory Store
```

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/your-feature-name
```

3. Make your changes
4. Commit your changes

```bash
git commit -m "Add your feature"
```

5. Push to your branch

```bash
git push origin feature/your-feature-name
```

6. Open a Pull Request

## License

This project is currently intended for educational and portfolio purposes.

---

## Author

**Sahil Yadav**

Engineering Student | AI/ML Enthusiast | Software Developer

⭐ If you found this project useful, consider giving it a star!
