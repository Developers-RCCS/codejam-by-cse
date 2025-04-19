# Yuhasa - History Tutor Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot focused on answering questions about a Grade 11 History textbook. It uses Google's Gemini AI for language understanding and generation, and FAISS for efficient information retrieval from the textbook content.

**Core Philosophy (Simplified RAG):**

The current version emphasizes a **simple, direct, and transparent** RAG pipeline:

1.  **Chunking:** The PDF (`grade-11-history-text-book.pdf`) is processed using `pdf_chunker.py`. It splits the text into fixed-size word chunks (e.g., 400-500 words) with a small overlap (e.g., 100 words). This uses basic Python string/regex operations, avoiding complex NLP libraries. Each chunk is tagged only with essential metadata: its **page number** and the **first line of text on that page** (as a simple section identifier).
2.  **Indexing:** Text chunks are embedded using a sentence transformer model (via `gemini_utils.py`) and stored in a FAISS index (`faiss_index.index`) along with their metadata (`faiss_metadata.pkl`). This is handled by `faiss_store.py`.
3.  **Retrieval:** When a user asks a question (`web.py` or `cli_chat.py`), the query is embedded. The `RetrieverAgent` (`agents/retriever.py`) performs a direct similarity search against the FAISS index to find the top-k most relevant text chunks (e.g., k=5).
4.  **Context Construction:** The retrieved chunks are concatenated, clearly marked with their page and section metadata (e.g., `--- [Page 12, Section: Early Kingdoms] ---`), and sorted by relevance.
5.  **Generation:** This formatted context, along with the original query and strict instructions, is passed to the Gemini model via the `GeneratorAgent` (`agents/generator.py`). The instructions mandate that Gemini must answer *only* using the provided context and cite the page/section for every fact. If the answer isn't in the context, it must state so directly.

This approach avoids complex intermediate steps like query expansion, re-ranking, or sophisticated section detection, focusing instead on maximizing the quality of the retrieved chunks and ensuring the final answer is directly grounded in and attributed to the source text.

## Prerequisites

*   Python 3.10+
*   pip
*   Google AI API Key
*   Git (Optional)

## Setup

1.  **Clone:** `git clone <url>` & `cd <dir>`
2.  **Environment:** `python -m venv venv`, activate (`.\venv\Scripts\activate` or `source venv/bin/activate`)
3.  **Install:** `pip install -r requirements.txt`
4.  **API Key:** Create `.env` file with `GOOGLE_API_KEY=YOUR_API_KEY_HERE`
5.  **Process PDF:** Run `python faiss_store.py` to create the index files (`faiss_index.index`, `faiss_metadata.pkl`) from `grade-11-history-text-book.pdf`.

## Running the Application

*   **Web UI:** `python web.py` (Access at `http://127.0.0.1:5000`)
*   **CLI:** `python cli_chat.py`

## Project Structure Overview

*   `web.py`: Flask web server.
*   `cli_chat.py`: Command-line interface.
*   `config.py`: Configuration (API keys).
*   `pdf_chunker.py`: **Simple fixed-size word chunking** of the PDF.
*   `faiss_store.py`: Creates and saves the FAISS index and metadata.
*   `gemini_utils.py`: Gemini API interaction helpers (embedding, generation).
*   `agents/retriever.py`: **Directly retrieves top-k chunks** from FAISS.
*   `agents/generator.py`: **Constructs prompt with context/citations** and calls Gemini.
*   `agents/base.py`: Base class for agents (minimal).
*   `templates/index.html`: Web UI template.
*   `static/`: CSS/JS for web UI.
*   `chats/`: Stores chat history JSON files.
*   `grade-11-history-text-book.pdf`: The source document.
*   `faiss_index.index`, `faiss_metadata.pkl`: Vector store files.
*   `requirements.txt`: Dependencies.
*   `README.md`: This file.
*   `PROJECT_EXPLANATION.md`: (Potentially outdated) Detailed explanation.
