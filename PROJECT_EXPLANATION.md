# Project Explanation: Yuhasa History Tutor

## Overview

This project, "Yuhasa," is an intelligent chatbot designed to act as a history tutor. It leverages a technique called Retrieval-Augmented Generation (RAG) to answer user questions based specifically on the content of the provided "Grade 11 History Textbook" PDF. Instead of relying solely on its general knowledge, the chatbot first retrieves relevant passages from the textbook and then uses that information to generate a comprehensive and contextually accurate answer.

## Core Goal

The primary goal is to provide students with a tool to explore and understand their history textbook content interactively. Users can ask questions in natural language, and the chatbot will provide answers grounded in the specific information present in the document.

## Key Processes and Components

1.  **Data Ingestion and Processing (`pdf_chunker.py`, `embed_store.py`):**
    *   **Loading:** The system first reads the `grade-11-history-text-book.pdf`.
    *   **Chunking:** The text content is broken down into smaller, manageable chunks (paragraphs or sections). This is crucial for effective retrieval.
    *   **Embedding:** Each text chunk is converted into a numerical representation called an "embedding" using a machine learning model (likely via `gemini_utils.py` or a dedicated embedding model). Embeddings capture the semantic meaning of the text.
    *   **Vector Store Creation:** These embeddings (vectors) and their corresponding text chunks are stored in a specialized database called a vector store. This project uses FAISS (`faiss_store.py`), which allows for very fast searching of similar vectors. The store consists of `faiss_index.index` (for the vectors) and `faiss_metadata.pkl` (linking vectors back to text and metadata). This step only needs to be done once unless the source PDF changes.

2.  **User Interaction (Web: `app.py`, `templates/index.html`; CLI: `cli_chat.py`):**
    *   Users can interact with the chatbot through either a web-based graphical interface or a simple command-line interface.
    *   The user types a question (e.g., "What were the main causes of World War 1 according to the textbook?").

3.  **Retrieval-Augmented Generation (RAG) Pipeline (`query_answer.py`, `agents/` directory):**
    *   **Query Analysis (`agents/query_analyzer.py` - potentially):** The user's query might be analyzed or rephrased for better retrieval.
    *   **Query Embedding:** The user's question is also converted into an embedding using the same model as the document chunks.
    *   **Retrieval (`agents/retriever.py`, `faiss_store.py`):** The system searches the FAISS vector store for the text chunks whose embeddings are most similar (closest in vector space) to the query embedding. These retrieved chunks are considered the most relevant context from the textbook.
    *   **Context Expansion (`agents/context_expander.py` - potentially):** The retrieved context might be expanded or refined.
    *   **Prompt Formulation:** A prompt is constructed for the generative AI model (Gemini). This prompt typically includes:
        *   The original user question.
        *   The relevant text chunks retrieved from the textbook (the "context").
        *   The recent conversation history (to maintain context in multi-turn conversations).
        *   Instructions for the AI (e.g., "Answer the user's question using *only* the provided context.").
    *   **Generation (`agents/generator.py`, `gemini_utils.py`):** The constructed prompt is sent to the Google Gemini API. Gemini reads the prompt, understands the question and the provided context, and generates a natural language answer.
    *   **Reference Tracking (`agents/reference_tracker.py` - potentially):** The system might track which parts of the retrieved context were used to generate the answer, potentially for citation purposes (though this isn't explicitly shown in the UI).
    *   **Orchestration (`agents/orchestrator.py`):** This component likely manages the flow between the different agents (retriever, generator, etc.) ensuring they work together correctly.

4.  **Response Delivery:**
    *   The generated answer is sent back to the user interface (web or CLI) and displayed.
    *   Conversation history is saved (`chats/` directory) to maintain context for future interactions within the same session.

## Technology Stack

*   **Language:** Python
*   **Web Framework:** Flask (`app.py`)
*   **Generative AI:** Google Gemini (`gemini_utils.py`)
*   **Vector Store:** FAISS (`faiss_store.py`)
*   **PDF Processing:** PyPDF (likely used in `pdf_chunker.py`)
*   **Frontend:** HTML, CSS, JavaScript (`templates/`, `static/`)
*   **Potential Libraries:** LangChain (often used for RAG orchestration), python-dotenv (for environment variables), NumPy.
