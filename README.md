# Histronaut - History Tutor Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot focused on answering questions about a Grade 11 History textbook. It uses Google's Gemini AI for language understanding and generation, and FAISS for efficient information retrieval from the textbook content. The project intentionally uses minimal dependencies (Flask, FAISS, Gemini, PyPDF) for simplicity, speed, and maintainability.

## Prerequisites

*   **Python:** Version 3.10 or higher recommended.
*   **pip:** Python package installer.
*   **Google AI API Key:** You need an API key from Google AI Studio (or Google Cloud Vertex AI) for the Gemini models.
*   **Git:** (Optional) If cloning the repository.

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file lists the necessary dependencies: `flask`, `google-generativeai`, `faiss-cpu`, `pypdf`, `python-dotenv`, `numpy`)*

3.  **Configure API Key:**
    *   Create a file named `.env` in the root project directory.
    *   Add your Google AI API key to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    *   The `config.py` file likely loads this key.

4.  **Process the PDF (if not already done):**
    The project needs to process the `grade-11-history-text-book.pdf` into a vector store. There might be a script for this, or it might happen automatically on the first run. Check `main.py`, `embed_store.py`, or `pdf_chunker.py` for clues. If a specific script exists (e.g., `python embed_store.py`), run it:
    ```bash
    python faiss_store.py
    ```
    This will create the `faiss_index.index` and `faiss_metadata.pkl` files (or similar).

## Running the Application

There seem to be multiple ways to interact with the chatbot:

1.  **Web Interface (Recommended):**
    *   Run the Flask web server:
        ```bash
        python web.py
        ```
    *   Open your web browser and navigate to the address provided (usually `http://127.0.0.1:5000` or similar).

2.  **Command-Line Interface:**
    *   Run the CLI script:
        ```bash
        python cli_chat.py
        ```
    *   Interact with the bot directly in your terminal. Type 'exit' or 'quit' to end the session.

## Project Overview
[Read the Project Overview](https://github.com/Developers-RCCS/codejam-by-cse/blob/main/PROJECT_EXPLANATION.md)
