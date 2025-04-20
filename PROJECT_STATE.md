# Project State: History RAG Chatbot (Future Minds Competition)

**Overall Goal:** Build a high-performing, multi-agent RAG chatbot using Gemini 1.5 Flash to answer questions based on the provided Grade 11 History textbook PDF. The system must be accurate, fast, provide detailed answers with citations, include innovative features (like timelines, web search), and adhere to competition rules. Target high scores in Context Precision, Answer Faithfulness, Correctness, Reference Accuracy, and Innovation.

**Current Status:**
*   Project structure created (`src`, `data`).
*   PDF processed, FAISS vector store created and saved (`data/faiss_index`).
*   Basic RAG pipeline implemented (`src/rag_core.py`) using FAISS, SentenceTransformers, Langchain (LCEL), and Gemini 1.5 Flash.
*   API key handling via `.env` implemented.
*   `main.py` successfully runs a sample query through the RAG pipeline.
*   `requirements.txt` and `README.md` are up-to-date.

**Current Task:**
*   Refactor the RAG logic in `src/rag_core.py` into a class (e.g., `RAGSystem`).
*   Modify the response function to accept a query ID and return a dictionary matching the competition submission structure (`ID`, `Context` (string), `Answer`, `Sections` (placeholder), `Pages`).
*   Add basic `try...except` blocks for file loading (FAISS index) and the Gemini API call.
*   Update `main.py` to load the `queries.json` file (assume it's placed in `data/queries.json`).
*   Process the *first* query from `queries.json` using the `RAGSystem` class.
*   Print the structured output for the processed query.
*   Update `README.md` regarding `queries.json`.

**Next Step:** Implement processing for all queries in `queries.json`, generate the CSV submission file, and introduce metadata extraction (like sections) during PDF processing.

**Key Technologies:** Python, Langchain, PyMuPDF, SentenceTransformers, FAISS, Google Generative AI (Gemini), os, json, python-dotenv.

**Competition Format Note:** The target output format is ID, Context (retrieved text), Answer (generated), Sections (list/string), Pages (list/string). We will use an empty list or placeholder string for 'Sections' for now.

**Coding Style:** Write clean, readable Python code. Use descriptive variable and function names. **Do NOT include any comments in the code.** Prioritize simplicity and execution speed. Handle errors gracefully where possible.
