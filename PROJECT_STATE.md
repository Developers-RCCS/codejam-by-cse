# Project State: History RAG Chatbot (Future Minds Competition)

**Overall Goal:** Build a high-performing, multi-agent RAG chatbot using Gemini 1.5 Flash to answer questions based on the provided Grade 11 History textbook PDF. The system must be accurate, fast, provide detailed answers with citations, include innovative features (like timelines, web search), and adhere to competition rules. Target high scores in Context Precision, Answer Faithfulness, Correctness, Reference Accuracy, and Innovation.

**Current Status:** Initializing project structure and setting up the document processing pipeline. The textbook PDF (`history_textbook.pdf`) is located in the `data/raw` directory.

**Current Task:**
*   Create basic project folders (`src`, `data`). (Already partially done, ensuring structure is correct)
*   Implement PDF parsing using PyMuPDF.
*   Implement text chunking using Langchain's `RecursiveCharacterTextSplitter`.
*   Generate embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`.
*   Store chunks and embeddings in a local FAISS index saved to the `data` folder.
*   Create initial `requirements.txt` and `README.md`.

**Next Step:** Implement the basic RAG retrieval and generation logic using the created FAISS index.

**Key Technologies:** Python, Langchain, PyMuPDF, SentenceTransformers, FAISS, Gemini 1.5 Flash (for generation later).

**Coding Style:** Write clean, readable Python code. Use descriptive variable and function names. **Do NOT include any comments in the code.** Prioritize simplicity and execution speed.
