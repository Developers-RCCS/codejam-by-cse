import os
from dotenv import load_dotenv
from pdf_processor import load_and_chunk_pdf, create_vector_store
from rag_core import load_retriever, setup_rag_chain, get_rag_response

load_dotenv() 

PDF_PATH = os.path.join("..", "data", "raw", "history_textbook.pdf")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_SAVE_PATH = os.path.join("..", "data", "faiss_index")

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file in the project root and add your key:")
        print("GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
        return

    if not os.path.exists(FAISS_SAVE_PATH):
        print("FAISS index not found. Running PDF processing first...")
        print("Starting PDF processing...")
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF file not found at {PDF_PATH}")
            return

        print(f"Loading and chunking PDF from: {PDF_PATH}")
        documents = load_and_chunk_pdf(PDF_PATH)

        if not documents:
            print("No documents were processed from the PDF.")
            return

        print(f"Successfully loaded and chunked PDF into {len(documents)} documents.")
        print("Creating vector store...")
        create_vector_store(documents, EMBEDDING_MODEL_NAME, FAISS_SAVE_PATH)
        print("PDF Processing complete.")
    else:
        print(f"Found existing FAISS index at {FAISS_SAVE_PATH}")

    print("Loading retriever...")
    retriever = load_retriever(EMBEDDING_MODEL_NAME, FAISS_SAVE_PATH)
    print("Setting up RAG chain...")
    rag_chain = setup_rag_chain(retriever, api_key)

    sample_query = "What were the key developments in the coal industry during the Industrial Revolution?"
    print(f"\n--- Running Sample Query ---")
    print(f"Query: {sample_query}")

    result = get_rag_response(sample_query, rag_chain, retriever)

    print("\nRetrieved Context Snippets (First 500 chars):")
    context_str = "\n---\n".join([doc.page_content for doc in result['context']])
    print(f"{context_str[:500]}...")

    print(f"\nSource Page Numbers: {result['pages']}")
    print(f"\nGenerated Answer:")
    print(result['answer'])
    print("--------------------------")

if __name__ == "__main__":
    main()
