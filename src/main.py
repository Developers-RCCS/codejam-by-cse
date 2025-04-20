import os
from pdf_processor import load_and_chunk_pdf, create_vector_store

# Assuming main.py is run from the 'src' directory
PDF_PATH = os.path.join("..", "data", "raw", "history_textbook.pdf")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_SAVE_PATH = os.path.join("..", "data", "faiss_index") # Save inside 'data' folder

def main():
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
    print("Processing complete.")

if __name__ == "__main__":
    main()
