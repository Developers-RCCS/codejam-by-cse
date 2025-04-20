import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import os

def load_and_chunk_pdf(pdf_path):
    documents = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            if text:
                documents.append(Document(page_content=text, metadata={'page_number': page_num + 1}))
        pdf_document.close()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def create_vector_store(documents, embedding_model_name, save_path):
    try:
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode([doc.page_content for doc in documents])

        # Langchain FAISS integration expects an embedding *function*,
        # but SentenceTransformer gives embeddings directly.
        # We need a compatible embedding interface.
        # Let's use a simple wrapper or find a Langchain compatible way.

        # Using Langchain's FAISS.from_documents which handles embedding internally if provided a compatible embedder
        # We need an embedder class that conforms to Langchain's expectations.
        # Let's try using Langchain's SentenceTransformerEmbeddings wrapper if available,
        # otherwise, we might need to adapt. For now, let's assume direct use or a simple embedder.

        # Re-checking Langchain FAISS.from_documents: It takes documents and an *embedding object*.
        # We need to ensure SentenceTransformer integrates smoothly or use Langchain's wrapper.
        # Let's use Langchain's wrapper for Sentence Transformers.
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)

        # Check if save_path directory exists, create if not
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
             os.makedirs(save_dir) # Create directory if it doesn't exist

        vector_store = FAISS.from_documents(documents, embedding_function)
        vector_store.save_local(save_path)
        print(f"Vector store created and saved to {save_path}")
    except Exception as e:
        print(f"Error creating vector store: {e}")

