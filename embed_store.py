from pdf_chunker import load_and_chunk_pdf
from gemini_utils import embed_text
import chromadb
from chromadb.config import Settings

# Load and chunk the textbook
chunks = load_and_chunk_pdf("grade-11-history-text-book.pdf")

# Init Chroma client (local DB)
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    chroma_db_impl="duckdb+parquet"
))

collection = chroma_client.get_or_create_collection(name="textbook_chunks")

# Add each chunk with embedding
for i, chunk in enumerate(chunks):
    print(f"Embedding chunk {i + 1}/{len(chunks)}")
    embedding = embed_text(chunk["text"])
    collection.add(
        ids=[f"chunk-{i}"],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{"page": chunk["page"]}]
    )

print("✅ All chunks embedded and stored in Chroma!")
