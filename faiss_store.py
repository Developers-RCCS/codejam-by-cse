from pdf_chunker import load_and_chunk_pdf
from gemini_utils import embed_text
import faiss
import numpy as np
import pickle
import os

# Load chunks
chunks = load_and_chunk_pdf("grade-11-history-text-book.pdf")

# Embed all chunks
texts = []
embeddings = []
metadatas = []

for i, chunk in enumerate(chunks):
    print(f"Embedding chunk {i + 1}/{len(chunks)}")
    emb = embed_text(chunk["text"])
    embeddings.append(emb)
    texts.append(chunk["text"])
    metadatas.append({"page": chunk["page"]})

# Convert to numpy array
embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)

# Save index + metadata
faiss.write_index(index, "faiss_index.index")
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump({"texts": texts, "metadatas": metadatas}, f)

print("✅ FAISS index + metadata saved.")
