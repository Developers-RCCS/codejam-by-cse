from pdf_chunker import load_and_chunk_pdf # Use the updated chunker
from gemini_utils import embed_text
import faiss
import numpy as np
import pickle
import os

# Define PDF path and output paths
PDF_PATH = "grade-11-history-text-book.pdf"
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.pkl"

print(f"🔄 Starting FAISS index build process for {PDF_PATH}...")

# Load chunks with enhanced metadata using the updated pdf_chunker
# Adjust chunk_size and overlap as needed
chunks = load_and_chunk_pdf(PDF_PATH, chunk_size=500, overlap=50)

if not chunks:
    print("❌ No chunks were generated. Exiting.")
    exit()

# Embed all chunks
texts = []
embeddings = []
metadatas = [] # Will store dicts like {"page": X, "section": Y, "paragraphs": Z}

print(f"🧠 Embedding {len(chunks)} chunks...")
for i, chunk in enumerate(chunks):
    # Simple progress indicator
    if (i + 1) % 50 == 0:
        print(f"   Embedding chunk {i + 1}/{len(chunks)}")
    try:
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
        texts.append(chunk["text"])
        # Store the whole metadata dict from the chunker
        metadatas.append(chunk["metadata"])
    except Exception as e:
        print(f"⚠️ Error embedding chunk {i + 1}: {e}. Skipping this chunk.")

if not embeddings:
    print("❌ No embeddings were generated. Exiting.")
    exit()

# Convert to numpy array
embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype("float32")
print(f"🔢 Converted embeddings to NumPy array shape: {embeddings_np.shape}")

# Build FAISS index (using IndexFlatL2 for simplicity)
print(f"🛠️ Building FAISS index (Dimension: {embedding_dim})...")
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)
print(f"✅ FAISS index built. Total vectors: {index.ntotal}")

# Save index + metadata (texts and the enhanced metadatas list)
print(f"💾 Saving FAISS index to {INDEX_PATH}...")
faiss.write_index(index, INDEX_PATH)
print(f"💾 Saving metadata to {METADATA_PATH}...")
with open(METADATA_PATH, "wb") as f:
    pickle.dump({"texts": texts, "metadatas": metadatas}, f)

print("✅ FAISS index build process complete.")
