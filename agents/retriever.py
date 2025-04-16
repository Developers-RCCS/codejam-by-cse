# agents/retriever.py
import faiss
import numpy as np
import pickle
from .base import BaseAgent
from gemini_utils import embed_text

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving relevant text chunks."""
    def __init__(self, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        print("💾 Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.texts = self.metadata["texts"]
        self.metadatas = self.metadata["metadatas"]
        print("✅ FAISS index loaded.")

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieves top_k relevant chunks for the given query."""
        print(f"🔎 Searching for chunks related to: '{query}'")
        query_embedding = embed_text(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding_np, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts): # Ensure index is valid
                 results.append({
                     "text": self.texts[idx],
                     "metadata": self.metadatas[idx],
                     "score": distances[0][i] # Lower distance is better
                 })
            else:
                print(f"⚠️ Warning: Invalid index {idx} returned by FAISS search.")

        print(f"✅ Found {len(results)} relevant chunks.")
        return results
