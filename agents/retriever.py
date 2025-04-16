# agents/retriever.py
import faiss
import numpy as np
import pickle
from .base import BaseAgent
from gemini_utils import embed_text

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving and re-ranking relevant text chunks."""
    def __init__(self, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        print("💾 Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.texts = self.metadata["texts"]
        self.metadatas = self.metadata["metadatas"]
        print("✅ FAISS index loaded.")

    def calculate_keyword_score(self, text: str, keywords: list[str]) -> float:
        """Calculates a simple keyword overlap score."""
        score = 0
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        # Normalize score (optional, simple count for now)
        return float(score)

    def re_rank_chunks(self, initial_results: list[dict], keywords: list[str]) -> list[dict]:
        """Re-ranks chunks based on semantic distance and keyword overlap."""
        print("⚖️ Re-ranking retrieved chunks...")
        if not initial_results:
            return []

        max_faiss_dist = max(r["score"] for r in initial_results) if initial_results else 1.0
        if max_faiss_dist == 0: max_faiss_dist = 1.0 # Avoid division by zero

        for result in initial_results:
            # 1. Calculate Keyword Score
            keyword_score = self.calculate_keyword_score(result["text"], keywords)

            # 2. Normalize FAISS distance (lower is better -> higher score is better)
            # Simple inversion and normalization
            semantic_score = 1.0 - (result["score"] / max_faiss_dist)

            # 3. Combine Scores (Simple weighted average, adjust weights as needed)
            # Example: 70% semantic, 30% keyword
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score

            result["keyword_score"] = keyword_score
            result["semantic_score"] = semantic_score
            result["combined_score"] = combined_score
            # Basic confidence: use the combined score for now
            result["confidence"] = combined_score

        # Sort by combined_score descending (higher is better)
        ranked_results = sorted(initial_results, key=lambda x: x["combined_score"], reverse=True)
        print(f"✅ Re-ranking complete. Top score: {ranked_results[0]['combined_score']:.2f}" if ranked_results else "✅ Re-ranking complete. No results.")
        return ranked_results

    def run(self, query: str, keywords: list[str], top_k: int = 10) -> list[dict]: # Increase initial retrieval
        """Retrieves top_k relevant chunks, re-ranks them, and adds confidence."""
        print(f"🔎 Retrieving top {top_k} chunks for: '{query}'")
        query_embedding = embed_text(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding_np, top_k)

        initial_results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.texts):
                initial_results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": distances[0][i] # Raw FAISS distance (lower is better)
                })
            else:
                print(f"⚠️ Warning: Invalid index {idx} returned by FAISS search.")

        print(f"✅ Found {len(initial_results)} initial chunks.")

        # Re-rank the initial results
        ranked_results = self.re_rank_chunks(initial_results, keywords)

        # Return only top N results after re-ranking if desired, e.g., top 5
        final_top_k = 5
        return ranked_results[:final_top_k]
