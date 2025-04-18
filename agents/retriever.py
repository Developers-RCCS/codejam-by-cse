# agents/retriever.py
import faiss
import numpy as np
import pickle
import re
from collections import Counter
import time  # Added for profiling
from .base import BaseAgent
from gemini_utils import embed_text

# --- Configuration ---
DEFAULT_HYBRID_INITIAL_TOP_K = 15  # Tunable parameter for initial FAISS candidate count in hybrid search.

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving and re-ranking relevant text chunks."""
    def __init__(self, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        init_start_time = time.time()
        print("💾 Loading FAISS index and metadata...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.texts = self.metadata["texts"]
        self.metadatas = self.metadata["metadatas"]
        print(f"✅ FAISS index & metadata loaded in {time.time() - init_start_time:.2f}s ({len(self.texts)} chunks).")

    def bm25_tokenize(self, text):
        """Tokenize text for BM25 scoring."""
        return re.findall(r'\b\w+\b', text.lower())

    def simple_keyword_score(self, text_lower, query_keywords_set):
        """Calculate a simple score based on keyword overlap."""
        text_tokens = set(re.findall(r'\b\w+\b', text_lower))
        common_keywords = text_tokens.intersection(query_keywords_set)
        return len(common_keywords) / len(query_keywords_set) if query_keywords_set else 0.0

    def simple_entity_score(self, text_lower, entities):
        """Calculate score based on simple entity presence."""
        score = 0.0
        if not entities:
            return 0.0
        for entity in entities:
            if entity.lower() in text_lower:
                score += 1
        return score / len(entities)

    def section_relevance_score(self, metadata, query_type):
        """Score chunks based on section relevance to query type."""
        section = metadata.get("section", "").lower()
        score = 0.5
        if query_type == "factual" and any(term in section for term in ["overview", "introduction", "summary", "facts", "data"]):
            score = 0.8
        elif query_type == "causal/analytical" and any(term in section for term in ["causes", "effects", "impact", "analysis", "consequences"]):
            score = 0.8
        elif query_type == "comparative" and any(term in section for term in ["comparison", "versus", "differences", "similarities"]):
            score = 0.8
        return score

    def re_rank_chunks(self, initial_results, query, query_analysis):
        """Re-rank chunks based on multiple factors: semantic, keyword, entity, and metadata."""
        rerank_start_time = time.time()
        print("⚖️ Re-ranking retrieved chunks...")
        if not initial_results:
            print("  No initial results to re-rank.")
            return []

        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("query_type", "unknown")
        query_keywords_set = set(keywords)
        print(f"  Extracted Keywords: {keywords}, Entities: {entities}, Type: {query_type}")

        weights = {
            "semantic": 0.5,
            "keyword": 0.3,
            "entity": 0.15,
            "section": 0.05
        }

        max_faiss_dist = max(r["score"] for r in initial_results) if initial_results else 1.0
        if max_faiss_dist <= 0:
            max_faiss_dist = 1.0

        for result in initial_results:
            text_lower = result["text"].lower()
            result["semantic_score"] = max(0.0, 1.0 - (result["score"] / max_faiss_dist))
            result["keyword_score"] = self.simple_keyword_score(text_lower, query_keywords_set)
            result["entity_score"] = self.simple_entity_score(text_lower, entities)
            result["section_score"] = self.section_relevance_score(result["metadata"], query_type)

            combined_score = (
                weights["semantic"] * result["semantic_score"] +
                weights["keyword"] * result["keyword_score"] +
                weights["entity"] * result["entity_score"] +
                weights["section"] * result["section_score"]
            )

            result["combined_score"] = combined_score

            if combined_score > 0.8:
                confidence = 0.9
            elif combined_score > 0.6:
                confidence = 0.7
            elif combined_score > 0.4:
                confidence = 0.5
            elif combined_score > 0.2:
                confidence = 0.3
            else:
                confidence = 0.1
            result["confidence"] = confidence

        ranked_results = sorted(initial_results, key=lambda x: x["combined_score"], reverse=True)
        total_rerank_time = time.time() - rerank_start_time
        print(f"✅ Re-ranking complete. Top score: {ranked_results[0]['combined_score']:.2f} with confidence {ranked_results[0]['confidence']:.2f}" if ranked_results else "✅ Re-ranking complete. No results.")
        return ranked_results

    def run(self, query: str, query_analysis: dict, initial_top_k: int = DEFAULT_HYBRID_INITIAL_TOP_K, final_top_k: int = 5):
        """Retrieves chunks using semantic search, filters and re-ranks them."""
        run_start_time = time.time()
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        print(f"🔎 Running hybrid retrieval for: '{query}' (Initial K={initial_top_k}, Final K={final_top_k})")
        print(f"   Keywords: {keywords}")
        print(f"   Entities: {entities}")

        query_embedding = embed_text(query)
        if query_embedding is None:
            print("  Error: Failed to generate query embedding.")
            return []
        query_embedding_np = np.array([query_embedding]).astype("float32")

        try:
            distances, indices = self.index.search(query_embedding_np, initial_top_k)
        except Exception as e:
            print(f"  Error during FAISS search: {e}")
            return []

        initial_results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.texts):
                initial_results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": distances[0][i]
                })

        print(f"✅ Retrieved {len(initial_results)} valid chunks via semantic search.")
        ranked_results = self.re_rank_chunks(initial_results, query, query_analysis)
        final_results = ranked_results[:final_top_k]
        total_run_time = time.time() - run_start_time
        print(f"--- Returning {len(final_results)} final results ---")
        return final_results
