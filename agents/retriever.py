# agents/retriever.py
import faiss
import numpy as np
import pickle
import re
import itertools  # For combinations
from collections import Counter
import time  # Added for profiling
from .base import BaseAgent
from gemini_utils import embed_text

# --- Configuration ---
DEFAULT_HYBRID_INITIAL_TOP_K = 25  # Increased initial K for expansion

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
        print(f"  Re-ranking based on -> Keywords: {keywords}, Entities: {entities}, Type: {query_type}")  # Enhanced logging

        # --- Tuned Weights ---
        weights = {
            "semantic": 0.2,
            "keyword": 0.5,
            "entity": 0.3,
            "section": 0.0
        }
        # ---------------------

        # Normalize semantic scores (FAISS distances are lower for better matches)
        max_faiss_dist = max(r["score"] for r in initial_results) if initial_results else 1.0
        if max_faiss_dist <= 0:  # Avoid division by zero
            max_faiss_dist = 1.0

        print(f"  Re-ranking {len(initial_results)} chunks...")
        for i, result in enumerate(initial_results):
            text_lower = result["text"].lower()
            result["semantic_score"] = max(0.0, 1.0 - (max(0.0, result["score"]) / max_faiss_dist))
            result["keyword_score"] = self.simple_keyword_score(text_lower, query_keywords_set)
            result["entity_score"] = self.simple_entity_score(text_lower, entities)
            result["section_score"] = self.section_relevance_score(result["metadata"], query_type)

            combined_score = (
                weights["semantic"] * result["semantic_score"] +
                weights["keyword"] * result["keyword_score"] +
                weights["entity"] * result["entity_score"]
            )
            result["combined_score"] = combined_score

            if combined_score > 0.75:
                confidence = 0.95
            elif combined_score > 0.6:
                confidence = 0.8
            elif combined_score > 0.45:
                confidence = 0.65
            elif combined_score > 0.3:
                confidence = 0.5
            else:
                confidence = 0.3
            result["confidence"] = confidence

        ranked_results = sorted(initial_results, key=lambda x: x["combined_score"], reverse=True)

        print(f"🔍 Filtering {len(ranked_results)} re-ranked chunks for keyword/entity presence...")
        filtered_results = []
        query_terms_lower = {k.lower() for k in keywords} | {e.lower() for e in entities}
        if not query_terms_lower:
             print("  ⚠️ No keywords or entities found in query analysis, skipping filtering.")
             filtered_results = ranked_results
        else:
            for result in ranked_results:
                text_lower = result["text"].lower()
                if any(term in text_lower for term in query_terms_lower):
                    filtered_results.append(result)

        print(f"✅ Filtered down to {len(filtered_results)} chunks containing query keywords/entities.")
        print("  Top 5 Filtered & Re-ranked Chunks (Combined | Sem | Key | Ent | Conf | Page):")
        for i, r in enumerate(filtered_results[:5]):
            page = r.get("metadata", {}).get("page", "?")
            print(f"    {i+1}. Score={r['combined_score']:.3f} (S:{r['semantic_score']:.2f} K:{r['keyword_score']:.2f} E:{r['entity_score']:.2f}) | Conf={r['confidence']:.2f} | Page={page} | Text: {r['text'][:100]}...")

        total_rerank_time = time.time() - rerank_start_time
        top_score_info = f"Top score: {filtered_results[0]['combined_score']:.2f} with confidence {filtered_results[0]['confidence']:.2f}" if filtered_results else "No results after filtering."
        print(f"✅ Re-ranking & Filtering complete in {total_rerank_time:.4f}s. {top_score_info}")
        return filtered_results

    def _simple_expand_query(self, query_analysis: dict, max_expansions: int = 2) -> list[str]:
        """Generates simple query variations based on keywords and entities."""
        expansions = []
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])

        terms = list(set(entities + keywords))
        if not terms:
            return []

        if len(terms) >= 2:
            priority_terms = entities if entities else keywords
            other_terms = keywords if entities else []

            pairs = list(itertools.product(priority_terms, other_terms)) if entities and keywords else []
            if len(pairs) < max_expansions and len(priority_terms) >= 2:
                 pairs.extend(list(itertools.combinations(priority_terms, 2)))

            if len(pairs) < max_expansions:
                 pairs.extend([(t,) for t in terms])

            for pair in pairs:
                 expansions.append(" ".join(pair))
                 if len(expansions) >= max_expansions:
                     break

        if not expansions and terms:
             expansions.extend(terms[:max_expansions])

        unique_expansions = list(dict.fromkeys(expansions))
        print(f"  Generated query expansions: {unique_expansions[:max_expansions]}")
        return unique_expansions[:max_expansions]

    def run(self, query: str, query_analysis: dict, initial_top_k: int = DEFAULT_HYBRID_INITIAL_TOP_K, final_top_k: int = 5):
        """Retrieves chunks using semantic search (with expansion), filters and re-ranks them."""
        run_start_time = time.time()
        print(f"🔎 Running hybrid retrieval for: '{query}' (Initial K={initial_top_k}, Final K={final_top_k})")

        expansion_start_time = time.time()
        expanded_queries = self._simple_expand_query(query_analysis)
        all_queries = [query] + expanded_queries

        query_embeddings = []
        for q in all_queries:
            emb = embed_text(q)
            if emb:
                query_embeddings.append(np.array([emb]).astype("float32"))
            else:
                 print(f"  Warning: Failed to generate embedding for query variant: '{q}'")

        if not query_embeddings:
             print("  Error: Failed to generate any query embeddings.")
             return []
        expansion_time = time.time() - expansion_start_time
        print(f"  Query expansion & embedding took: {expansion_time:.4f}s")

        faiss_start_time = time.time()
        all_distances = []
        all_indices = []
        try:
            for q_emb in query_embeddings:
                 distances, indices = self.index.search(q_emb, initial_top_k)
                 all_distances.append(distances[0])
                 all_indices.append(indices[0])
            print(f"  FAISS search completed for {len(query_embeddings)} query variants.")
        except Exception as e:
            print(f"  Error during FAISS search: {e}")
            return []

        combined_results_map = {}
        for i in range(len(all_indices)):
             indices_list = all_indices[i]
             distances_list = all_distances[i]
             for j, idx in enumerate(indices_list):
                 if idx != -1:
                     distance = distances_list[j]
                     if idx not in combined_results_map or distance < combined_results_map[idx]:
                         combined_results_map[idx] = distance

        sorted_combined_indices = sorted(combined_results_map.keys(), key=lambda idx: combined_results_map[idx])
        faiss_time = time.time() - faiss_start_time
        print(f"  Combined FAISS search & result merging took: {faiss_time:.4f}s")

        initial_results = []
        valid_indices_count = 0
        fetch_limit = int(initial_top_k * 1.2)
        for idx in sorted_combined_indices[:fetch_limit]:
            if 0 <= idx < len(self.texts):
                initial_results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": combined_results_map[idx]
                })
                valid_indices_count += 1
            else:
                print(f"  Warning: Invalid index {idx} encountered after combining searches.")

        initial_results = initial_results[:initial_top_k]

        print(f"✅ Retrieved {len(initial_results)} unique valid chunks via expanded semantic search (target initial_top_k={initial_top_k}).")

        ranked_results = self.re_rank_chunks(initial_results, query, query_analysis)

        final_results = ranked_results[:final_top_k]

        total_run_time = time.time() - run_start_time
        print(f"⏱️ Total Retrieval Time (incl. expansion): {total_run_time:.4f}s")
        print(f"--- Returning {len(final_results)} final results ---")
        return final_results
