# agents/retriever.py
import faiss
import numpy as np
import pickle
import re
from collections import Counter
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
        print(f"✅ FAISS index loaded with {len(self.texts)} chunks.")

    def bm25_tokenize(self, text):
        """Tokenize text for BM25 scoring."""
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\b\w+\b', text.lower())
    
    def calculate_tf(self, text, query_tokens):
        """Calculate term frequency for each query token in the text."""
        text_tokens = self.bm25_tokenize(text)
        token_counts = Counter(text_tokens)
        
        # Calculate term frequency for each query token
        tf_scores = {}
        for token in query_tokens:
            tf = token_counts.get(token, 0) / max(len(text_tokens), 1)
            tf_scores[token] = tf
            
        return tf_scores
    
    def calculate_idf(self, query_tokens):
        """Calculate inverse document frequency for query tokens."""
        # Count documents containing each token
        doc_counts = {}
        total_docs = len(self.texts)
        
        for token in query_tokens:
            count = sum(1 for text in self.texts if token in self.bm25_tokenize(text))
            # Add smoothing to avoid division by zero
            doc_counts[token] = count if count > 0 else 0.5
            
        # Calculate IDF
        idf_scores = {}
        for token, count in doc_counts.items():
            idf_scores[token] = np.log((total_docs - count + 0.5) / (count + 0.5) + 1)
            
        return idf_scores
    
    def keyword_bm25_score(self, text, query, idf_scores):
        """Calculate BM25-inspired score for keyword matching."""
        query_tokens = self.bm25_tokenize(query)
        if not query_tokens:
            return 0.0
            
        # Get term frequencies
        tf_scores = self.calculate_tf(text, query_tokens)
        
        # BM25 parameters
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        
        # Calculate average document length
        avg_doc_len = sum(len(self.bm25_tokenize(doc)) for doc in self.texts) / max(len(self.texts), 1)
        
        # Current document length
        doc_len = len(self.bm25_tokenize(text))
        
        # Calculate BM25 score
        score = 0.0
        for token in query_tokens:
            if token in tf_scores and token in idf_scores:
                numerator = tf_scores[token] * (k1 + 1)
                denominator = tf_scores[token] + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf_scores[token] * (numerator / denominator)
                
        return score

    def entity_match_score(self, text, entities):
        """Calculate score based on entity matches in text."""
        if not entities:
            return 0.0
            
        # Count exact matches of entities in text
        text_lower = text.lower()
        match_count = sum(1 for entity in entities if entity.lower() in text_lower)
        
        # Normalize by number of entities
        normalized_score = match_count / len(entities)
        return normalized_score
    
    def section_relevance_score(self, metadata, query_type):
        """Score chunks based on section relevance to query type."""
        # Default score
        score = 0.5
        
        section = metadata.get("section", "").lower()
        
        # Boost scores for sections likely to contain answers for different query types
        if query_type == "factual" and any(term in section for term in ["overview", "introduction", "summary", "facts", "data"]):
            score = 0.8
        elif query_type == "causal/analytical" and any(term in section for term in ["causes", "effects", "impact", "analysis", "consequences"]):
            score = 0.8
        elif query_type == "comparative" and any(term in section for term in ["comparison", "versus", "differences", "similarities"]):
            score = 0.8
            
        return score

    def re_rank_chunks(self, initial_results, query, query_analysis):
        """Re-rank chunks based on multiple factors: semantic, keyword, entity, and metadata."""
        print("⚖️ Re-ranking retrieved chunks...")
        if not initial_results:
            return []
            
        # Extract data from query analysis
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("query_type", "unknown")
        
        # Pre-calculate IDF scores for efficiency
        query_tokens = self.bm25_tokenize(query)
        idf_scores = self.calculate_idf(query_tokens)

        # Calculate weights for different scores
        weights = {
            "semantic": 0.5,    # Vector similarity (from FAISS)
            "keyword": 0.25,    # Keyword/BM25 score
            "entity": 0.15,     # Named entity matching
            "section": 0.1      # Section relevance
        }
        
        # Normalize semantic scores (lower FAISS distance = better match)
        max_faiss_dist = max(r["score"] for r in initial_results) if initial_results else 1.0
        if max_faiss_dist == 0: max_faiss_dist = 1.0  # Avoid division by zero
        
        for result in initial_results:
            # 1. Normalize semantic score (invert distance)
            result["semantic_score"] = 1.0 - (result["score"] / max_faiss_dist)
            
            # 2. Calculate keyword/BM25 score
            result["keyword_score"] = self.keyword_bm25_score(result["text"], query, idf_scores)
            
            # 3. Calculate entity match score
            result["entity_score"] = self.entity_match_score(result["text"], entities)
            
            # 4. Calculate section relevance score
            result["section_score"] = self.section_relevance_score(result["metadata"], query_type)
            
            # 5. Calculate combined score
            combined_score = (
                weights["semantic"] * result["semantic_score"] +
                weights["keyword"] * result["keyword_score"] +
                weights["entity"] * result["entity_score"] +
                weights["section"] * result["section_score"]
            )
            
            result["combined_score"] = combined_score
            
            # Calculate confidence based on combined score
            # Scale to [0.0, 1.0] range - can adjust thresholds as needed
            if combined_score > 0.8:
                confidence = 0.9  # Very high confidence
            elif combined_score > 0.6:
                confidence = 0.7  # High confidence
            elif combined_score > 0.4:
                confidence = 0.5  # Medium confidence
            elif combined_score > 0.2:
                confidence = 0.3  # Low confidence
            else:
                confidence = 0.1  # Very low confidence
                
            result["confidence"] = confidence
        
        # Sort by combined score (descending)
        ranked_results = sorted(initial_results, key=lambda x: x["combined_score"], reverse=True)
        
        # Log ranking details
        print(f"✅ Re-ranking complete. Top score: {ranked_results[0]['combined_score']:.2f} with confidence {ranked_results[0]['confidence']:.2f}" if ranked_results else "✅ Re-ranking complete. No results.")
        
        return ranked_results

    def run(self, query: str, query_analysis: dict, top_k: int = 10):
        """Retrieves chunks using semantic search, filters and re-ranks them."""
        # Extract data from query analysis
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        
        print(f"🔎 Running hybrid retrieval for: '{query}'")
        print(f"   Keywords: {keywords}")
        print(f"   Entities: {entities}")
        
        # 1. Initial semantic search
        query_embedding = embed_text(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        # 2. Gather initial results
        initial_results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.texts):
                initial_results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": distances[0][i]  # FAISS distance (lower is better)
                })
        
        print(f"✅ Retrieved {len(initial_results)} chunks through semantic search.")
        
        # 3. Re-rank results
        ranked_results = self.re_rank_chunks(initial_results, query, query_analysis)
        
        # 4. Return the re-ranked results (limit to top N)
        final_top_k = min(5, len(ranked_results))
        return ranked_results[:final_top_k]
