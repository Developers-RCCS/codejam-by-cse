import logging
import time
import faiss
import pickle
import numpy as np
import itertools
import re
import nltk
from nltk.corpus import stopwords
from .base import BaseAgent
from gemini_utils import embed_text
from utils.text_utils import simple_keyword_score, simple_entity_score, section_relevance_score
from config import Config

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_INITIAL_TOP_K = Config.RETRIEVER_INITIAL_K
DEFAULT_HYBRID_FINAL_TOP_K = Config.RETRIEVER_FINAL_K

# --- Download NLTK data if not present (optional, can be done offline) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK averaged_perceptron_tagger...")
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)
# --- End NLTK Download ---

# --- Constants for Query Analysis ---
HISTORY_KEYWORDS = {
    "history", "historical", "past", "event", "events", "timeline", "era", "period",
    "king", "queen", "kingdom", "empire", "dynasty", "ruler", "ancient", "medieval",
    "colonial", "independence", "war", "battle", "treaty", "culture", "society",
    "archaeology", "source", "primary", "secondary", "vijaya", "kandy", "anuradhapura",
    "polonnaruwa", "sigiriya", "portuguese", "dutch", "british", "ceylon", "lanka"
    # Add more specific terms relevant to the textbook
}
STOP_WORDS = set(stopwords.words('english'))
# --- End Constants ---

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving and re-ranking relevant text chunks."""
    def __init__(self, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        logger.info(f"💾 Loading FAISS index from: {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"✅ FAISS index loaded successfully. Index dimension: {self.index.d}, Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}", exc_info=True)
            raise
        logger.info(f"💾 Loading metadata from: {metadata_path}")
        try:
            with open(metadata_path, "rb") as f:
                self.metadatas = pickle.load(f)
            # Pre-extract texts for faster access if needed elsewhere
            self.texts = [m.pop('text', '') for m in self.metadatas] # Extract text and remove from metadata dict
            logger.info(f"✅ Metadata loaded successfully. Number of entries: {len(self.metadatas)}")
            if len(self.metadatas) != self.index.ntotal:
                 logger.warning(f"⚠️ Mismatch between index size ({self.index.ntotal}) and metadata count ({len(self.metadatas)}).")
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}", exc_info=True)
            raise

    def re_rank_chunks(self, initial_results, query, query_analysis):
        """Re-rank chunks based on multiple factors using utility functions."""
        rerank_start_time = time.time()
        logger.info("⚖️ Re-ranking retrieved chunks...")
        if not initial_results:
            logger.warning("No initial results to re-rank.")
            return []

        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("query_type", "unknown")
        intent_type = query_analysis.get("intent_type", "new_topic") # Get intent
        topic_keywords = query_analysis.get("topic_keywords", []) # Get topic keywords
        topic_entities = query_analysis.get("topic_entities", []) # Get topic entities

        query_keywords_set = set(keywords)
        topic_terms_set = set(topic_keywords + topic_entities) # Combine topic terms

        logger.debug(f"Re-ranking based on -> Query Keywords: {keywords}, Entities: {entities}, Type: {query_type}, Intent: {intent_type}, Topic Terms: {topic_terms_set}")

        # --- Tuned Weights ---
        # Adjust weights based on intent? (Example)
        if intent_type in ["follow_up", "clarification"] and topic_terms_set:
             logger.debug("Adjusting weights for follow-up/clarification intent.")
             weights = {
                 "semantic": 0.15, # Slightly lower semantic weight for current query
                 "keyword": 0.4, # Keep keyword weight
                 "entity": 0.25, # Keep entity weight
                 "topic": 0.2, # Add weight for topic relevance
                 "section": 0.0
             }
        else:
             weights = {
                 "semantic": 0.2,
                 "keyword": 0.5,
                 "entity": 0.3,
                 "topic": 0.0, # No topic weight for new topics
                 "section": 0.0
             }
        # ---------------------

        # Normalize semantic scores (FAISS distances are lower for better matches)
        max_faiss_dist = max(r["score"] for r in initial_results) if initial_results else 1.0
        if max_faiss_dist <= 0:  # Avoid division by zero
            max_faiss_dist = 1.0

        logger.debug(f"Re-ranking {len(initial_results)} chunks...")
        for i, result in enumerate(initial_results):
            text_lower = self.texts[result["index"]].lower() # Get text using index
            result["text"] = self.texts[result["index"]] # Add full text back for generator
            result["metadata"] = self.metadatas[result["index"]] # Add metadata back

            result["semantic_score"] = max(0.0, 1.0 - (max(0.0, result["score"]) / max_faiss_dist))
            # Use utility functions for scoring
            result["keyword_score"] = simple_keyword_score(text_lower, query_keywords_set)
            result["entity_score"] = simple_entity_score(text_lower, entities)
            result["section_score"] = section_relevance_score(result["metadata"], query_type)
            # Add topic score if applicable
            result["topic_score"] = simple_keyword_score(text_lower, topic_terms_set) if weights["topic"] > 0 else 0.0

            combined_score = (
                weights["semantic"] * result["semantic_score"] +
                weights["keyword"] * result["keyword_score"] +
                weights["entity"] * result["entity_score"] +
                weights["topic"] * result["topic_score"] # Include topic score
                # + weights["section"] * result["section_score"] # Section score currently unused
            )
            result["combined_score"] = combined_score

            # Confidence calculation (can be refined)
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

        # Sort by combined score
        ranked_results = sorted(initial_results, key=lambda x: x["combined_score"], reverse=True)

        # Filter based on presence of *query* keywords/entities (important!)
        logger.info(f"🔍 Filtering {len(ranked_results)} re-ranked chunks for *query* keyword/entity presence...")
        filtered_results = []
        query_terms_lower = {k.lower() for k in keywords} | {e.lower() for e in entities}

        # If the query itself has no terms, but it's a follow-up, rely on topic terms for filtering?
        # Or maybe skip filtering if query terms are absent? Let's skip for now.
        if not query_terms_lower and intent_type not in ["follow_up", "clarification"]:
             logger.warning("⚠️ No keywords or entities found in query analysis, and not a follow-up. Skipping filtering.")
             filtered_results = ranked_results
        elif not query_terms_lower and intent_type in ["follow_up", "clarification"]:
             logger.warning("⚠️ No keywords or entities in query, but it's a follow-up/clarification. Filtering based on *topic* terms.")
             filter_terms = {t.lower() for t in topic_terms_set} # Use topic terms for filtering
             if not filter_terms:
                 logger.warning("⚠️ No topic terms found either. Skipping filtering.")
                 filtered_results = ranked_results
             else:
                 for result in ranked_results:
                     text_lower = result["text"].lower()
                     # Check for topic terms instead of query terms
                     if any(re.search(r'\b' + re.escape(term) + r'\b', text_lower) for term in filter_terms):
                         filtered_results.append(result)
        else:
             # Standard filtering based on query terms
             filter_terms = query_terms_lower
             for result in ranked_results:
                 text_lower = result["text"].lower()
                 if any(re.search(r'\b' + re.escape(term) + r'\b', text_lower) for term in filter_terms):
                     filtered_results.append(result)


        logger.info(f"✅ Filtered down to {len(filtered_results)} chunks containing relevant terms.")
        logger.debug("Top 5 Filtered & Re-ranked Chunks (Combined | Sem | Key | Ent | Top | Conf | Page):")
        for i, r in enumerate(filtered_results[:5]):
            page = r.get("metadata", {}).get("page", "?")
            logger.debug(f"{i+1}. Score={r['combined_score']:.3f} (S:{r['semantic_score']:.2f} K:{r['keyword_score']:.2f} E:{r['entity_score']:.2f} T:{r['topic_score']:.2f}) | Conf={r['confidence']:.2f} | Page={page} | Text: {r['text'][:100]}...")

        total_rerank_time = time.time() - rerank_start_time
        logger.info(f"Step 2b: Re-ranking & Filtering took: {total_rerank_time:.4f}s")
        return filtered_results


    def _simple_expand_query(self, query_analysis: dict, max_expansions: int = 2) -> list[str]:
        """Generates simple query variations based on keywords and entities."""
        expansions = []
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        # Consider adding topic terms if it's a follow-up with few query terms?
        intent_type = query_analysis.get("intent_type", "new_topic")
        topic_keywords = query_analysis.get("topic_keywords", [])
        topic_entities = query_analysis.get("topic_entities", [])

        terms = list(set(entities + keywords))

        # If few terms in query but it's a follow-up, add topic terms to expansion base
        if len(terms) < 2 and intent_type in ["follow_up", "clarification"]:
            logger.debug("Expanding query using topic terms for follow-up.")
            terms.extend(topic_keywords)
            terms.extend(topic_entities)
            terms = list(set(terms)) # Ensure uniqueness

        if not terms:
            return []

        # Prioritize entities for combinations
        priority_terms = entities if entities else keywords
        other_terms = keywords if entities else []

        # Generate pairs (priority x other, priority x priority)
        pairs = []
        if priority_terms and other_terms:
             pairs.extend(list(itertools.product(priority_terms, other_terms)))
        if len(priority_terms) >= 2:
             pairs.extend(list(itertools.combinations(priority_terms, 2)))

        # Add single terms if not enough pairs
        if len(pairs) < max_expansions:
             pairs.extend([(t,) for t in terms]) # Add single terms

        # Create expansion strings
        for pair in pairs:
             expansions.append(" ".join(pair))
             if len(expansions) >= max_expansions:
                 break

        # Fallback: if still no expansions, use top terms directly
        if not expansions and terms:
             expansions.extend(terms[:max_expansions])

        unique_expansions = list(dict.fromkeys(expansions)) # Maintain order while making unique
        logger.debug(f"Generated query expansions: {unique_expansions[:max_expansions]}")
        return unique_expansions[:max_expansions]


    def run(self, query: str, query_analysis: dict, initial_top_k: int = DEFAULT_HYBRID_INITIAL_TOP_K, final_top_k: int = 5):
        """Retrieves chunks using semantic search (with expansion), filters and re-ranks them."""
        run_start_time = time.time()
        logger.info(f"🔎 Running hybrid retrieval for: '{query}' (Initial K={initial_top_k}, Final K={final_top_k})")
        logger.debug(f"Query Analysis for Retrieval: {query_analysis}") # Log full analysis

        expansion_start_time = time.time()
        # Use original query if analysis didn't refine, otherwise use refined
        query_to_expand = query_analysis.get("original_query", query) # Use original for expansion base
        expanded_queries = self._simple_expand_query(query_analysis)
        all_queries = [query_to_expand] + expanded_queries # Include original query

        query_embeddings = []
        for q in all_queries:
            emb = embed_text(q)
            if emb:
                query_embeddings.append(np.array([emb]).astype("float32"))
            else:
                 logger.warning(f"Failed to generate embedding for query variant: '{q}'")

        if not query_embeddings:
             logger.error("Failed to generate any query embeddings.")
             return []

        expansion_time = time.time() - expansion_start_time
        logger.info(f"Step 2a: Query expansion & embedding took: {expansion_time:.4f}s") # Corrected this line

        faiss_start_time = time.time()
        # ...existing code...