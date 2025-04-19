# agents/query_analyzer.py
import spacy
import logging # Added import
from .base import BaseAgent
import re
import time

# Load the spaCy model once when the class is instantiated
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("❌ Error loading spaCy model 'en_core_web_sm'.")
    print("   Please run: python -m spacy download en_core_web_sm")
    nlp = None # Set nlp to None if loading fails

logger = logging.getLogger(__name__) # Get a logger for this module

class QueryAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing the user query."""
    def run(self, query: str, chat_history: list = None) -> dict: # Add chat_history parameter
        start_time = time.time()
        logger.debug(f"Analyzing query: '{query}' with history: {chat_history is not None}") # Log if history is present
        if not nlp:
            logger.warning("spaCy model not loaded, falling back to basic analysis.")
            # Fallback basic extraction (similar to previous web.py logic)
            keywords = re.findall(r'"(.*?)"|\b[A-Z][a-zA-Z]+\b', query)
            entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', query)
            keywords = list(set([k.strip().lower() for k in keywords if k]))
            entities = list(set([e.strip() for e in entities if len(e.split()) > 1 or e in keywords]))
        else:
            # TODO: Incorporate chat_history into spaCy analysis if needed
            # For now, just process the current query
            doc = nlp(query)
            
            # Extract Named Entities (GPE, PERSON, ORG, LOC, EVENT, DATE etc.)
            entities = list(set([ent.text.strip() for ent in doc.ents if ent.label_ in ["GPE", "PERSON", "ORG", "LOC", "EVENT", "DATE", "FAC", "PRODUCT", "WORK_OF_ART"]]))
            
            # Extract Keywords (Noun chunks and Proper Nouns)
            keywords = list(set([chunk.text.lower().strip() for chunk in doc.noun_chunks]))
            # Add proper nouns that might not be part of chunks or recognized entities
            keywords.extend([token.text.lower().strip() for token in doc if token.pos_ == "PROPN" and token.text not in entities])
            # Remove duplicates that might exist between entities and keywords after lowercasing
            keywords = list(set(keywords)) 
            # Optional: Remove very short keywords if needed
            # keywords = [kw for kw in keywords if len(kw) > 2]

        # Determine Query Type (Keep existing logic)
        query_lower = query.lower()
        query_type = "unknown"
        if "cause" in query_lower or "why" in query_lower or "effect" in query_lower or "impact" in query_lower:
            query_type = "causal/analytical"
        elif "compare" in query_lower or "difference" in query_lower or "similar" in query_lower or "contrast" in query_lower:
            query_type = "comparative"
        elif re.match(r"^(what|who|when|where|which)\s+(is|was|are|were|did|do|does)\b", query_lower) or \
             re.match(r"^(define|describe|explain|list)\b", query_lower):
             query_type = "factual"
        # Add more rules if needed

        analysis = {
            "original_query": query, # Add the original query here
            "keywords": keywords,
            "entities": entities,
            "query_type": query_type,
            # Optionally include history info if used
            # "history_considered": chat_history is not None 
        }
        
        end_time = time.time()
        # Log the extracted information
        logger.debug(f"Analysis Results: Keywords: {analysis['keywords']}, Entities: {analysis['entities']}, Query Type: {analysis['query_type']}")
        logger.debug(f"Analysis Time: {end_time - start_time:.4f}s")
        
        return analysis
