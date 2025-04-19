# agents/query_analyzer.py
import logging
from .base import BaseAgent
import re
import time

logger = logging.getLogger(__name__) # Get a logger for this module

class QueryAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing the user query using simple regex methods."""
    def run(self, query: str, chat_history: list = None) -> dict: # Add chat_history parameter
        start_time = time.time()
        logger.debug(f"Analyzing query: '{query}' with history: {chat_history is not None}") # Log if history is present

        # Use basic regex extraction (similar to previous fallback logic)
        # Extract potential keywords (quoted phrases or capitalized words)
        keywords = re.findall(r'"(.*?)"|\b[A-Z][a-zA-Z]+\b', query)
        # Extract potential entities (multi-word capitalized phrases)
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', query)
        
        # Clean up keywords: lowercase and remove duplicates/empty strings
        keywords = list(set([k.strip().lower() for k in keywords if k]))
        # Clean up entities: remove duplicates and single words already in keywords
        entities = list(set([e.strip() for e in entities if len(e.split()) > 1 or e in keywords]))
        # Further refine keywords: remove any that are now part of multi-word entities
        entity_words = set(word.lower() for entity in entities for word in entity.split())
        keywords = [kw for kw in keywords if kw not in entity_words and kw not in [e.lower() for e in entities]]

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
