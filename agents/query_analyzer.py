# agents/query_analyzer.py
from .base import BaseAgent
import re

class QueryAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing the user query."""

    def classify_query(self, query: str) -> str:
        """Basic keyword-based query classification."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "contrast", "difference", "similarities"]):
            return "comparative"
        elif any(word in query_lower for word in ["how", "why", "explain", "impact", "affect", "factors", "consequences"]):
            return "causal/analytical"
        elif any(word in query_lower for word in ["when", "who", "what", "list", "describe", "name"]):
            # 'what' and 'describe' can be ambiguous, leaning towards factual for now
            return "factual"
        else:
            return "unknown"

    def extract_keywords_entities(self, query: str) -> tuple[list[str], list[str]]:
        """Basic keyword and potential named entity extraction."""
        # Simple keyword extraction (lowercase, remove punctuation, split)
        keywords = re.findall(r'\b\w+\b', query.lower())
        # Rudimentary entity extraction (Capitalized words, not at the start)
        # This is very basic and will have many false positives/negatives
        potential_entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        # Filter out common words that might start a sentence if logic was more complex
        # For simplicity, we accept this basic approach for now.
        return keywords, potential_entities

    def rewrite_query(self, query: str, query_type: str, keywords: list, entities: list) -> str:
        """Placeholder for query rewriting/decomposition."""
        # In a real system, this would use the analysis to reformulate complex queries.
        # Example: Decompose "Compare X and Y" into "What is X?" and "What is Y?" and "How are X and Y similar/different?"
        # For now, it just returns the original query.
        print("📝 Query rewriting/decomposition (placeholder)...")
        return query # No changes yet

    def run(self, query: str) -> dict:
        """Analyzes the query, classifies it, extracts keywords/entities, and prepares for retrieval."""
        print(f"🤔 Analyzing query: '{query}'")

        query_type = self.classify_query(query)
        keywords, potential_entities = self.extract_keywords_entities(query)
        refined_query = self.rewrite_query(query, query_type, keywords, potential_entities)

        analysis = {
            "original_query": query,
            "query_type": query_type,
            "keywords": keywords,
            "potential_entities": potential_entities,
            "refined_query": refined_query # Use this for retrieval
        }
        print(f"✅ Query analysis complete: Type='{analysis['query_type']}', Keywords={len(analysis['keywords'])}, Entities={len(analysis['potential_entities'])}")
        return analysis
