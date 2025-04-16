# agents/query_analyzer.py
from .base import BaseAgent
import re
import string

class QueryAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing the user query."""

    def __init__(self):
        self.question_starters = {
            "factual": ["what", "when", "where", "who", "which", "list", "name", "identify", "define"],
            "causal/analytical": ["why", "how", "explain", "analyze", "describe", "what caused", "what were the effects", 
                                 "what factors", "what was the impact", "how did", "in what ways"],
            "comparative": ["compare", "contrast", "what are the differences", "what are the similarities", 
                           "how does", "how do", "what is the relationship", "distinguish between"]
        }
        self.multi_part_indicators = ["and", "also", "additionally", "moreover", "furthermore", "trace", "outline"]

    def classify_query(self, query: str) -> dict:
        """Classify query by type and complexity."""
        query_lower = query.lower()
        query_type = "unknown"
        complexity = "simple"
        
        # Check for multi-part or complex questions
        if any(indicator in query_lower for indicator in self.multi_part_indicators):
            complexity = "complex"
            
        # Count the number of question marks as a simple heuristic for multiple questions
        if query_lower.count("?") > 1:
            complexity = "complex"
            
        # Check for comparative questions
        if any(term in query_lower for term in self.question_starters["comparative"]):
            query_type = "comparative"
            
        # Check for causal/analytical questions
        elif any(term in query_lower for term in self.question_starters["causal/analytical"]):
            query_type = "causal/analytical"
            
        # Check for factual questions
        elif any(term in query_lower for term in self.question_starters["factual"]):
            query_type = "factual"
            
        # Check for likely multi-part complex queries by length and structure
        if len(query.split()) > 15:
            complexity = "complex"  # Longer queries are more likely complex
            
        return {"type": query_type, "complexity": complexity}

    def extract_entities(self, query: str) -> list:
        """Extract potential named entities from query."""
        # Clean query
        query = query.strip()
        
        # Extract capitalized phrases (likely entities)
        capitalized_pattern = r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b'
        potential_entities = re.findall(capitalized_pattern, query)
        
        # Filter out sentence starters that are capitalized
        if potential_entities and query.startswith(potential_entities[0]):
            potential_entities = potential_entities[1:]
            
        # Filter out common question words that might be capitalized
        stop_words = ["What", "Who", "When", "Where", "Why", "How", "Which", "List", "Explain"]
        filtered_entities = [e for e in potential_entities if e not in stop_words]
        
        return filtered_entities

    def extract_keywords(self, query: str) -> list:
        """Extract important keywords from query."""
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        clean_query = query.translate(translator).lower()
        
        # Split into words
        words = clean_query.split()
        
        # Remove stopwords
        stopwords = ["the", "a", "an", "and", "or", "but", "if", "because", "as", "what", 
                     "which", "who", "whom", "whose", "when", "where", "why", "how", "is", 
                     "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                     "do", "does", "did", "can", "could", "will", "would", "should", "may", 
                     "might", "must", "of", "for", "about", "to", "in", "on", "by", "with"]
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords

    def rewrite_complex_query(self, query: str, query_analysis: dict) -> list:
        """Decompose complex queries into simpler sub-queries."""
        if query_analysis["complexity"] != "complex":
            return [query]  # Return original if not complex
            
        # For comparative questions, try to split into comparable parts
        if query_analysis["type"] == "comparative":
            if "compare" in query.lower() and "and" in query.lower():
                parts = query.lower().split("compare")[1].split("and")
                if len(parts) >= 2:
                    topics = [part.strip(" ,.?!") for part in parts]
                    sub_queries = [
                        f"What is {topics[0]}?",
                        f"What is {topics[1]}?",
                        f"What are the similarities between {topics[0]} and {topics[1]}?",
                        f"What are the differences between {topics[0]} and {topics[1]}?"
                    ]
                    return sub_queries
                    
        # For multi-part questions, try to split by conjunctions
        if "and" in query:
            parts = query.split("and")
            if len(parts) >= 2 and all(len(part.split()) > 3 for part in parts):
                return [part.strip() + "?" for part in parts]
                
        # For now, just return the original query if we can't properly decompose
        return [query]

    def run(self, query: str) -> dict:
        """Analyze query type, complexity, entities, and keywords."""
        print(f"🤔 Analyzing query: '{query}'")
        
        # Get query classification
        classification = self.classify_query(query)
        query_type = classification["type"]
        complexity = classification["complexity"]
        
        # Extract entities and keywords
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        
        # Determine if query needs decomposition
        sub_queries = self.rewrite_complex_query(query, classification)
        needs_decomposition = len(sub_queries) > 1
        
        analysis = {
            "original_query": query,
            "query_type": query_type,
            "complexity": complexity,
            "entities": entities,
            "keywords": keywords,
            "needs_decomposition": needs_decomposition,
            "sub_queries": sub_queries,
            "refined_query": query  # Default to original query
        }
        
        print(f"✅ Query analysis complete: Type='{query_type}', Complexity='{complexity}', " +
              f"Entities={len(entities)}, Keywords={len(keywords)}, " +
              f"Needs Decomposition={needs_decomposition}")
        
        return analysis
