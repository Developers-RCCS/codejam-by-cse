# agents/orchestrator.py
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent
from .web_search_agent import WebSearchAgent
from .conversation_memory import ConversationMemoryAgent
import time
import numpy as np
from gemini_utils import embed_text
from typing import List, Dict, Tuple, Set, Any
import logging
import re
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Orchestrator")

class SourceConflictDetector:
    """Class for detecting and resolving conflicts between different information sources."""
    
    def __init__(self):
        """Initialize the conflict detector."""
        # Source reliability scores (configurable)
        self.source_reliability = {
            # Primary sources
            "textbook": 0.85,  # Textbook is highly reliable but may be dated
            
            # Web sources with reliability tiers
            # Tier 1: High reliability (educational/institutional)
            "web_en.wikipedia.org": 0.80,  # Wikipedia is generally reliable
            "web_kids.nationalgeographic.com": 0.85,  # National Geographic Kids is reliable
            "web_www.britannica.com": 0.82,  # Encyclopedia Britannica
            "web_www.bbc.co.uk": 0.80,  # BBC
            "web_airandspace.si.edu": 0.88,  # Smithsonian Museum sources are highly reliable
            "web_encyclopedia.ushmm.org": 0.88,  # Holocaust Museum is highly reliable
            
            # Tier 2: Medium reliability
            "web_www.history.com": 0.75,  # History Channel
            "web_www.nasa.gov": 0.85,  # NASA
            "web_www.nationalgeographic.com": 0.82,  # National Geographic
            
            # Tier 3: Mixed reliability
            "web_www.historyforkids.net": 0.70,
            "web_www.ducksters.com": 0.65,
            
            # Default for other web sources
            "web": 0.60,
            
            # Recency-adjusted sources (empty for now, filled during analysis)
            "recent_web": 0.00,  # Placeholder, actual value set during conflict resolution
        }
        
        # Embedding cache for semantic comparison
        self.embedding_cache = {}
        
        # Track conflicts for analysis
        self.detected_conflicts = []
        self.conflict_resolutions = []
    
    def reset_conflict_tracking(self):
        """Reset conflict tracking for a new query."""
        self.detected_conflicts = []
        self.conflict_resolutions = []
        
    def get_source_reliability(self, chunk: Dict) -> float:
        """
        Get the reliability score for a chunk based on its source with enhanced domain recognition.
        
        Args:
            chunk: Content chunk with metadata
            
        Returns:
            float: Reliability score from 0.0 to 1.0
        """
        metadata = chunk.get("metadata", {})
        source_type = metadata.get("source_type", "")
        
        # Handle textbook sources
        if source_type == "textbook":
            # Check if this is an older chapter/section that might be outdated
            page_num = metadata.get("page", 0)
            if isinstance(page_num, str):
                try:
                    page_num = int(page_num)
                except ValueError:
                    page_num = 0
                    
            # If we have info about textbook recency, we could adjust here
            # For now, return the base textbook reliability
            return self.source_reliability["textbook"]
            
        # Handle web sources with domain-specific reliability
        elif source_type == "web":
            url = metadata.get("url", "")
            if url:
                # Extract domain for reliability lookup
                domain = urlparse(url).netloc if '//' in url else url.split('/')[0]
                domain_key = f"web_{domain}"
                
                if domain_key in self.source_reliability:
                    reliability = self.source_reliability[domain_key]
                    
                    # Apply recency boost based on published date if available
                    published_date = metadata.get("published_date")
                    if published_date:
                        try:
                            # Parse date and apply recency boost
                            date_obj = datetime.strptime(published_date, "%Y-%m-%d")
                            now = datetime.now()
                            days_old = (now - date_obj).days
                            
                            # Boost for content less than 1 year old
                            if days_old < 365:
                                recency_boost = 0.1 * (1 - (days_old / 365))
                                return min(0.95, reliability + recency_boost)
                        except:
                            pass  # If date parsing fails, continue with standard reliability
                            
                    return reliability
            
            # Use default web reliability if no specific score
            return self.source_reliability["web"]
            
        # Default reliability for unknown sources
        return 0.5
    
    def evaluate_content_semantic_similarity(self, chunk1: Dict, chunk2: Dict) -> float:
        """
        Evaluate semantic similarity between two chunks using embeddings.
        
        Args:
            chunk1: First content chunk
            chunk2: Second content chunk
            
        Returns:
            float: Similarity score from 0.0 to 1.0
        """
        # Get embeddings for each chunk, using cache if available
        chunk1_id = id(chunk1)
        if chunk1_id not in self.embedding_cache:
            self.embedding_cache[chunk1_id] = np.array(embed_text(chunk1["text"][:1000]), dtype="float32")
        embedding1 = self.embedding_cache[chunk1_id]
        
        chunk2_id = id(chunk2)
        if chunk2_id not in self.embedding_cache:
            self.embedding_cache[chunk2_id] = np.array(embed_text(chunk2["text"][:1000]), dtype="float32")
        embedding2 = self.embedding_cache[chunk2_id]
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def extract_key_facts(self, text: str) -> List[str]:
        """
        Extract potential key facts from text for contradiction detection.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List[str]: List of extracted key facts
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for sentences that likely contain facts (numbers, dates, named entities)
        fact_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',  # Dates
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',  # Month Day, Year
            r'\b\d+\s+(percent|per cent|%)\b',  # Percentages
            r'\b(first|second|third|fourth|fifth|last|final)\b',  # Ordinals
            r'\b(not|never|failed|succeeded|discovered|invented|created|built|destroyed)\b',  # Key verbs
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Potential named entities
        ]
        
        potential_facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence matches any fact pattern
            for pattern in fact_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    potential_facts.append(sentence)
                    break
                    
        return potential_facts
    
    def detect_conflicts(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Detect potential conflicts between chunks from different sources with enhanced detection.
        
        Args:
            chunks: List of content chunks
            
        Returns:
            Dict: Map of conflict topics to list of conflicting chunks
        """
        if len(chunks) < 2:
            return {}
        
        # Reset conflict tracking
        self.reset_conflict_tracking()
            
        # Group chunks by high-level topics
        topics = {}
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.get("text", ""):
                continue
                
            # Get or create embedding for this chunk
            chunk_id = id(chunk)
            if chunk_id not in self.embedding_cache:
                self.embedding_cache[chunk_id] = np.array(embed_text(chunk["text"][:1000]), dtype="float32")
            chunk_embedding = self.embedding_cache[chunk_id]
            
            # Find the most similar topic or create a new one
            best_topic = None
            best_similarity = 0.6  # Threshold for considering chunks as the same topic
            
            for topic, topic_chunks in topics.items():
                # Compare to first chunk in the topic
                topic_chunk_id = id(topic_chunks[0])
                if topic_chunk_id not in self.embedding_cache:
                    self.embedding_cache[topic_chunk_id] = np.array(embed_text(topic_chunks[0]["text"][:1000]), dtype="float32")
                topic_embedding = self.embedding_cache[topic_chunk_id]
                
                # Calculate similarity
                similarity = np.dot(chunk_embedding, topic_embedding) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(topic_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_topic = topic
            
            if best_topic:
                topics[best_topic].append(chunk)
            else:
                # Create a new topic using first 50 chars as identifier
                topic_id = chunk["text"][:50].strip()
                topics[topic_id] = [chunk]
        
        # Find topics with potential conflicts (different sources)
        conflicts = {}
        for topic, topic_chunks in topics.items():
            if len(topic_chunks) < 2:
                continue
                
            # Check if chunks are from different source types
            source_types = set(chunk.get("metadata", {}).get("source_type", "") for chunk in topic_chunks)
            
            # If we have mixed sources, analyze for contradictions
            if len(source_types) > 1 or (len(topic_chunks) >= 3 and "web" in source_types):
                # Check content for contradiction patterns
                has_contradiction = self._check_for_contradiction(topic, topic_chunks)
                if has_contradiction:
                    conflicts[topic] = topic_chunks
                    
                    # Log detection
                    self.detected_conflicts.append({
                        "topic": topic,
                        "sources": [c.get("metadata", {}).get("source_type", "unknown") for c in topic_chunks],
                        "chunks": len(topic_chunks),
                        "contradiction_details": has_contradiction
                    })
        
        # Log conflict detection results
        if conflicts:
            logger.info(f"Detected {len(conflicts)} topics with potential source conflicts")
        
        return conflicts
    
    def _check_for_contradiction(self, topic: str, chunks: List[Dict]) -> Dict:
        """
        Check if chunks potentially contradict each other with improved detection mechanisms.
        
        Args:
            topic: Topic identifier
            chunks: List of potentially conflicting chunks
            
        Returns:
            Dict: Contradiction information or False if no contradiction
        """
        # Extract text content from each chunk
        texts = [chunk["text"].lower() for chunk in chunks]
        
        # Method 1: Pattern-based contradiction detection
        contradiction_patterns = [
            {"positive": "did", "negative": "did not", "subject": ""},
            {"positive": "was", "negative": "was not", "subject": ""},
            {"positive": "were", "negative": "were not", "subject": ""},
            {"positive": "is", "negative": "is not", "subject": ""},
            {"positive": "are", "negative": "are not", "subject": ""},
            {"positive": "can", "negative": "cannot", "subject": ""},
            {"positive": "has", "negative": "has not", "subject": ""},
            {"positive": "does", "negative": "does not", "subject": ""},
            {"positive": "successful", "negative": "unsuccessful", "subject": ""},
            {"positive": "true", "negative": "false", "subject": ""},
            {"positive": "correct", "negative": "incorrect", "subject": ""},
            {"positive": "before", "negative": "after", "subject": ""}
        ]
        
        # Check for direct contradictions
        for pattern in contradiction_patterns:
            pos, neg = pattern["positive"], pattern["negative"]
            
            for i, text_a in enumerate(texts):
                if pos in text_a:
                    # Get surrounding context to identify the subject
                    pos_context = self._extract_context(text_a, pos, window=30)
                    
                    for j, text_b in enumerate(texts):
                        if i != j and neg in text_b:
                            # Get surrounding context
                            neg_context = self._extract_context(text_b, neg, window=30)
                            
                            # If contexts are discussing the same subject
                            similarity = self._text_similarity(pos_context, neg_context)
                            if similarity > 0.6:
                                return {
                                    "type": "direct_contradiction",
                                    "pattern": f"{pos} vs {neg}",
                                    "context_a": pos_context,
                                    "context_b": neg_context,
                                    "similarity": similarity,
                                    "chunk_indices": [i, j]
                                }
        
        # Method 2: Numerical fact contradiction
        numerical_patterns = [
            r'(\d{4})\s*(-|to)\s*(\d{4})',  # Date ranges like 1939-1945
            r'(in|on|around|about)\s+(\d{4})',  # Years like "in 1939"
            r'(\d+)\s+(percent|per cent|%)',  # Percentages
            r'(\d+)\s+(people|persons|soldiers|casualties)',  # Counts of people
        ]
        
        # Extract numerical facts from each text
        for pattern in numerical_patterns:
            for i, text_a in enumerate(texts):
                matches_a = re.finditer(pattern, text_a, re.IGNORECASE)
                
                for match_a in matches_a:
                    # Get the full match and the surrounding context
                    num_fact_a = match_a.group(0)
                    context_a = self._extract_context(text_a, num_fact_a, window=50)
                    
                    for j, text_b in enumerate(texts):
                        if i == j:
                            continue
                            
                        # Look for similar context but different numbers
                        matches_b = re.finditer(pattern, text_b, re.IGNORECASE)
                        for match_b in matches_b:
                            num_fact_b = match_b.group(0)
                            
                            # If the numerical facts are different
                            if num_fact_a != num_fact_b:
                                context_b = self._extract_context(text_b, num_fact_b, window=50)
                                
                                # Check if contexts are similar but numbers differ
                                context_similarity = self._text_similarity(context_a, context_b, exclude=num_fact_a)
                                if context_similarity > 0.65:
                                    return {
                                        "type": "numerical_contradiction",
                                        "fact_a": num_fact_a,
                                        "fact_b": num_fact_b,
                                        "context_a": context_a,
                                        "context_b": context_b,
                                        "similarity": context_similarity,
                                        "chunk_indices": [i, j]
                                    }
        
        # Method 3: Named entity attribution contradictions
        # Simplified version - look for sentences that mention the same entity with contradictory actions
        for i, text_a in enumerate(texts):
            sentences_a = re.split(r'[.!?]+', text_a)
            
            for sentence_a in sentences_a:
                # Look for sentences with entities (simplified as capitalized words)
                entities = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', sentence_a)
                
                for entity in entities:
                    entity_lower = entity.lower()
                    
                    # Skip common non-entity capitalized words
                    if entity_lower in ['i', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
                                       'august', 'september', 'october', 'november', 'december']:
                        continue
                        
                    # Find sentences in other chunks that mention the same entity
                    for j, text_b in enumerate(texts):
                        if i == j:
                            continue
                            
                        sentences_b = re.split(r'[.!?]+', text_b)
                        for sentence_b in sentences_b:
                            if entity.lower() in sentence_b.lower():
                                # Check if actions/descriptions are different with similar context
                                # Get verb phrases around the entity
                                a_context = self._extract_context(sentence_a, entity, window=60)
                                b_context = self._extract_context(sentence_b, entity, window=60)
                                
                                if a_context and b_context:
                                    # Check semantic similarity of contexts
                                    embedding_a = embed_text(a_context)
                                    embedding_b = embed_text(b_context)
                                    similarity = np.dot(embedding_a, embedding_b) / (
                                        np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
                                    )
                                    
                                    # If contexts are somewhat similar but not too similar
                                    # (might be discussing same entity but with conflicting info)
                                    if 0.6 <= similarity <= 0.85:
                                        return {
                                            "type": "entity_attribution",
                                            "entity": entity,
                                            "context_a": a_context,
                                            "context_b": b_context,
                                            "similarity": float(similarity),
                                            "chunk_indices": [i, j]
                                        }
        
        return False
    
    def _extract_context(self, text: str, target: str, window: int = 30) -> str:
        """Extract text surrounding a target phrase."""
        if not target in text:
            return ""
            
        index = text.find(target)
        start = max(0, index - window)
        end = min(len(text), index + len(target) + window)
        return text[start:end].strip()
    
    def _text_similarity(self, text_a: str, text_b: str, exclude: str = "") -> float:
        """Calculate semantic similarity between two text snippets."""
        # Remove the exclusion text if provided
        if exclude:
            text_a = text_a.replace(exclude, "")
            text_b = text_b.replace(exclude, "")
            
        # Clean up
        text_a = text_a.strip()
        text_b = text_b.strip()
        
        if not text_a or not text_b:
            return 0.0
            
        # Get embeddings
        embedding_a = embed_text(text_a)
        embedding_b = embed_text(text_b)
        
        # Calculate similarity
        similarity = np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        )
        
        return float(similarity)
    
    def resolve_conflicts(self, conflicts: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Resolve conflicts between chunks with enhanced reliability analysis and multi-perspective support.
        
        Args:
            conflicts: Map of topics to conflicting chunks
            
        Returns:
            Dict: Map of topics to resolution info
        """
        resolutions = {}
        
        for topic, chunks in conflicts.items():
            # Calculate reliability-weighted evidence
            weighted_chunks = []
            for chunk in chunks:
                reliability = self.get_source_reliability(chunk)
                
                # Recency boost for web sources with recent information
                recency_boost = 0
                metadata = chunk.get("metadata", {})
                if metadata.get("source_type") == "web":
                    # Default small boost for web (assumes more recent than textbook)
                    recency_boost = 0.05
                    
                    # Additional boost for explicitly recent content
                    if "published_date" in metadata:
                        try:
                            pub_date = datetime.strptime(metadata["published_date"], "%Y-%m-%d")
                            now = datetime.now()
                            if (now - pub_date).days < 365:  # Less than a year old
                                recency_boost = 0.1
                        except:
                            pass  # If date parsing fails, use default boost
                
                weighted_chunks.append({
                    "chunk": chunk,
                    "reliability": reliability,
                    "recency_boost": recency_boost,
                    "weight": reliability + recency_boost,
                    "source_type": chunk.get("metadata", {}).get("source_type", "unknown"),
                    "domain": self._extract_domain(chunk.get("metadata", {}).get("url", ""))
                })
            
            # Sort by weight
            sorted_chunks = sorted(weighted_chunks, key=lambda x: x["weight"], reverse=True)
            
            # Determine if conflict needs presentation of multiple perspectives
            weights = [c["weight"] for c in sorted_chunks]
            
            # Multiple perspectives criteria:
            # 1. If top two sources are close in reliability (within 0.15)
            # 2. If sources are from different tiers (e.g., textbook vs. web)
            # 3. If the conflict involves a significant numerical or factual disagreement
            
            multiple_perspectives = False
            perspective_reason = ""
            
            # Check weight difference
            if len(weights) >= 2 and (weights[0] - weights[1]) < 0.15:
                multiple_perspectives = True
                perspective_reason = "similar_reliability"
            
            # Check source types
            source_types = set(c["source_type"] for c in sorted_chunks)
            if len(source_types) > 1 and "textbook" in source_types and len(chunks) > 2:
                multiple_perspectives = True
                perspective_reason = "varied_sources"
                
            # Get conflict type if detected
            conflict_info = self._check_for_contradiction(topic, [c["chunk"] for c in sorted_chunks])
            if conflict_info and conflict_info.get("type") == "numerical_contradiction":
                multiple_perspectives = True
                perspective_reason = "numerical_conflict"
            
            # Create resolution info
            resolution = {
                "primary": sorted_chunks[0]["chunk"],
                "alternatives": [c["chunk"] for c in sorted_chunks[1:3]] if multiple_perspectives else [], # Limit to top 2 alternatives
                "multiple_perspectives": multiple_perspectives,
                "perspective_reason": perspective_reason,
                "confidence": sorted_chunks[0]["weight"],
                "confidence_margin": weights[0] - weights[1] if len(weights) > 1 else 1.0,
                "sources": {c["source_type"]: c["reliability"] for c in sorted_chunks},
                "source_weights": {c["domain"] if c["domain"] else c["source_type"]: c["weight"] for c in sorted_chunks},
                "conflict_info": conflict_info
            }
            
            resolutions[topic] = resolution
            
            # Track resolution for analysis
            self.conflict_resolutions.append({
                "topic": topic,
                "resolution_type": "multiple_perspectives" if multiple_perspectives else "single_source",
                "reason": perspective_reason,
                "primary_source": sorted_chunks[0]["source_type"],
                "primary_domain": sorted_chunks[0]["domain"],
                "alternatives_count": len(resolution["alternatives"]),
                "confidence": resolution["confidence"]
            })
        
        return resolutions
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
            
        try:
            domain = urlparse(url).netloc
            return domain
        except:
            return ""
    
    def get_conflict_statistics(self) -> Dict:
        """
        Get statistics about detected conflicts and resolutions.
        
        Returns:
            Dict: Statistics about conflicts
        """
        return {
            "total_conflicts_detected": len(self.detected_conflicts),
            "conflicts_by_type": self._count_by_field(self.detected_conflicts, 
                                                    lambda x: x.get("contradiction_details", {}).get("type", "unknown")),
            "resolutions_by_type": self._count_by_field(self.conflict_resolutions, "resolution_type"),
            "sources_in_conflicts": self._sources_in_conflicts(),
            "primary_resolution_sources": self._count_by_field(self.conflict_resolutions, "primary_source"),
            "multi_perspective_reasons": self._count_by_field(self.conflict_resolutions, "reason")
        }
    
    def _count_by_field(self, items: List[Dict], field_or_func) -> Dict:
        """Count items by a field or function result."""
        result = defaultdict(int)
        
        for item in items:
            if callable(field_or_func):
                key = field_or_func(item)
            else:
                key = item.get(field_or_func, "unknown")
                
            result[key] += 1
            
        return dict(result)
    
    def _sources_in_conflicts(self) -> Dict[str, int]:
        """Count sources involved in conflicts."""
        sources = defaultdict(int)
        
        for conflict in self.detected_conflicts:
            for source in conflict.get("sources", []):
                sources[source] += 1
                
        return dict(sources)


class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the QA workflow."""
    def __init__(self):
        print("🚀 Initializing Orchestrator and sub-agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.reference_tracker = ReferenceTrackerAgent()
        self.context_expander = ContextExpansionAgent()
        self.web_search_agent = WebSearchAgent(cache_dir="web_cache")
        self.conflict_detector = SourceConflictDetector()
        # Add conversation memory agent
        self.conversation_memory = ConversationMemoryAgent()
        print("✅ Orchestrator ready.")
    
    def _is_conversational_query(self, query: str) -> bool:
        """
        Determine if a query is conversational rather than information-seeking.
        
        Args:
            query: The user's query
            
        Returns:
            bool: True if the query seems conversational
        """
        conversational_patterns = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bthanks\b", r"\bthank you\b", 
            r"\bhow are you\b", r"\bnice to meet\b", r"\bbye\b", r"\bsee you\b",
            r"\bgood morning\b", r"\bgood afternoon\b", r"\bgood evening\b",
            r"\bgood night\b", r"\bwhat's up\b", r"\bwhat about you\b"
        ]
        
        # Check for greetings and conversational phrases
        query_lower = query.lower()
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return True
                
        # Check if very short (likely a greeting or acknowledgment)
        if len(query_lower.split()) <= 2:
            return True
            
        return False
    
    def _get_conversational_response(self, query: str, conversation_context: dict) -> str:
        """
        Generate a purely conversational response without information retrieval.
        
        Args:
            query: User query
            conversation_context: Current conversation context
            
        Returns:
            str: Conversational response
        """
        # Simple response templates for common conversational queries
        greetings = {
            "hello": ["Hi there! I'm Yuhasa, your history tutor. What historical topic are you interested in exploring today?",
                     "Hello! I'm excited to help you learn about history today. What would you like to discuss?"],
            "hi": ["Hi! I'm Yuhasa, your friendly history tutor. What historical period or event would you like to learn about?",
                  "Hello there! Ready to explore some fascinating history together? What topic interests you?"],
            "hey": ["Hey! I'm Yuhasa. I'm here to make learning history fun and engaging. What shall we explore today?",
                   "Hey there! What aspect of history are you curious about today?"],
            "thank": ["You're very welcome! I'm glad I could help. Is there anything else about history you'd like to explore?",
                     "It's my pleasure! I really enjoy discussing history. What else would you like to know about?"],
            "bye": ["Goodbye! Feel free to come back anytime you have history questions. Have a great day!",
                   "See you later! I'm always here to help with your history studies. Take care!"],
            "good morning": ["Good morning! A perfect time to explore some fascinating historical events. What would you like to learn today?"],
            "good afternoon": ["Good afternoon! Ready for some historical exploration? What topic shall we dive into?"],
            "good evening": ["Good evening! There's always time for a bit of history. What would you like to discuss?"]
        }
        
        # Check for matching patterns
        query_lower = query.lower()
        
        for key, responses in greetings.items():
            if key in query_lower:
                return responses[0] if not responses else random.choice(responses)
                
        # Default conversational response
        state = conversation_context.get("conversation_state", "greeting")
        rapport = conversation_context.get("rapport_level", 0)
        
        if state == "greeting" or rapport < 3:
            return "I'm Yuhasa, your history tutor. I specialize in Grade 11 history content. What historical topic would you like to explore today?"
        else:
            return "I'm here to help with any history questions you have. What aspect of history shall we explore next?"

    def _should_use_web_search(self, query_analysis: dict, retrieved_chunks: list) -> bool:
        """
        Determine if web search should be used based on query analysis and retrieved content.
        
        Args:
            query_analysis: Analysis of the user query
            retrieved_chunks: Chunks retrieved from the textbook
            
        Returns:
            bool: True if web search should be used
        """
        # Check if we have any relevant textbook chunks
        if not retrieved_chunks:
            print("📋 No relevant textbook chunks found, will use web search")
            return True
            
        # Check confidence of top textbook chunks
        top_confidence = retrieved_chunks[0].get("confidence", 0) if retrieved_chunks else 0
        if top_confidence < 0.4:
            print(f"📋 Low confidence in textbook results ({top_confidence:.2f}), will supplement with web search")
            return True
            
        # Check for specific entities that might indicate web knowledge is needed
        web_likely_entities = ["Wright brothers", "Hitler", "Marie Antoinette", "Mahaweli", "Sri Lanka"]
        entities = query_analysis.get("entities", [])
        
        for entity in entities:
            for web_entity in web_likely_entities:
                if web_entity.lower() in entity.lower():
                    print(f"📋 Detected entity '{entity}' that may benefit from web search")
                    return True
                    
        # Check for web-specific keywords in query
        web_keywords = ["recent", "latest", "modern", "current", "today", "now", "website", "online", "internet"]
        query_lower = query_analysis.get("original_query", "").lower()
        
        for keyword in web_keywords:
            if keyword in query_lower:
                print(f"📋 Query contains '{keyword}' which suggests web search might be helpful")
                return True
                
        print("📋 Textbook content appears sufficient, no web search needed")
        return False
        
    def _combine_sources(self, textbook_chunks: list, web_chunks: list, query_analysis: dict) -> list:
        """
        Combine textbook and web sources with intelligent source balancing and conflict resolution.
        
        Args:
            textbook_chunks: Chunks from textbook
            web_chunks: Chunks from web search
            query_analysis: Analysis of the user query
            
        Returns:
            list: Combined and prioritized chunks
        """
        start_time = time.time()
        
        if not web_chunks:
            logger.info("No web chunks to combine, using textbook chunks only")
            return textbook_chunks
            
        if not textbook_chunks:
            logger.info("No textbook chunks to combine, using web chunks only")
            return web_chunks
        
        # Add source type to metadata
        for chunk in textbook_chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["source_type"] = "textbook"
            
        for chunk in web_chunks:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["source_type"] = "web"
            
        # Combine chunks
        combined = textbook_chunks + web_chunks
        
        # Sort by confidence
        sorted_chunks = sorted(combined, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Check for potential conflicts in the content
        conflicts = self.conflict_detector.detect_conflicts(sorted_chunks)
        
        if conflicts:
            # Resolve conflicts using the conflict detector
            resolutions = self.conflict_detector.resolve_conflicts(conflicts)
            logger.info(f"Detected and resolved {len(conflicts)} potential source conflicts")
            
            # Adjust confidence scores for conflicting chunks
            for topic, resolution in resolutions.items():
                primary_chunk = resolution["primary"]
                alternatives = resolution["alternatives"]
                
                # Boost the primary chunk's confidence
                if "confidence" in primary_chunk:
                    primary_chunk["confidence"] = min(0.99, primary_chunk["confidence"] * 1.1)
                
                # Tag chunks with conflict resolution metadata
                primary_chunk["metadata"]["conflict_resolution"] = "primary"
                primary_chunk["metadata"]["multiple_perspectives"] = resolution["multiple_perspectives"]
                primary_chunk["metadata"]["conflict_confidence"] = resolution["confidence"]
                
                for alt_chunk in alternatives:
                    alt_chunk["metadata"]["conflict_resolution"] = "alternative"
        
        # Re-sort after potential confidence adjustments
        sorted_chunks = sorted(sorted_chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Calculate balance ratio based on query needs
        # Default balance: 60% textbook, 40% web
        textbook_ratio = 0.6
        web_ratio = 0.4
        
        # Adjust ratio based on query analysis
        query_lower = query_analysis.get("original_query", "").lower()
        web_focus_keywords = ["recent", "latest", "current", "today", "now"]
        if any(keyword in query_lower for keyword in web_focus_keywords):
            # More emphasis on web for recency-focused queries
            textbook_ratio = 0.3
            web_ratio = 0.7
            
        # Select chunks based on the ratio
        result = []
        max_chunks = 6  # Maximum chunks to avoid context overflow
        textbook_target = int(max_chunks * textbook_ratio)
        web_target = int(max_chunks * web_ratio)
        
        textbook_added = 0
        web_added = 0
        
        # First ensure we include primary chunks from conflict resolution
        conflict_chunks = []
        for chunk in sorted_chunks:
            if chunk["metadata"].get("conflict_resolution") == "primary":
                conflict_chunks.append(chunk)
                if chunk["metadata"]["source_type"] == "textbook":
                    textbook_added += 1
                else:
                    web_added += 1
            
            # Add alternatives if multiple perspectives needed
            if chunk["metadata"].get("conflict_resolution") == "primary" and chunk["metadata"].get("multiple_perspectives", False):
                for alt_chunk in sorted_chunks:
                    if alt_chunk["metadata"].get("conflict_resolution") == "alternative" and alt_chunk not in conflict_chunks:
                        conflict_chunks.append(alt_chunk)
                        if alt_chunk["metadata"]["source_type"] == "textbook":
                            textbook_added += 1
                        else:
                            web_added += 1
                        break  # Only add one alternative perspective
        
        # Add conflict chunks first
        result.extend(conflict_chunks)
        
        # Then add remaining high-confidence chunks while maintaining the target ratio
        for chunk in sorted_chunks:
            # Skip conflict chunks already added
            if chunk in conflict_chunks:
                continue
                
            # Stop if we've reached max chunks
            if len(result) >= max_chunks:
                break
                
            # Add textbook chunks until we hit target
            if chunk["metadata"]["source_type"] == "textbook" and textbook_added < textbook_target:
                result.append(chunk)
                textbook_added += 1
            
            # Add web chunks until we hit target    
            elif chunk["metadata"]["source_type"] == "web" and web_added < web_target:
                result.append(chunk)
                web_added += 1
        
        # If we still have room and haven't met targets, add more chunks
        if len(result) < max_chunks:
            remaining_slots = max_chunks - len(result)
            for chunk in sorted_chunks:
                if chunk not in result and len(result) < max_chunks:
                    result.append(chunk)
        
        # Log source distribution
        textbook_percent = textbook_added / max(1, len(result)) * 100
        web_percent = web_added / max(1, len(result)) * 100
        print(f"🔀 Combined sources: {textbook_added} textbook chunks ({textbook_percent:.1f}%) and {web_added} web chunks ({web_percent:.1f}%)")
        
        # Log performance
        duration = time.time() - start_time
        logger.info(f"Source combination completed in {duration:.3f}s")
        
        return result

    def run(self, query: str, chat_history: list = None, session_id: str = None) -> dict:
        """Runs the full QA pipeline with enhanced query handling, retrieval, context expansion, and personalization."""
        print(f"\n🔄 Orchestrating response for query: '{query}'")

        # Initialize or get existing session with conversation memory
        if not session_id or not self.conversation_memory.current_session_id:
            session_id = self.conversation_memory.start_new_session(session_id)
            print(f"📝 Started new conversation session: {session_id}")
        elif session_id != self.conversation_memory.current_session_id:
            self.conversation_memory.start_new_session(session_id)
            print(f"📝 Switched to conversation session: {session_id}")
            
        # Add user message to conversation memory
        self.conversation_memory.add_message("user", query)
            
        # Get conversation context for personalization
        conversation_context = self.conversation_memory.generate_personalized_context()
        
        # Check if this is a conversational query rather than information-seeking
        if self._is_conversational_query(query):
            print("💬 Detected conversational query, generating social response")
            response = self._get_conversational_response(query, conversation_context)
            
            # Add response to conversation memory
            self.conversation_memory.add_message("bot", response)
            
            return {
                "answer": response,
                "references": {},
                "query_analysis": {"query_type": "conversational", "original_query": query},
                "retrieved_chunks": []
            }

        # 1. Analyze Query
        query_analysis = self.query_analyzer.run(query=query)
        refined_query = query_analysis.get("refined_query", query)
        
        # Handle complex queries with decomposition if needed
        if query_analysis.get("needs_decomposition", False) and len(query_analysis.get("sub_queries", [])) > 1:
            print(f"🧩 Complex query detected, processing {len(query_analysis['sub_queries'])} sub-queries")
            # This would be implemented to run sub-queries separately and combine results
            # For now, we'll continue with the main refined query

        # 2. Retrieve Context with advanced retriever
        retrieved_chunks = self.retriever.run(
            query=refined_query, 
            query_analysis=query_analysis
        )

        # 3. Decide if web search is needed
        web_chunks = []
        if self._should_use_web_search(query_analysis, retrieved_chunks):
            print("🌐 Supplementing with web search...")
            web_chunks = self.web_search_agent.run(
                query=refined_query,
                query_analysis=query_analysis,
                max_results=3
            )

        # 4. Combine sources if web results are available
        if web_chunks:
            combined_chunks = self._combine_sources(retrieved_chunks, web_chunks, query_analysis)
        else:
            combined_chunks = retrieved_chunks

        # 5. Assess & Expand Context
        final_context_chunks, aggregated_metadata = self.context_expander.run(
            retrieved_chunks=combined_chunks,
            query_analysis=query_analysis,
            retriever_agent=self.retriever
        )

        # Add chat history to conversation context for continuity
        conversation_context["chat_history"] = self.conversation_memory.get_chat_history(3)

        if not final_context_chunks:
            print("⚠️ No relevant context found after retrieval/expansion.")
            friendly_no_info_response = "I don't have specific information about that in my sources. Could you try asking about a different aspect of history, or rephrase your question? I'd be happy to help with topics related to Grade 11 history."
            
            # Add response to conversation memory
            self.conversation_memory.add_message("bot", friendly_no_info_response)
            
            return {
                "answer": friendly_no_info_response,
                "references": {"pages": [], "sections": [], "web_sources": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 6. Generate Answer with query-type-aware prompting and personalization
        generation_result = self.generator.run(
            query=query_analysis["original_query"], 
            context_chunks=final_context_chunks,
            query_analysis=query_analysis,
            conversation_context=conversation_context
        )
        
        # Extract answer and explained concepts
        answer = generation_result["answer"] if isinstance(generation_result, dict) else generation_result
        explained_concepts = generation_result.get("explained_concepts", []) if isinstance(generation_result, dict) else []
        
        # Record explained concepts in conversation memory
        for concept in explained_concepts:
            self.conversation_memory.record_explained_concept(concept)

        # 7. Use the metadata from context expander for references
        references = aggregated_metadata
        
        # 8. Extract web sources for citation
        web_sources = []
        for chunk in final_context_chunks:
            if chunk["metadata"].get("source_type") == "web" and "url" in chunk["metadata"]:
                source_url = chunk["metadata"]["url"]
                source_topic = chunk["metadata"].get("topic", "")
                
                # Check if already added
                if not any(source["url"] == source_url for source in web_sources):
                    web_sources.append({
                        "url": source_url,
                        "topic": source_topic
                    })
        
        # Add web sources to references
        references["web_sources"] = web_sources
        
        # Add response to conversation memory
        self.conversation_memory.add_message("bot", answer, {"explained_concepts": explained_concepts})

        print("✅ Orchestration complete.")
        return {
            "answer": answer,
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
