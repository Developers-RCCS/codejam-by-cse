# agents/orchestrator.py
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent
from .web_search_agent import WebSearchAgent
import time
import numpy as np
from gemini_utils import embed_text
from typing import List, Dict, Tuple, Set, Any
import logging

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
            "textbook": 0.85,  # Textbook is highly reliable but may be dated
            "web_en.wikipedia.org": 0.80,  # Wikipedia is generally reliable
            "web_kids.nationalgeographic.com": 0.85,  # National Geographic Kids is reliable
            "web_www.history.com": 0.75,
            "web_www.britannica.com": 0.82,
            "web_airandspace.si.edu": 0.85,  # Museum sources are highly reliable
            "web_encyclopedia.ushmm.org": 0.85,  # Holocaust Museum is highly reliable
            "web_www.bbc.co.uk": 0.80,
            "web": 0.65,  # Default for other web sources
        }
        self.embedding_cache = {}
        
    def get_source_reliability(self, chunk: Dict) -> float:
        """
        Get the reliability score for a chunk based on its source.
        
        Args:
            chunk: Content chunk with metadata
            
        Returns:
            float: Reliability score from 0.0 to 1.0
        """
        metadata = chunk.get("metadata", {})
        source_type = metadata.get("source_type", "")
        
        if source_type == "textbook":
            return self.source_reliability["textbook"]
        elif source_type == "web":
            url = metadata.get("url", "")
            if url:
                domain = url.split('/')[2] if '//' in url else url.split('/')[0]
                domain_key = f"web_{domain}"
                if domain_key in self.source_reliability:
                    return self.source_reliability[domain_key]
            
            # Use default web reliability if no specific score
            return self.source_reliability["web"]
            
        # Default reliability
        return 0.5
    
    def detect_conflicts(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Detect potential conflicts between chunks from different sources.
        
        Args:
            chunks: List of content chunks
            
        Returns:
            Dict: Map of conflict topics to list of conflicting chunks
        """
        if len(chunks) < 2:
            return {}
            
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
            if len(source_types) > 1:
                # Check key text features to detect potential contradictions
                has_contradiction = self._check_for_contradiction(topic_chunks)
                if has_contradiction:
                    conflicts[topic] = topic_chunks
        
        return conflicts
    
    def _check_for_contradiction(self, chunks: List[Dict]) -> bool:
        """
        Check if chunks potentially contradict each other.
        
        This is a simple implementation that checks for opposing key phrases.
        A more sophisticated approach could use NLI models.
        
        Args:
            chunks: List of potentially conflicting chunks
            
        Returns:
            bool: True if contradiction detected
        """
        # Simple patterns that might indicate contradictions
        contradiction_patterns = [
            ("did not", "did"),
            ("never", "did"),
            ("false", "true"),
            ("myth", "fact"),
            ("incorrect", "correct")
        ]
        
        # Convert all text to lowercase for comparison
        texts = [chunk["text"].lower() for chunk in chunks]
        
        # Check for contradiction patterns
        for pattern_a, pattern_b in contradiction_patterns:
            for i, text_a in enumerate(texts):
                if pattern_a in text_a:
                    for j, text_b in enumerate(texts):
                        if i != j and pattern_b in text_b:
                            return True
        
        return False
    
    def resolve_conflicts(self, conflicts: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Resolve conflicts between chunks by using reliability and recency.
        
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
                recency_boost = 0.1 if chunk.get("metadata", {}).get("source_type") == "web" else 0
                
                weighted_chunks.append({
                    "chunk": chunk,
                    "reliability": reliability,
                    "weight": reliability + recency_boost,
                    "source_type": chunk.get("metadata", {}).get("source_type", "unknown")
                })
            
            # Sort by weight
            sorted_chunks = sorted(weighted_chunks, key=lambda x: x["weight"], reverse=True)
            
            # Determine if conflict needs presentation of multiple perspectives
            weights = [c["weight"] for c in sorted_chunks]
            
            # If top two sources are close in reliability, present both perspectives
            multiple_perspectives = False
            if len(weights) >= 2 and (weights[0] - weights[1]) < 0.15:
                multiple_perspectives = True
            
            # Create resolution info
            resolutions[topic] = {
                "primary": sorted_chunks[0]["chunk"],
                "alternatives": [c["chunk"] for c in sorted_chunks[1:]] if multiple_perspectives else [],
                "multiple_perspectives": multiple_perspectives,
                "confidence": sorted_chunks[0]["weight"],
                "sources": {c["source_type"]: c["reliability"] for c in sorted_chunks}
            }
        
        return resolutions


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
        print("✅ Orchestrator ready.")

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

    def run(self, query: str, chat_history: list = None) -> dict:
        """Runs the full QA pipeline with enhanced query handling, retrieval, and context expansion."""
        print(f"\n🔄 Orchestrating response for query: '{query}'")

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

        if not final_context_chunks:
            print("⚠️ No relevant context found after retrieval/expansion.")
            return {
                "answer": "I couldn't find relevant information to answer that question.",
                "references": {"pages": [], "sections": [], "web_sources": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 6. Generate Answer with query-type-aware prompting
        answer = self.generator.run(
            query=query_analysis["original_query"], 
            context_chunks=final_context_chunks,
            query_analysis=query_analysis
        )

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

        # 9. Format final output
        final_answer = answer
        
        print("✅ Orchestration complete.")
        return {
            "answer": final_answer,
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
