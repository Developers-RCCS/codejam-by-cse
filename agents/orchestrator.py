# agents/orchestrator.py
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent
from .web_search_agent import WebSearchAgent

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
        Combine textbook and web sources based on relevance and query needs.
        
        Args:
            textbook_chunks: Chunks from textbook
            web_chunks: Chunks from web search
            query_analysis: Analysis of the user query
            
        Returns:
            list: Combined and prioritized chunks
        """
        if not web_chunks:
            return textbook_chunks
            
        if not textbook_chunks:
            return web_chunks
            
        # Add source type to metadata
        for chunk in textbook_chunks:
            chunk["metadata"]["source_type"] = "textbook"
            
        for chunk in web_chunks:
            chunk["metadata"]["source_type"] = "web"
            
        # Combine chunks
        combined = textbook_chunks + web_chunks
        
        # Sort by confidence
        sorted_chunks = sorted(combined, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Ensure diversity - don't let one source dominate completely
        result = []
        textbook_added = 0
        web_added = 0
        
        # First, add the highest confidence chunk regardless of source
        result.append(sorted_chunks[0])
        if sorted_chunks[0]["metadata"]["source_type"] == "textbook":
            textbook_added += 1
        else:
            web_added += 1
            
        # Then add remaining chunks ensuring diversity
        for chunk in sorted_chunks[1:]:
            # Limit total to 5-6 chunks to avoid context overflow
            if len(result) >= 5:
                break
                
            # Ensure at least one from each source if available
            if chunk["metadata"]["source_type"] == "textbook" and textbook_added == 0:
                result.append(chunk)
                textbook_added += 1
            elif chunk["metadata"]["source_type"] == "web" and web_added == 0:
                result.append(chunk)
                web_added += 1
            # Otherwise add based on confidence but maintain balance
            elif chunk["metadata"]["source_type"] == "textbook" and (textbook_added <= web_added or web_added >= 2):
                result.append(chunk)
                textbook_added += 1
            elif chunk["metadata"]["source_type"] == "web" and (web_added <= textbook_added or textbook_added >= 2):
                result.append(chunk)
                web_added += 1
                
        print(f"🔀 Combined sources: {textbook_added} textbook chunks and {web_added} web chunks")
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
