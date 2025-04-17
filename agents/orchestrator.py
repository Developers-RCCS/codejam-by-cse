# agents/orchestrator.py
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the QA workflow."""
    def __init__(self):
        print("🚀 Initializing Orchestrator and sub-agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.reference_tracker = ReferenceTrackerAgent()
        self.context_expander = ContextExpansionAgent()
        print("✅ Orchestrator ready.")

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

        # 3. Assess & Expand Context
        final_context_chunks, aggregated_metadata = self.context_expander.run(
            retrieved_chunks=retrieved_chunks,
            query_analysis=query_analysis,
            retriever_agent=self.retriever
        )

        if not final_context_chunks:
            print("⚠️ No relevant context found after retrieval/expansion.")
            return {
                "answer": "I couldn't find relevant information in the textbook to answer that question.",
                "references": {"pages": [], "sections": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 4. Generate Answer with query-type-aware prompting
        answer = self.generator.run(
            query=query_analysis["original_query"], 
            context_chunks=final_context_chunks,
            query_analysis=query_analysis
        )

        # 5. Use the metadata from context expander for references
        references = aggregated_metadata

        # 6. Format final output
        final_answer = answer
        
        print("✅ Orchestration complete.")
        return {
            "answer": final_answer,
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
