# agents/orchestrator.py
import logging
import time
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the QA workflow."""
    def __init__(self):
        logger.info("🚀 Initializing Orchestrator and sub-agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.reference_tracker = ReferenceTrackerAgent()
        self.context_expander = ContextExpansionAgent()
        logger.info("✅ Orchestrator ready.")

    def run(self, query: str, chat_history: list = None) -> dict:
        """Runs the full QA pipeline with query analysis, retrieval, context expansion, and generation.

        Args:
            query: The user's input query.
            chat_history: A list of previous conversation messages.

        Returns:
            A dictionary containing the final answer, references, query analysis,
            and retrieved context chunks.
        """
        orchestration_start_time = time.time()
        logger.info(f"\n🔄 Orchestrating response for query: '{query}'")

        # 1. Analyze Query
        analysis_start_time = time.time()
        logger.info("Step 1: Analyzing query...")
        query_analysis = self.query_analyzer.run(query=query)
        refined_query = query_analysis.get("refined_query", query)
        analysis_duration = time.time() - analysis_start_time
        logger.info(f"Step 1: Query analysis complete ({analysis_duration:.4f}s). Result: {query_analysis}")

        # Handle complex queries (Placeholder)
        if query_analysis.get("needs_decomposition", False) and len(query_analysis.get("sub_queries", [])) > 1:
            logger.info(f"🧩 Complex query detected, processing {len(query_analysis['sub_queries'])} sub-queries (decomposition not fully implemented)")
            # Future implementation: run sub-queries separately and combine results

        # 2. Retrieve Context
        retrieval_start_time = time.time()
        logger.info("Step 2: Retrieving initial context...")
        retrieved_chunks = self.retriever.run(
            query=refined_query,
            query_analysis=query_analysis
        )
        retrieval_duration = time.time() - retrieval_start_time
        logger.info(f"Step 2: Initial retrieval complete ({retrieval_duration:.4f}s). Retrieved {len(retrieved_chunks)} chunks.")

        # 3. Assess & Expand Context
        expansion_start_time = time.time()
        logger.info("Step 3: Assessing and expanding context...")
        final_context_chunks, aggregated_metadata = self.context_expander.run(
            retrieved_chunks=retrieved_chunks,
            query_analysis=query_analysis,
            retriever_agent=self.retriever
        )
        expansion_duration = time.time() - expansion_start_time
        logger.info(f"Step 3: Context assessment/expansion complete ({expansion_duration:.4f}s). Final context: {len(final_context_chunks)} chunks.")

        # NOTE: The generator now handles the "no relevant context" case internally based on keyword/entity check.
        # We no longer need the explicit check and fallback message generation here.

        # 4. Generate Answer
        generation_start_time = time.time()
        logger.info("Step 4: Generating answer...")
        final_answer = self.generator.run(
            query=query_analysis.get("original_query", query),
            context_chunks=final_context_chunks,
            query_analysis=query_analysis,
            chat_history=chat_history
        )
        generation_duration = time.time() - generation_start_time
        logger.info(f"Step 4: Answer generation complete ({generation_duration:.4f}s).")
        logger.debug(f"Generated answer (first 100 chars): {final_answer[:100]}")

        # 5. Final Output Formatting (References)
        references = aggregated_metadata
        logger.debug(f"Final references: {references}")

        orchestration_duration = time.time() - orchestration_start_time
        logger.info(f"✅ Orchestration complete. Total time: {orchestration_duration:.4f}s")
        if orchestration_duration > 3.0:
             logger.warning(f"⏱️ Total orchestration time ({orchestration_duration:.4f}s) exceeded target threshold of 3 seconds.")

        return {
            "answer": final_answer,
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
