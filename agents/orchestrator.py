# agents/orchestrator.py
import logging
import re
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent
from utils.messages import get_random_message, CLOSING_REMARKS
from utils.text_utils import post_process_answer

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

    def _post_process_answer(self, raw_answer: str) -> str:
        """Applies final polishing touches using utility functions."""
        processed = post_process_answer(raw_answer)

        # Add a friendly closing remark if missing
        ends_with_punctuation = processed.endswith(('.', '!', '?', ';', ')'))
        ends_with_closing = any(processed.lower().endswith(remark.lower()) for remark in CLOSING_REMARKS)

        if processed and not ends_with_punctuation and not ends_with_closing:
            # Add a space if the last character isn't already whitespace
            if processed and not processed[-1].isspace():
                processed += " "
            processed += get_random_message('closing')

        return processed

    def run(self, query: str, chat_history: list = None) -> dict:
        """Runs the full QA pipeline with enhanced query handling, retrieval, and context expansion."""
        logger.info(f"\n🔄 Orchestrating response for query: '{query}'")

        # 1. Analyze Query
        query_analysis = self.query_analyzer.run(query=query)
        refined_query = query_analysis.get("refined_query", query)
        logger.debug(f"Query analysis result: {query_analysis}")

        # Handle complex queries with decomposition if needed
        if query_analysis.get("needs_decomposition", False) and len(query_analysis.get("sub_queries", [])) > 1:
            logger.info(f"🧩 Complex query detected, processing {len(query_analysis['sub_queries'])} sub-queries")
            # Future implementation: run sub-queries separately and combine results

        # 2. Retrieve Context with advanced retriever
        retrieved_chunks = self.retriever.run(
            query=refined_query,
            query_analysis=query_analysis
        )
        logger.debug(f"Retrieved {len(retrieved_chunks)} initial chunks.")

        # 3. Assess & Expand Context
        final_context_chunks, aggregated_metadata = self.context_expander.run(
            retrieved_chunks=retrieved_chunks,
            query_analysis=query_analysis,
            retriever_agent=self.retriever
        )
        logger.debug(f"Expanded/filtered context to {len(final_context_chunks)} chunks.")

        if not final_context_chunks:
            logger.warning("⚠️ No relevant context found after retrieval/expansion.")
            # Use random "not found" message via util function
            final_answer = get_random_message('not_found')
            return {
                "answer": final_answer,
                "references": {"pages": [], "sections": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 4. Generate Answer with query-type-aware prompting
        raw_answer = self.generator.run(
            query=query_analysis.get("original_query", query),
            context_chunks=final_context_chunks,
            query_analysis=query_analysis,
            chat_history=chat_history
        )
        logger.debug(f"Raw generated answer (first 100 chars): {raw_answer[:100]}")

        # 5. Post-process the answer using the updated local method
        final_answer = self._post_process_answer(raw_answer)

        # 6. Use the metadata from context expander for references
        references = aggregated_metadata
        logger.debug(f"Final references: {references}")

        # 7. Format final output
        logger.info("✅ Orchestration complete.")
        return {
            "answer": final_answer,
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
