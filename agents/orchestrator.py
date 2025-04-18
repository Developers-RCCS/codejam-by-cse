# agents/orchestrator.py
import logging
import re
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent
from utils.messages import get_random_message, CLOSING_REMARKS, PLAYFUL_FOLLOWUPS
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
        # First apply basic text processing
        processed = post_process_answer(raw_answer)
        
        # Check if the answer already has a closing remark
        ends_with_closing = False
        for remark in CLOSING_REMARKS:
            if processed.lower().endswith(remark.lower()):
                ends_with_closing = True
                break
        
        # If no closing remark, add a supportive check-in or playful follow-up
        if processed and not ends_with_closing:
            # Add a space if the last character isn't already whitespace
            if processed[-1] not in [' ', '\n', '\t', '.', '?', '!']:
                processed += ". "
            elif processed[-1] not in [' ', '\n', '\t']:
                processed += " "
                
            # For multi-part answers (paragraphs), ensure we have nice short, friendly paragraphs
            if len(processed.split('\n\n')) > 1 or len(processed) > 300:
                # Add a playful follow-up for more complex answers
                processed += get_random_message('followup')
            else:
                # Add a supportive closing for simpler answers
                processed += get_random_message('supportive_closing')

        return processed

    def _enhance_not_found_response(self, response: str) -> str:
        """Adds a gentle tease to not-found responses."""
        # Add a playful tease if it doesn't already have one
        teases = [
            "You're really putting my history brain to the test today! 😉", 
            "Trying to stump me? You've found a good one!", 
            "Clever question! You've got me on a historical treasure hunt!"
        ]
        
        # Check if response already has one of our teases
        has_tease = any(tease.lower() in response.lower() for tease in teases)
        
        if not has_tease:
            tease = teases[0]  # Default to first tease
            # Insert tease before any closing remark if present
            for closing in CLOSING_REMARKS:
                if closing in response:
                    return response.replace(closing, f"{tease} {closing}")
            
            # Otherwise just append the tease
            if response[-1] not in [' ', '\n', '\t', '.', '?', '!']:
                response += ". "
            elif response[-1] not in [' ', '\n', '\t']:
                response += " "
                
            return f"{response}{tease}"
        
        return response

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
            # Use random "not found" message via util function with a gentle tease
            final_answer = self._enhance_not_found_response(get_random_message('not_found'))
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
