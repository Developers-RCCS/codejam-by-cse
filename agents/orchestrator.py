# agents/orchestrator.py
import re
import random
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent

# Define friendly "not found" messages (moved from web.py)
NOT_FOUND_MESSAGES = [
    "Ooh, that's a tricky one! My textbook doesn't seem to go into detail on that specific point. Maybe we could try phrasing it differently, or ask about something related? 😊",
    "Hmm, stumped me there! Looks like the textbook is a bit quiet on that particular topic. Got another historical mystery for me?",
    "Good question! I scanned my notes (aka the textbook!), but couldn't find the specifics on that. What else is on your mind?",
    "My apologies, but the provided textbook excerpts don't seem to cover that. Is there another angle we could explore?",
    "Interesting question! Unfortunately, the details aren't in the sections I have access to. Perhaps we can focus on a related event mentioned in the book?"
]

# Define friendly closing remarks for post-processing (moved from web.py)
CLOSING_REMARKS = [
    "Hope that helps! Ask me another!",
    "Anything else you're curious about?",
    "Happy to help! What's next on your mind? 😉",
    "Let me know if you have more questions!",
    "Was there anything else I can help you with today?"
]

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

    def _post_process_answer(self, raw_answer: str) -> str:
        """Applies final polishing touches to the generated answer (moved from web.py)."""
        processed = raw_answer

        # Remove common boilerplate leading phrases (case-insensitive)
        processed = re.sub(r"^based on the context provided,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
        processed = re.sub(r"^according to the text,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
        processed = re.sub(r"^the provided context states that,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Remove common boilerplate closing phrases (case-insensitive)
        processed = re.sub(r"in conclusion,?$\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
        processed = re.sub(r"to summarize,?$\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Trim leading/trailing whitespace again after potential removals
        processed = processed.strip()

        # Add a friendly closing remark if missing
        ends_with_punctuation = processed.endswith(('.', '!', '?', ';', ')'))
        ends_with_closing = any(processed.lower().endswith(remark.lower()) for remark in CLOSING_REMARKS)

        if processed and not ends_with_punctuation and not ends_with_closing:
            # Add a space if the last character isn't already whitespace
            if processed and not processed[-1].isspace():
                processed += " "
            processed += random.choice(CLOSING_REMARKS)

        return processed

    def run(self, query: str, chat_history: list = None) -> dict:
        """Runs the full QA pipeline with enhanced query handling, retrieval, and context expansion."""
        print(f"\n🔄 Orchestrating response for query: '{query}'")

        # 1. Analyze Query
        query_analysis = self.query_analyzer.run(query=query)
        refined_query = query_analysis.get("refined_query", query)

        # Handle complex queries with decomposition if needed
        if query_analysis.get("needs_decomposition", False) and len(query_analysis.get("sub_queries", [])) > 1:
            print(f"🧩 Complex query detected, processing {len(query_analysis['sub_queries'])} sub-queries")
            # Future implementation: run sub-queries separately and combine results

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
            # Use random "not found" message
            final_answer = random.choice(NOT_FOUND_MESSAGES)
            return {
                "answer": final_answer,
                "references": {"pages": [], "sections": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 4. Generate Answer with query-type-aware prompting
        raw_answer = self.generator.run(
            query=query_analysis.get("original_query", query), # Use .get() for safety
            context_chunks=final_context_chunks,
            query_analysis=query_analysis,
            chat_history=chat_history # Pass history to generator
        )

        # 5. Post-process the answer
        final_answer = self._post_process_answer(raw_answer) # Apply post-processing

        # 6. Use the metadata from context expander for references
        references = aggregated_metadata

        # 7. Format final output
        print("✅ Orchestration complete.")
        return {
            "answer": final_answer, # Return post-processed answer
            "references": references,
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks
        }
