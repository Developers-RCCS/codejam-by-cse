# agents/orchestrator.py
import logging
import time
import random  # Import random for selecting responses
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
from .context_expander import ContextExpansionAgent

logger = logging.getLogger(__name__)

# --- Greeting/General Chat Handling ---
COMMON_GREETINGS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo", "sup"
}
GENERAL_CHAT = {
    "thanks", "thank you", "ok", "okay", "cool", "awesome", "great", "bye", "goodbye", "see you", "how are you", "how's it going", "what's up"
}
FRIENDLY_RESPONSES = [
    "Hi there! 😊 How can I help you explore Sri Lankan history today?",
    "Hello! Ready to dive into some history? Ask me anything about the textbook.",
    "Hey! What aspect of Sri Lankan history are you interested in learning about?",
    "Greetings! I'm here to help with your questions about the Grade 11 history text.",
    "Thanks for stopping by! What historical topic is on your mind?",
    "You're welcome! Anything else I can help you find in the history text?",
    "Okay! Let me know your next question about Sri Lankan history.",
    "Glad I could help! Feel free to ask more questions.",
    "Goodbye! Come back anytime to learn more history.",
    "See you later! Happy studying!",
    "I'm doing well, thank you! Ready to assist with your history questions. What would you like to know?",
]
ACKNOWLEDGEMENT_RESPONSES = [
    "You're welcome!",
    "No problem!",
    "Glad I could help!",
    "Anytime!",
]
FAREWELL_RESPONSES = [
    "Goodbye!",
    "See you later!",
    "Take care!",
    "Happy studying!",
]
HOW_ARE_YOU_RESPONSES = [
    "I'm doing well, thank you for asking! I'm ready to help you with Sri Lankan history. What's your question?",
    "I'm an AI, so I don't have feelings, but I'm fully operational and ready to assist you with history!",
    "Functioning optimally! How can I help you with the history textbook today?",
]
# --- End Greeting/General Chat Handling ---


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
        """Runs the full QA pipeline or handles greetings/general chat.

        Args:
            query: The user's input query.
            chat_history: A list of previous conversation messages.

        Returns:
            A dictionary containing the final answer, references, query analysis,
            and retrieved context chunks.
        """
        orchestration_start_time = time.time()
        logger.info(f"\n🔄 Orchestrating response for query: '{query}'")

        # --- Check for Greetings/General Chat ---
        normalized_query = query.lower().strip().rstrip('?.!')
        if normalized_query in COMMON_GREETINGS:
            response = random.choice([r for r in FRIENDLY_RESPONSES if "Hi" in r or "Hello" in r or "Hey" in r or "Greetings" in r])
            logger.info(f"💬 Detected greeting. Responding: '{response}'")
            return {"answer": response, "references": [], "query_analysis": {"type": "greeting"}, "retrieved_chunks": []}
        elif normalized_query in {"thanks", "thank you"}:
            response = random.choice(ACKNOWLEDGEMENT_RESPONSES)
            logger.info(f"💬 Detected thanks. Responding: '{response}'")
            return {"answer": response, "references": [], "query_analysis": {"type": "acknowledgement"}, "retrieved_chunks": []}
        elif normalized_query in {"bye", "goodbye", "see you"}:
            response = random.choice(FAREWELL_RESPONSES)
            logger.info(f"💬 Detected farewell. Responding: '{response}'")
            return {"answer": response, "references": [], "query_analysis": {"type": "farewell"}, "retrieved_chunks": []}
        elif normalized_query in {"how are you", "how's it going", "what's up"}:
            response = random.choice(HOW_ARE_YOU_RESPONSES)
            logger.info(f"💬 Detected 'how are you'. Responding: '{response}'")
            return {"answer": response, "references": [], "query_analysis": {"type": "status_inquiry"}, "retrieved_chunks": []}
        elif normalized_query in GENERAL_CHAT:  # Catch other general phrases
            response = random.choice([r for r in FRIENDLY_RESPONSES if "Okay" in r or "Glad" in r])  # Generic positive response
            logger.info(f"💬 Detected general chat. Responding: '{response}'")
            return {"answer": response, "references": [], "query_analysis": {"type": "general_chat"}, "retrieved_chunks": []}
        # --- End Check ---

        # 1. Analyze Query
        analysis_start_time = time.time()
        logger.info("Step 1: Analyzing query...")
        query_analysis = self.query_analyzer.run(query=query, chat_history=chat_history)  # Pass chat_history
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
