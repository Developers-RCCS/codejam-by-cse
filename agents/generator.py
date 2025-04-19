# agents/generator.py
import logging
import time
import google.generativeai as genai
from .base import BaseAgent
from config import Config

logger = logging.getLogger(__name__)

genai.configure(api_key=Config.GEMINI_API_KEY)

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini based on provided context."""
    def __init__(self):
        init_start_time = time.time()
        logger.info("Initializing Generator Agent...")
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        logger.info(f"Generator Agent initialized with model. Duration: {time.time() - init_start_time:.2f}s")

    def create_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """Creates a direct prompt for Gemini using retrieved context."""
        prompt_start_time = time.time()
        formatted_context = ""
        # Assuming lower score (distance) from FAISS L2 is better
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get("score", float('inf')))

        for i, chunk in enumerate(sorted_chunks):
            page = chunk["metadata"].get("page", "Unknown")
            section = chunk["metadata"].get("section", "Unknown")
            formatted_context += f"--- Context {i+1} [Page {page}, Section: {section}] ---\n"
            formatted_context += chunk["text"]
            formatted_context += "\n\n"

        instructions = (
            "Instructions:\n"
            "1. Answer the following **Question** using *only* the provided **Context** information above.\n"
            "2. Cite the source for *every* piece of information you use in the format [p. PageNumber, Section: SectionName]. Cite all relevant sources if multiple apply.\n"
            "3. If the context does not contain the answer, state clearly: \"The provided context does not contain information about [topic of the question].\" Do not apologize or add filler phrases.\n"
            "4. Be direct and factual. Do not add information not present in the context. Use Markdown for lists if needed."
        )

        complete_prompt = f"**Context:**\n{formatted_context.strip()}\n\n**Question:**\n{query}\n\n{instructions}\n\n**Answer:**"

        logger.debug(f"Created direct prompt. Length: {len(complete_prompt)}. Took: {time.time() - prompt_start_time:.4f}s")
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict]) -> str:
        """Generates an answer based *only* on the query and context chunks."""
        run_start_time = time.time()
        logger.info(f"Generating answer for query: '{query}'")

        if not context_chunks:
            logger.warning("No context chunks provided to GeneratorAgent. Returning fallback.")
            return "The provided context does not contain information about this question."

        prompt = self.create_prompt(query, context_chunks)

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.1, # Low temperature for factual, context-based answers
            )

            logger.info("Calling Gemini API...")
            gemini_start_time = time.time()
            response = self.gemini.generate_content(
                prompt,
                generation_config=generation_config
            )
            gemini_duration = time.time() - gemini_start_time
            logger.info(f"Gemini API call successful. Duration: {gemini_duration:.4f}s")

            if response.parts:
                answer = response.text
                logger.debug(f"Raw Gemini Response Text:\n{answer}")
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Gemini response blocked. Reason: {response.prompt_feedback.block_reason}")
                answer = f"The model could not generate a response due to safety settings (Reason: {response.prompt_feedback.block_reason})."
            else:
                logger.warning("Gemini response was empty or had no parts.")
                answer = "The model could not generate a response based on the provided context."

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            answer = "An error occurred while generating the answer."

        total_run_time = time.time() - run_start_time
        logger.info(f"GeneratorAgent finished in {total_run_time:.4f}s.")
        return answer

