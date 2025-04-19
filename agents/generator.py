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
        """Initializes the Generator Agent with the Gemini model."""
        init_start_time = time.time()
        logger.info("Initializing Generator Agent...")
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        logger.info(f"Generator Agent initialized with model. Duration: {time.time() - init_start_time:.2f}s")

    def create_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """
        Creates a direct prompt for Gemini using retrieved context chunks.

        Formats the context with clear separators and metadata (page, section)
        and provides specific instructions for the model.

        Args:
            query (str): The user's question.
            context_chunks (list[dict]): A list of retrieved context chunks,
                                         expected to have 'text' and 'metadata'.

        Returns:
            str: The fully formatted prompt string for the Gemini model.
        """
        prompt_start_time = time.time()
        formatted_context = ""
        # Sort chunks by score (assuming lower distance is better)
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get("score", float('inf')))

        if not sorted_chunks:
            formatted_context = "No context provided."
        else:
            for i, chunk in enumerate(sorted_chunks):
                page = chunk.get("metadata", {}).get("page", "Unknown")
                # Use the simplified section name (first line)
                section = chunk.get("metadata", {}).get("section", "Unknown Section")
                formatted_context += f"--- [Page {page}, Section: {section}] ---\n"
                formatted_context += chunk.get("text", "")
                formatted_context += "\n\n"

        # Clear instructions for Gemini
        instructions = (
            "**Instructions:**\n"
            "1. Answer the following **Question** using *only* the provided **Context** information above.\n"
            "2. For *every* factual statement in your answer, you **must** cite the source using the format `[Page X, Section: Y]`. Cite all relevant sources if multiple apply.\n"
            "3. If the provided **Context** does not contain the answer to the **Question**, state *only*: \"The provided context does not contain information to answer this question.\" Do not add any explanation or apology.\n"
            "4. Answer directly and concisely. Do not add information not present in the context.\n"
            "5. If the question has multiple parts, use bullet points or numbered lists in your answer."
        )

        complete_prompt = f"**Context:**\n{formatted_context.strip()}\n\n**Question:**\n{query}\n\n{instructions}\n\n**Answer:**"

        logger.debug(f"Created direct prompt. Length: {len(complete_prompt)}. Took: {time.time() - prompt_start_time:.4f}s")
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generates an answer using the Gemini model based *only* on the query and context chunks.

        Args:
            query (str): The user's question.
            context_chunks (list[dict]): The context chunks retrieved by the RetrieverAgent.

        Returns:
            str: The generated answer from the Gemini model, or an error/fallback message.
        """
        run_start_time = time.time()
        logger.info(f"Generating answer for query: '{query}'")

        if not context_chunks:
            logger.warning("No context chunks provided to GeneratorAgent. Returning fallback.")
            return "The provided context does not contain information to answer this question."

        prompt = self.create_prompt(query, context_chunks)

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.0, # Set to 0.0 for maximum factuality and adherence to context
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
                # Provide a more user-friendly message
                answer = f"I cannot provide an answer due to content restrictions (Reason: {response.prompt_feedback.block_reason})."
            else:
                logger.warning("Gemini response was empty or had no parts.")
                answer = "The model could not generate a response based on the provided context."

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            answer = "An error occurred while generating the answer."

        total_run_time = time.time() - run_start_time
        logger.info(f"GeneratorAgent finished in {total_run_time:.4f}s.")
        return answer

