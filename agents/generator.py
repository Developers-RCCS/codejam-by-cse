# agents/generator.py
import re
import time
import logging
import google.generativeai as genai
from .base import BaseAgent
from gemini_utils import setup_gemini
from utils.text_utils import post_process_answer, format_multi_part_answer
from config import Config
import spacy
from collections import defaultdict

logger = logging.getLogger(__name__)

# Revised Q&A pairs for a direct, factual persona
EXAMPLE_QA_PAIRS = """
Student: Why should I study history?
Tutor: Studying history helps understand how the past shapes the future [p. 3, Section: Introduction]. It provides context for current events and societal structures.

Student: Who was the first king of Sri Lanka?
Tutor: According to the provided text, King Vijaya was the first recorded ruler of Sri Lanka, arriving in 543 BCE from North India [p. 15, Section: Early Kingdoms]. He established the initial kingdom.

Student: What's the difference between primary and secondary sources?
Tutor: Primary sources are original materials from the time period being studied, such as diaries or artifacts [p. 7, Section: Historical Sources]. Secondary sources are analyses or interpretations created after the events, like the textbook itself [p. 7, Section: Historical Sources].
"""

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        logger.info("✨ Initializing Gemini model...")
        self.config = Config()
        try:
            self.gemini = setup_gemini()
            logger.info("✅ Gemini model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            self.gemini = None

        # Load spaCy model for NER and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error("❌ Error loading spaCy model 'en_core_web_sm'. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize feedback storage
        self.feedback = defaultdict(list)

    def _check_context_relevance(self, context_chunks: list[dict], query_analysis: dict) -> bool:
        """Check if any context chunk contains keywords or entities from the query.

        Args:
            context_chunks: A list of dictionaries, where each dictionary represents a context chunk
                            and should have a "text" key.
            query_analysis: A dictionary containing the analysis of the user query, expected to have
                            "keywords" and "entities" keys (lists of strings).

        Returns:
            True if any chunk contains a keyword or entity, False otherwise.
        """
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        search_terms = set([k.lower() for k in keywords] + [e.lower() for e in entities])
        logger.debug(f"Checking context relevance. Search terms: {search_terms}")

        if not search_terms:
            logger.debug("No keywords/entities found in query analysis, assuming context is relevant.")
            return True  # If no terms to check, assume relevance or let LLM decide

        found_relevant_chunk = False
        for i, chunk in enumerate(context_chunks):
            text_lower = chunk.get("text", "").lower()
            for term in search_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    logger.debug(f"Found relevant term '{term}' in context chunk {i+1}.")
                    found_relevant_chunk = True
                    break  # Found a relevant term in this chunk, move to next chunk if needed (though one is enough)
            if found_relevant_chunk:
                break  # Found a relevant chunk, no need to check further

        if not found_relevant_chunk:
            logger.warning("No relevant terms found in any context chunk.")

        return found_relevant_chunk

    def _enhanced_check_context_relevance(self, context_chunks: list[dict], query_analysis: dict) -> bool:
        """Enhanced check for context relevance using NER and dependency parsing."""
        if not self.nlp:
            logger.warning("spaCy model not loaded, falling back to basic relevance check.")
            return self._check_context_relevance(context_chunks, query_analysis)

        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        search_terms = set([k.lower() for k in keywords] + [e.lower() for e in entities])
        logger.debug(f"Enhanced checking context relevance. Search terms: {search_terms}")

        if not search_terms:
            logger.debug("No keywords/entities found in query analysis, assuming context is relevant.")
            return True  # If no terms to check, assume relevance or let LLM decide

        found_relevant_chunk = False
        for i, chunk in enumerate(context_chunks):
            text_lower = chunk.get("text", "").lower()
            doc = self.nlp(text_lower)
            chunk_entities = [ent.text.lower() for ent in doc.ents]
            chunk_keywords = [token.text.lower() for token in doc if token.dep_ in ("nsubj", "dobj", "pobj")]

            for term in search_terms:
                if term in chunk_entities or term in chunk_keywords:
                    logger.debug(f"Found relevant term '{term}' in context chunk {i+1} using NER/Dependency Parsing.")
                    found_relevant_chunk = True
                    break  # Found a relevant term in this chunk, move to next chunk if needed (though one is enough)
            if found_relevant_chunk:
                break  # Found a relevant chunk, no need to check further

        if not found_relevant_chunk:
            logger.warning("No relevant terms found in any context chunk using NER/Dependency Parsing.")

        return found_relevant_chunk

    def _update_feedback(self, query: str, context_chunks: list[dict], relevance: bool):
        """Update feedback loop with user interaction data."""
        self.feedback[query].append({
            "context_chunks": context_chunks,
            "relevance": relevance
        })
        logger.debug(f"Feedback updated for query: '{query}' with relevance: {relevance}")

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict, chat_history: list = None) -> str:
        """Create an effective prompt based on query type, context, and history."""
        if context_chunks and "confidence" in context_chunks[0]:
            sorted_chunks = sorted(context_chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        else:
            sorted_chunks = context_chunks

        formatted_context = ""
        for i, chunk in enumerate(sorted_chunks):
            page = chunk["metadata"].get("page", "Unknown")
            section = chunk["metadata"].get("section", "Unknown")
            formatted_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] ---\n"
            formatted_context += chunk["text"]
            formatted_context += "\n"
        logger.debug(f"Formatted {len(sorted_chunks)} context chunks for prompt.")

        history_str = ""
        if chat_history:
            history_str += "\n\n## Conversation History:\n"
            limited_history = chat_history[-self.config.MAX_HISTORY_MESSAGES:]
            for msg in limited_history:
                sender = msg.get("sender")
                content = msg.get("message", "")
                role = "Student" if sender == "user" else "Tutor"
                history_str += f"{role}: {content}\n"
            logger.debug(f"Added {len(limited_history)} messages from chat history to prompt.")

        query_type = query_analysis.get("query_type", "unknown")
        complexity = query_analysis.get("complexity", "simple")

        base_prompt = f"""**Context Information:**
{formatted_context}

**Conversation History:**
{history_str}
**Student's Current Question:**
{query}
"""

        common_instructions = f"""\
You are a factual history tutor bot. Your goal is to provide direct, detailed, and accurate answers based *only* on the provided **Context Information**.

**Instructions:**
1.  Read the **Context Information** and **Conversation History** carefully.
2.  Answer the **Student's Current Question** directly and precisely using **only** the provided **Context Information**. Synthesize information across multiple context excerpts if necessary to form a complete answer.
3.  Cite **every** factual statement, detail, date, or name using the format `[p. PageNumber, Section: SectionName]`. If multiple sources support a statement, cite them all.
4.  If the context contains partial or indirect information, synthesize the best possible answer, explicitly stating what is known and what is not, citing the supporting evidence. Do **not** apologize for missing information.
5.  If the **Context Information** genuinely contains **no relevant information** to answer any part of the question (after careful checking), state clearly: "The provided context does not contain information about [specific topic of the question]." Do not apologize or use filler phrases like "Unfortunately..." or "I couldn't find...".
6.  Structure multi-part answers or lists using Markdown (e.g., bullet points `*`, numbered lists `1.`).
7.  Maintain a neutral, informative, and direct tone. Do not use emojis, apologies ("sorry", "unfortunately"), or unnecessary conversational filler ("What else can I help you with?", "Is there anything else?").
8.  **Strict Rule:** Never hedge (e.g., "might be," "could be," "suggests") if the context provides a direct fact. Never invent facts or information not present in the **Context Information**.

**Example Q&A Pairs:**
{EXAMPLE_QA_PAIRS}
"""
        if query_type == "factual":
            specific_instructions = """\
- Focus on extracting specific facts, dates, names, and events directly relevant to the question. Ensure every fact is cited.
"""
        elif query_type == "causal/analytical":
            specific_instructions = """\
- Explain the 'why' or 'how' behind events or developments, using cited evidence from the context.
- Structure your analysis logically (e.g., cause-effect, sequence of events), citing each point.
"""
        elif query_type == "comparative":
            specific_instructions = """\
- Clearly identify the similarities and differences, citing the source for each point of comparison.
- Organize the comparison point-by-point using Markdown.
"""
        else:
            specific_instructions = """\
- Provide a clear and accurate explanation based strictly on the cited context.
"""

        if complexity == "complex":
            specific_instructions += """\
- This question may have multiple parts. Address all aspects thoroughly, citing each part.
- Structure your answer clearly using Markdown headings or lists.
"""

        complete_prompt = f"{base_prompt}\n**Tutor Persona & Instructions:**\n{common_instructions}{specific_instructions}\n**Tutor's Answer:**"
        logger.debug(f"Created prompt for query type '{query_type}' and complexity '{complexity}'. Prompt length: {len(complete_prompt)}")
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None, chat_history: list = None) -> str:
        """Generates an answer based on the query, context, analysis, and history."""
        run_start_time = time.time()
        logger.debug(f"✍️ Generating answer for query: '{query}' with {len(context_chunks)} context chunks.")

        if not query_analysis:
            logger.warning("⚠️ Query analysis missing, using default analysis.")
            query_analysis = {"query_type": "unknown", "complexity": "simple", "keywords": [], "entities": []}

        if not context_chunks:
            logger.warning("⚠️ No context chunks provided.")
            # Adhering to strict non-apology rule
            fallback_message = "The provided context does not contain information to answer this question."
            logger.info(f"Fallback triggered: No context. Time: {time.time() - run_start_time:.4f}s")
            return fallback_message

        is_relevant = self._enhanced_check_context_relevance(context_chunks, query_analysis)
        self._update_feedback(query, context_chunks, is_relevant)
        if not is_relevant:
            logger.warning(f"⚠️ Context relevance check failed. Keywords/Entities: {query_analysis.get('keywords', []) + query_analysis.get('entities', [])}")
            # Adhering to strict non-apology rule
            fallback_message = "The provided context does not contain information relevant to this question."
            logger.info(f"Fallback triggered: Low relevance. Time: {time.time() - run_start_time:.4f}s")
            return fallback_message

        prompt = self.create_prompt(query, context_chunks, query_analysis, chat_history)

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for factual recall
            )

            logger.info("📞 Calling Gemini API...")
            gemini_start_time = time.time()
            response = self.gemini.generate_content(
                prompt,
                generation_config=generation_config
            )
            gemini_duration = time.time() - gemini_start_time
            logger.info(f"✅ Gemini API call successful. Duration: {gemini_duration:.4f}s")

            # Handle potential safety blocks or empty responses
            if not response.parts:
                logger.warning("⚠️ Gemini response has no parts (potentially blocked or empty).")
                # Provide a neutral fallback consistent with instructions
                return "The model could not generate a response based on the provided context."
            # Assuming response.text is the primary way to get content
            # Check if response.text exists and is not empty
            if hasattr(response, 'text') and response.text:
                raw_answer = response.text
                logger.debug(f"Raw answer received: {raw_answer[:200]}...")  # Log beginning of raw answer
            else:
                logger.warning("⚠️ Gemini response does not contain text or is empty.")
                # Provide a neutral fallback consistent with instructions
                return "The model generated an empty response based on the provided context."

            processing_start_time = time.time()
            processed_answer = post_process_answer(raw_answer)
            logger.debug(f"Processed answer: {processed_answer[:200]}...")  # Log beginning of processed answer

            complexity = query_analysis.get("complexity", "simple")
            if complexity == "complex":
                final_answer = format_multi_part_answer(processed_answer, complexity)
                logger.debug("Formatted answer for complex query.")
            else:
                final_answer = processed_answer

            processing_duration = time.time() - processing_start_time
            total_run_time = time.time() - run_start_time
            logger.info(f"⚙️ Answer processing finished. Duration: {processing_duration:.4f}s")
            logger.info(f"✅ Answer generated successfully. Total run time: {total_run_time:.4f}s.")
            # Note: Benchmarking target of <2s depends heavily on external API latency.
            if total_run_time > 2.0:
                logger.warning(f"⏱️ Total response time ({total_run_time:.4f}s) exceeded target of 2 seconds.")
            return final_answer
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}", exc_info=True)
            # Provide a neutral error message
            error_message = "An error occurred while generating the response."
            return error_message

