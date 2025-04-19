# agents/generator.py
import re
import time
import logging
import enum  # Add enum import
import google.generativeai as genai
from .base import BaseAgent
from gemini_utils import setup_gemini
from utils.text_utils import post_process_answer
from config import Config

logger = logging.getLogger(__name__)

# --- Add Context Relevance Status Enum ---
class ContextRelevance(enum.Enum):
    RELEVANT = 1
    INSUFFICIENT_ACADEMIC = 2  # Context found, but not enough for the academic question
    IRRELEVANT = 3  # No relevant terms found at all

# --- Constants for Signaling ---
NEEDS_WEB_SEARCH_MARKER = "<<NEEDS_WEB_SEARCH>>"
IRRELEVANT_CONTEXT_MARKER = "<<IRRELEVANT_CONTEXT>>"
OFF_TOPIC_MARKER = "<<OFF_TOPIC>>"  # Added for clarity

# Revised Q&A pairs for a direct, factual persona
EXAMPLE_QA_PAIRS = """
Student: Why should I study history?
Tutor: Studying history helps understand how the past shapes the future [p. 3, Section: Introduction]. It provides context for current events and societal structures.

Student: Who was the first king of Sri Lanka?
Tutor: According to the provided text, King Vijaya was the first recorded ruler of Sri Lanka, arriving in 543 BCE from North India [p. 15, Section: Early Kingdoms]. He established the initial kingdom.

Student: What's the difference between primary and secondary sources?
Tutor: Primary sources are original materials from the time period being studied, such as diaries or artifacts [p. 7, Section: Historical Sources]. Secondary sources are analyses or interpretations created after the events, like the textbook itself [p. 7, Section: Historical Sources].

Student: Tell me more about the Kandyan Kingdom.
Tutor: The Kandyan Kingdom, located in the central hills, was the last independent kingdom in Sri Lanka before British rule [p. 88, Section: Kandyan Period]. It resisted European colonization for centuries.
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

    def _check_context_relevance(self, context_chunks: list[dict], query_analysis: dict) -> ContextRelevance:
        """Check context relevance based on query/topic terms."""
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        intent_type = query_analysis.get("intent_type", "new_topic")
        topic_keywords = query_analysis.get("topic_keywords", [])
        topic_entities = query_analysis.get("topic_entities", [])

        search_terms = set([k.lower() for k in keywords] + [e.lower() for e in entities])

        if intent_type in ["follow_up", "clarification"]:
            topic_terms = set([k.lower() for k in topic_keywords] + [e.lower() for e in topic_entities])
            search_terms.update(topic_terms)
            logger.debug(f"Checking context relevance (follow-up). Search terms (incl. topic): {search_terms}")
        else:
            logger.debug(f"Checking context relevance. Search terms: {search_terms}")

        if not search_terms and intent_type not in ["follow_up", "clarification"]:
            logger.warning("No specific search terms found in query analysis for a new topic.")
            query_type = query_analysis.get("query_type", "unknown")
            if query_type in ["factual", "causal/analytical", "comparative", "definition"]:
                return ContextRelevance.INSUFFICIENT_ACADEMIC
            else:
                return ContextRelevance.IRRELEVANT

        found_relevant_chunk = False
        for i, chunk in enumerate(context_chunks):
            text_lower = chunk.get("text", "").lower()
            for term in search_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    logger.debug(f"Found relevant term '{term}' in context chunk {i+1}.")
                    found_relevant_chunk = True
                    break
            if found_relevant_chunk:
                break

        if found_relevant_chunk:
            logger.debug("Found potentially relevant context chunk(s).")
            return ContextRelevance.RELEVANT
        else:
            logger.warning("No relevant query or topic terms found in any context chunk.")
            query_type = query_analysis.get("query_type", "unknown")
            if query_type in ["factual", "causal/analytical", "comparative", "definition"]:
                logger.warning("Marking as INSUFFICIENT_ACADEMIC due to lack of term match for academic query.")
                return ContextRelevance.INSUFFICIENT_ACADEMIC
            else:
                logger.warning("Marking as IRRELEVANT due to lack of term match for non-academic query.")
                return ContextRelevance.IRRELEVANT

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict, chat_history: list = None, web_results: list[dict] = None) -> str:
        """Create an effective prompt including optional web results."""
        query_type = query_analysis.get("query_type", "unknown")
        complexity = query_analysis.get("complexity", "simple")
        intent_type = query_analysis.get("intent_type", "new_topic")
        conversation_topic = query_analysis.get("conversation_topic")  # Can be None

        prompt = f"""You are Yuhasa, an AI History Tutor specializing *only* in the provided Sri Lankan Grade 11 History textbook content OR explicitly marked web research. Your goal is to answer student questions accurately and concisely based *strictly* on the **Context Information** and **Web Research** (if provided) below.

**Textbook Context Information:**
"""
        if context_chunks:
            for i, chunk in enumerate(context_chunks):
                page = chunk.get("metadata", {}).get("page", "?")
                section = chunk.get("metadata", {}).get("section", "Unknown Section")
                prompt += f"\n--- Context Chunk {i+1} (Page: {page}, Section: {section}) ---\n"
                prompt += chunk.get("text", "[No text found]") + "\n"
            prompt += "\n---\n"
        else:
            prompt += "[No relevant textbook context was found for this query.]\n---\n"

        if web_results:
            prompt += "\n**Web Research Information:**\n"
            prompt += "*Note: The following information is from external web sources and may supplement the textbook.*\n"
            for i, result in enumerate(web_results):
                url = result.get("url", "Unknown Source")
                snippet = result.get("snippet", "[No snippet available]")
                prompt += f"\n--- Web Result {i+1} (Source: {url}) ---\n"
                prompt += snippet + "\n"
            prompt += "\n---\n"

        if chat_history and len(chat_history) > 0:
            prompt += "\n**Recent Conversation History:**\n"
            history_limit = 4
            for msg in chat_history[-history_limit:]:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "\n---\n"

        prompt += f"\n**Your Task:** Answer the student's **Current Question** using the **Textbook Context Information** first. If that is insufficient or unavailable, use the **Web Research Information** (if provided). Consider the **Recent Conversation History** if relevant."

        if intent_type == "follow_up" and conversation_topic:
            prompt += f" The student's question seems to be a follow-up related to the recent topic of **'{conversation_topic}'**. Relate your answer to this topic if possible, based *only* on the provided context."
        elif intent_type == "clarification" and conversation_topic:
            prompt += f" The student is asking for clarification, likely related to the recent topic of **'{conversation_topic}'**. Provide more detail or rephrase information from the context relevant to both the question and the topic."
        elif intent_type == "topic_change":
            prompt += f" The student seems to be asking about a new topic."

        prompt += "\n\n**Current Question:**\n"
        prompt += query

        prompt += f"""

**Answering Rules:**
1.  Prioritize answers from the **Textbook Context Information**. Base answers *exclusively* on the provided information (Textbook or Web). Do not use external knowledge unless it's in the **Web Research Information**.
2.  If the textbook context directly answers the question, provide the answer clearly and concisely.
3.  Cite **every** factual statement, detail, date, or name from the **Textbook Context** using the format `[p. PageNumber, Section: SectionName]`.
4.  If using information *only* from **Web Research**, clearly state this at the beginning of the relevant sentence or paragraph (e.g., "From web research: ...") and cite the source URL at the end using `(Source: URL)`. If combining info, cite both appropriately.
5.  If the textbook context contains partial information and web research supplements it, synthesize the answer, clearly indicating which part comes from which source and citing accordingly.
6.  If **neither** the Textbook Context nor the Web Research Information (if provided) contains relevant information to answer the question, state clearly: "Based on the provided textbook context and web research, I cannot answer the question about [specific topic of the question]." Do not apologize or use filler phrases.
7.  Structure multi-part answers or lists using Markdown.
8.  Maintain a neutral, informative, and direct tone. Avoid emojis, apologies, or unnecessary conversational filler.
9.  **Strict Rule:** Never hedge if the context provides a direct fact. Never invent facts or information not present in the provided **Textbook Context** or **Web Research**.

**Example Q&A Pairs:**
{EXAMPLE_QA_PAIRS}
*Example using Web Research:*
Student: What is the current population of Sri Lanka?
Tutor: From web research: As of early 2025, the estimated population of Sri Lanka is around 22 million (Source: https://example.data.gov/population). The provided textbook does not contain current population figures.
"""
        prompt += "\n**Final Answer:**"

        return prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None, chat_history: list = None, web_results: list[dict] = None) -> str:
        """Generates an answer, potentially using web results, or signals special conditions."""
        run_start_time = time.time()
        logger.debug(f"✍️ Generating answer for query: '{query}' with {len(context_chunks)} context chunks and {len(web_results or [])} web results.")

        if not query_analysis:
            query_analysis = {"query_type": "unknown", "complexity": "simple", "keywords": [], "entities": [], "is_relevant_topic": True}

        if not web_results:
            if not query_analysis.get("is_relevant_topic", True):
                logger.warning(f"Query flagged as off-topic by analyzer: '{query}'")
                return OFF_TOPIC_MARKER

            if not context_chunks:
                logger.warning("⚠️ No context chunks provided.")
                query_type = query_analysis.get("query_type", "unknown")
                if query_type in ["factual", "causal/analytical", "comparative", "definition"]:
                    logger.info("No context chunks, academic query type. Signaling for web search.")
                    return NEEDS_WEB_SEARCH_MARKER
                else:
                    logger.info("No context chunks, non-academic query type. Signaling irrelevant context.")
                    return IRRELEVANT_CONTEXT_MARKER

            relevance_status = self._check_context_relevance(context_chunks, query_analysis)

            if relevance_status == ContextRelevance.INSUFFICIENT_ACADEMIC:
                logger.warning(f"⚠️ Context relevance check failed or deemed insufficient for academic query.")
                logger.info("Signaling for potential web search.")
                return NEEDS_WEB_SEARCH_MARKER

            elif relevance_status == ContextRelevance.IRRELEVANT:
                logger.warning(f"⚠️ Context relevance check failed (Irrelevant).")
                logger.info("Signaling irrelevant context.")
                return IRRELEVANT_CONTEXT_MARKER

        prompt = self.create_prompt(query, context_chunks, query_analysis, chat_history, web_results)

        try:
            logger.info("📞 Calling Gemini API...")
            gemini_start_time = time.time()
            response = self.gemini.generate_content(
                prompt,
                generation_config=self.config.GENERATION_CONFIG,
                safety_settings=self.config.SAFETY_SETTINGS
            )
            gemini_duration = time.time() - gemini_start_time
            logger.info(f"✅ Gemini API call successful. Duration: {gemini_duration:.4f}s")

            if not response.parts:
                logger.warning("⚠️ Gemini response has no parts (potentially blocked or empty).")
                if web_results:
                    return "The model could not generate a response based on the provided context and web research."
                else:
                    return "The model could not generate a response based on the provided context."

            if hasattr(response, 'text') and response.text:
                raw_answer = response.text
                logger.debug(f"Raw answer received: {raw_answer[:200]}...")
            else:
                logger.warning("⚠️ Gemini response does not contain text or is empty.")
                if web_results:
                    return "The model generated an empty response based on the provided context and web research."
                else:
                    return "The model generated an empty response based on the provided context."

            processing_start_time = time.time()
            processed_answer = post_process_answer(raw_answer)
            logger.debug(f"Processed answer: {processed_answer[:200]}...")

            final_answer = processed_answer

            processing_duration = time.time() - processing_start_time
            total_run_time = time.time() - run_start_time
            logger.info(f"⚙️ Answer processing finished. Duration: {processing_duration:.4f}s")
            logger.info(f"✅ Answer generated successfully. Total run time: {total_run_time:.4f}s.")
            if total_run_time > 5.0:
                logger.warning(f"⏱️ Total response time ({total_run_time:.4f}s) exceeded target of 5 seconds.")
            return final_answer

        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}", exc_info=True)
            error_message = "An error occurred while generating the response."
            return error_message

