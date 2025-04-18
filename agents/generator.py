# agents/generator.py
import re
import time
import random
import logging  # Added import
import google.generativeai as genai
from .base import BaseAgent
from gemini_utils import setup_gemini
from utils.messages import get_random_message, NOT_FOUND_MESSAGES, CLOSING_REMARKS, PLAYFUL_FOLLOWUPS
from utils.text_utils import post_process_answer, format_multi_part_answer

logger = logging.getLogger(__name__)  # Get a logger for this module

# Example Q&A pairs for the Yuhasa persona
EXAMPLE_QA_PAIRS = """
Student: Why should I study history?
Yuhasa: Because time travel is expensive, but a good question is free! 😉 History is like a treasure hunt—let's see what we can discover together! The textbook [p. 3] explains how knowing our past helps shape our future. What else can I help you explore today?

Student: Who was the first king of Sri Lanka?
Yuhasa: Ah, diving into royal affairs, are we? 📚 According to our textbook [p. 15], King Vijaya was the first recorded ruler of Sri Lanka in 543 BCE. He arrived from North India and established the kingdom that would evolve into modern Sri Lanka. Pretty impressive origin story, right? Let's see what other secrets history is hiding!

Student: What's the difference between primary and secondary sources?
Yuhasa: Ooh, clever question! Primary sources are the historical "selfies" – original documents from the time period like diaries or artifacts. Secondary sources are more like the history textbook [p. 7] we're using – analysis written after the fact. One gives you raw history, the other gives you the juicy interpretations! History is always more fun with you asking the questions! 😊
"""

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        logger.info("✨ Initializing Gemini model...")
        try:
            self.gemini = setup_gemini()
            logger.info("✅ Gemini model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            self.gemini = None

    def _check_context_relevance(self, context_chunks: list[dict], query_analysis: dict) -> bool:
        """Check if any context chunk contains keywords or entities from the query."""
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        search_terms = set([k.lower() for k in keywords] + [e.lower() for e in entities])

        if not search_terms:
            return True  # No specific terms to check against, assume relevance

        for chunk in context_chunks:
            text_lower = chunk.get("text", "").lower()
            for term in search_terms:
                # Use word boundaries for better matching, especially for shorter terms
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    return True  # Found at least one relevant term in one chunk
        return False  # No relevant terms found in any chunk

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict, chat_history: list = None) -> str:
        """Create an effective prompt based on query type, context, and history."""
        # Extract context text - if chunks have a confidence score, sort by it
        if context_chunks and "confidence" in context_chunks[0]:
            sorted_chunks = sorted(context_chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        else:
            sorted_chunks = context_chunks

        # Format context with metadata for better traceability
        formatted_context = ""
        for i, chunk in enumerate(sorted_chunks):
            page = chunk["metadata"].get("page", "Unknown")
            section = chunk["metadata"].get("section", "Unknown")
            formatted_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] ---\n"
            formatted_context += chunk["text"]
            formatted_context += "\n"

        # 🧠 Construct real-time dialogue memory (from web.py's reasoning_agent)
        conversation = ""
        if chat_history:
            trimmed = chat_history[-10:]  # Keep last 5 turns (10 messages)
            for msg in trimmed:
                role = "Student" if msg["sender"] == "user" else "Yuhasa"
                conversation += f"{role}: {msg['message']}\n"
        if not conversation:
            conversation = "(No recent conversation history)"
        # --- End dialogue memory ---

        # Get query type and complexity
        query_type = query_analysis.get("query_type", "unknown")
        complexity = query_analysis.get("complexity", "simple")

        # Base prompt structure - Incorporating history
        base_prompt = f"""**Recent Conversation:**
{conversation}
**Context Information:**
{formatted_context}

**Student's Current Question:**
{query}
"""

        # --- Enhanced Instructions (incorporating from web.py's reasoning_agent) ---
        common_instructions = f"""\
You are Yuhasa, a caring, supportive, and gently playful history tutor for Grade 11 students. You make learning feel like a fun adventure and create a comfortable environment for exploration and questions.

Use light humor, clever encouragement, and friendly emojis like 😊, 😄, 😉, 📚 to keep the conversation engaging. Never use kissing or romantic emojis.

Compliment student curiosity and make history feel like an exciting journey with a favorite teacher.

**Instructions:**
1. Carefully read the **Context Information** and **Recent Conversation**.
2. Answer the **Student's Current Question** using **only** the provided **Context Information**. 
3. Stay consistent with the **Recent Conversation**.
4. Cite sources using format [p. PageNumber], e.g., [p. 42] or [p. 15, 18].
5. If information isn't in context, kindly say so without robotic phrases.
6. Break long paragraphs into shorter, readable ones.
7. End with a positive invitation like "What else can I help you explore today?"
8. NEVER invent facts outside the provided context.
9. Use Markdown formatting where helpful.

**Example Q&A Pairs:**
{EXAMPLE_QA_PAIRS}
"""
        if query_type == "factual":
            specific_instructions = """\
- Focus on extracting specific facts, dates, names, and events directly relevant to the question.
- Keep the answer concise and to the point.
"""
        elif query_type == "causal/analytical":
            specific_instructions = """\
- Explain the 'why' or 'how' behind events or developments, using evidence from the context.
- Structure your analysis logically (e.g., cause-effect, sequence of events).
"""
        elif query_type == "comparative":
            specific_instructions = """\
- Clearly identify the similarities and differences between the items being compared.
- Organize the comparison point-by-point.
"""
        else:  # Default/unknown
            specific_instructions = """\
- Provide a clear and accurate explanation based on the context.
"""

        if complexity == "complex":
            specific_instructions += """\
- This question may have multiple parts. Ensure you address all aspects thoroughly.
- Structure your answer clearly, perhaps using bullet points if helpful.
"""
        # --- End Enhanced Instructions ---

        complete_prompt = f"{base_prompt}\n**Persona & Instructions:**\n{common_instructions}{specific_instructions}\n**Yuhasa's Answer:**"
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None, chat_history: list = None) -> str:
        """Generates an answer based on the query, context, analysis, and history."""
        run_start_time = time.time()
        logger.debug(f"✍️ Generating answer for query: '{query}' with {len(context_chunks)} context chunks.")

        if not query_analysis:
            # Fallback if analysis is missing (shouldn't happen in normal flow)
            logger.warning("⚠️ Query analysis missing, using default analysis.")
            query_analysis = {"query_type": "unknown", "complexity": "simple", "keywords": [], "entities": []}

        # --- Context Relevance Check ---
        if not context_chunks:
            logger.warning("⚠️ No context chunks provided.")
            fallback_message = random.choice(NOT_FOUND_MESSAGES)
            logger.info(f"Fallback triggered: No context. Time: {time.time() - run_start_time:.4f}s")
            # Append a random closing phrase even to fallback messages
            return fallback_message + " " + random.choice(CLOSING_REMARKS)

        is_relevant = self._check_context_relevance(context_chunks, query_analysis)
        if not is_relevant:
            logger.warning(f"⚠️ Context relevance check failed. Keywords/Entities: {query_analysis.get('keywords', []) + query_analysis.get('entities', [])}")
            fallback_message = random.choice(NOT_FOUND_MESSAGES)
            logger.info(f"Fallback triggered: Low relevance. Time: {time.time() - run_start_time:.4f}s")
            # Append a random closing phrase even to fallback messages
            return fallback_message + " " + random.choice(CLOSING_REMARKS)
        # --- End Context Relevance Check ---

        # Create appropriate prompt based on query type and history
        prompt = self.create_prompt(query, context_chunks, query_analysis, chat_history)

        try:
            # --- Add Generation Config (from web.py) ---
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,  # Lower temperature for more focused responses
                # Add other parameters like top_p, top_k if needed
            )
            # ---------------------------

            # Generate response with config
            logger.info("Calling Gemini API...")
            gemini_start_time = time.time()
            response = self.gemini.generate_content(
                prompt,
                generation_config=generation_config  # Pass the config here
            )
            raw_answer = response.text
            
            # Apply post-processing to remove robotic language and make more natural
            processed_answer = post_process_answer(raw_answer)
            
            # For complex questions, format the answer with better paragraph structure
            complexity = query_analysis.get("complexity", "simple")
            if complexity == "complex":
                final_answer = format_multi_part_answer(processed_answer, complexity)
            else:
                final_answer = processed_answer

            # Append a random closing phrase if it doesn't already have one
            for closing in CLOSING_REMARKS:
                if final_answer.lower().endswith(closing.lower()):
                    break
            else:  # No closing remark found
                closing_phrase = random.choice(CLOSING_REMARKS)
                # Ensure there's a space before appending if the answer doesn't end with punctuation
                if final_answer and final_answer[-1] not in ['.', '!', '?', ' ']:
                    final_answer += "."
                    
                if not final_answer.endswith(" "):
                    final_answer += " "
                    
                final_answer += closing_phrase

            total_run_time = time.time() - run_start_time
            logger.info(f"✅ Answer generated in {total_run_time:.4f}s.")
            return final_answer
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}", exc_info=True)
            # Keep a friendly error message, but maybe slightly more in persona?
            error_message = "Oops! Something went a bit sideways while I was thinking. Could you try asking that again? 😊"
            # Append a random closing phrase even to error messages
            return error_message + " " + random.choice(CLOSING_REMARKS)

