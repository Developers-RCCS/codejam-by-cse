# agents/generator.py
import re
import time
import random
from .base import BaseAgent
from gemini_utils import setup_gemini

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        print("✨ Initializing Gemini model...")
        self.gemini = setup_gemini()
        print("✅ Gemini model initialized.")

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

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict) -> str:
        """Create an effective prompt based on query type and context."""
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

        # Get query type and complexity
        query_type = query_analysis.get("query_type", "unknown")
        complexity = query_analysis.get("complexity", "simple")

        # Base prompt structure
        base_prompt = f"""**Context Information:**
{formatted_context}

**Question:**
{query}
"""

        # --- Enhanced Instructions ---
        # Define the Yuhasa persona and instructions
        common_instructions = """\
You are Yuhasa, a friendly, knowledgeable, and slightly witty history tutor for Grade 11 Sri Lankan students. Your tone is engaging, helpful, and positive. You sometimes use light humor or a slightly flirty remark where appropriate, but always remain respectful and focused on providing accurate historical information.

**Instructions:**
- NEVER start your response with phrases like 'Unfortunately, the provided text...', 'Based on the provided text...', or similar apologetic/robotic phrases.
- Directly answer the question using **only** the provided **Context Information** above. Synthesize information if needed.
- If you use information from the context, seamlessly mention the source like 'According to page X...' or 'The textbook mentions on page Y that...'. Use the format [p. PageNumber] for citations, e.g., [p. 42] or [p. 15, 18]. Cite every piece of information used.
- If the context truly lacks the information needed, state that clearly and concisely (e.g., "Hmm, the textbook excerpts don't seem to cover that specific detail.").
- Always be encouraging and invite the user to ask more questions.
- Example Tone: "User: Why did the Kandyan Kingdom fall? Yuhasa: Ah, the fall of Kandy! A dramatic chapter... According to page 112, internal conflicts played a big role... What else about the Kandyan era sparks your interest? 😉"
"""

        # --- Specific instructions based on query analysis (Keep these as they tailor the answer structure) ---
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

        complete_prompt = f"{base_prompt}\n**Persona & Instructions:**\n{common_instructions}{specific_instructions}\n**Answer:**" # Changed "Instructions:" to "Persona & Instructions:" for clarity
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None) -> str:
        """Generates an answer based on the query, context chunks, and query analysis."""
        run_start_time = time.time()
        print("✍️ Generating answer...")

        if not query_analysis:
            # Fallback if analysis is missing (shouldn't happen in normal flow)
            print("  ⚠️ Query analysis missing, using default analysis.")
            query_analysis = {"query_type": "unknown", "complexity": "simple", "keywords": [], "entities": []}

        # --- Context Relevance Check ---
        if not context_chunks:
            print("  ⚠️ No context chunks provided.")
            # Consider if a different message is needed here vs. low relevance
            fallback_message = "I couldn't retrieve any relevant information from the textbook for your question. Could you try rephrasing it?"
            print(f"  Fallback triggered: No context. Time: {time.time() - run_start_time:.4f}s")
            return fallback_message + " Let me know if you have another question!"

        is_relevant = self._check_context_relevance(context_chunks, query_analysis)
        if not is_relevant:
            print(f"  ⚠️ Context relevance check failed. Keywords/Entities: {query_analysis.get('keywords', []) + query_analysis.get('entities', [])}")
            # Friendly fallback message
            fallback_message = "Hmm, I looked through the relevant parts of the textbook but couldn't find specific details matching your question. Perhaps try asking in a different way?"
            print(f"  Fallback triggered: Low relevance. Time: {time.time() - run_start_time:.4f}s")
            return fallback_message + " I'm here if you want to ask something else!"
        # --- End Context Relevance Check ---

        # Create appropriate prompt based on query type
        prompt = self.create_prompt(query, context_chunks, query_analysis)

        try:
            # Generate response
            print("  Calling Gemini API...")
            gemini_start_time = time.time()
            response = self.gemini.generate_content(prompt)
            raw_answer = response.text
            print(f"  Gemini response received in {time.time() - gemini_start_time:.4f}s.")

            # Post-process the answer - REMOVED
            final_answer = raw_answer.strip() # Just strip whitespace now

            total_run_time = time.time() - run_start_time
            print(f"✅ Answer generated in {total_run_time:.4f}s.") # Updated log message
            return final_answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            # Keep a friendly error message, but maybe slightly more in persona?
            return "Oops! Something went a bit sideways while I was thinking. Could you try asking that again? 😊"

