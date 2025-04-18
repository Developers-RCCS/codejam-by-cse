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

    def _postprocess_answer(self, raw_answer: str, context_chunks: list[dict]) -> str:
        """Clean up the raw answer, make it more conversational, and add references."""
        print("✨ Post-processing generated answer...")
        processed_answer = raw_answer

        # Remove common robotic/negative phrases
        processed_answer = re.sub(r"Based on the provided context,? ", "", processed_answer, flags=re.IGNORECASE)
        processed_answer = re.sub(r"The provided context does not contain information about", "The textbook excerpts don't seem to cover", processed_answer, flags=re.IGNORECASE)
        processed_answer = re.sub(r"The context does not mention", "I couldn't find information on", processed_answer, flags=re.IGNORECASE)
        processed_answer = re.sub(r"I cannot answer this question based on the context provided.", "I couldn't find the answer to that in the provided excerpts.", processed_answer, flags=re.IGNORECASE)
        processed_answer = re.sub(r"Sorry, I cannot", "Unfortunately, I cannot", processed_answer, flags=re.IGNORECASE)  # Softer apology

        # Ensure page references are included if not present
        pages = sorted(list(set([chunk["metadata"].get("page", "") for chunk in context_chunks])))
        pages = [p for p in pages if p]  # Filter out empty values

        # Check if any page references are already in the answer (flexible format check)
        has_page_refs = re.search(r'\[pP]\s?\.?\s?\d+\]', processed_answer) is not None

        if pages and not has_page_refs:
            page_refs_str = f"[p. {', '.join(map(str, pages))}]"
            # Try to append references naturally
            if "excerpts don't seem to cover" in processed_answer or "couldn't find information on" in processed_answer:
                processed_answer += f" (checked {page_refs_str})."  # Append to negative statements
            else:
                processed_answer += f"\n\n*Source: {page_refs_str}*"  # Append as source otherwise

        # Add friendly closing
        closing_phrases = [
            "Hope that helps!",
            "Let me know if you have more questions!",
            "Happy to help further if you need!",
            "Was there anything else I can help you with?"
        ]
        # Avoid adding closing if the answer already ends similarly or is very short/negative
        if not any(phrase in processed_answer[-50:] for phrase in ["?", "!", "."]) or len(processed_answer) < 50:
            processed_answer += f" {random.choice(closing_phrases)}"  # Add random closing

        # Trim potential leading/trailing whitespace
        processed_answer = processed_answer.strip()

        print("✅ Post-processing complete.")
        return processed_answer

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
        common_instructions = """
You are Yuhasa, a friendly, knowledgeable, and engaging AI tutor specializing in Grade 11 History. Your goal is to help the student understand the material.
- Answer the question directly using **only** the provided **Context Information** above.
- **Synthesize** information from the context to provide a clear and comprehensive answer. Do not just summarize excerpts.
- **Cite the specific page number(s)** from the context in square brackets like this: [p. 42] or [p. 15, 18] for every piece of information you use.
- **Do not apologize** or use phrases like "Based on the context...", "The provided text...", or "I cannot answer...".
- If the context truly lacks the information needed, state that clearly and concisely (e.g., "The provided excerpts don't cover that specific detail.").
- Answer in a natural, conversational, and helpful tone. Be encouraging!
"""

        if query_type == "factual":
            specific_instructions = """
- Focus on extracting specific facts, dates, names, and events directly relevant to the question.
- Keep the answer concise and to the point.
"""
        elif query_type == "causal/analytical":
            specific_instructions = """
- Explain the 'why' or 'how' behind events or developments, using evidence from the context.
- Structure your analysis logically (e.g., cause-effect, sequence of events).
"""
        elif query_type == "comparative":
            specific_instructions = """
- Clearly identify the similarities and differences between the items being compared.
- Organize the comparison point-by-point.
"""
        else:  # Default/unknown
            specific_instructions = """
- Provide a clear and accurate explanation based on the context.
"""

        if complexity == "complex":
            specific_instructions += """
- This question may have multiple parts. Ensure you address all aspects thoroughly.
- Structure your answer clearly, perhaps using bullet points if helpful.
"""
        # --- End Enhanced Instructions ---

        complete_prompt = f"{base_prompt}\n**Instructions:**\n{common_instructions}{specific_instructions}\n**Answer:**"
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

            # Post-process the answer
            final_answer = self._postprocess_answer(raw_answer, context_chunks)

            total_run_time = time.time() - run_start_time
            print(f"✅ Answer generated and processed in {total_run_time:.4f}s.")
            return final_answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "Sorry, I encountered a technical hiccup while trying to answer. Could you try asking again?"

