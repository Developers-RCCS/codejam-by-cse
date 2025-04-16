# agents/generator.py
from .base import BaseAgent
from gemini_utils import setup_gemini

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        print("✨ Initializing Gemini model...")
        self.gemini = setup_gemini()
        print("✅ Gemini model initialized.")

    def run(self, query: str, context_chunks: list[dict]) -> str:
        """Generates an answer based on the query and context chunks."""
        print("✍️ Generating answer...")
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        pages = sorted(list(set([chunk["metadata"]["page"] for chunk in context_chunks]))) # Deduplicate and sort

        prompt = f"""**Context:**
{context_text}

**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Your goal is to answer the user's question accurately based *only* on the provided context above.
- Be concise and clear in your explanation.
- Cite the relevant page number(s) from the context like this: [p. XX].
- If the context doesn't contain the answer, state that clearly and do not guess.
- If multiple pages are relevant, cite them all (e.g., [p. 15, 18]).

**Question:** {query}

**Answer:**"""

        # print(f"\n--- PROMPT ---{prompt}\n--- END PROMPT ---\n") # For debugging

        try:
            response = self.gemini.generate_content(prompt)
            answer = response.text
            print("✅ Answer generated.")
            # Append references explicitly if not naturally included by the model
            # (This might be refined by the ReferenceTrackerAgent later)
            if pages and not any(f"[p. {p}]" in answer for p in pages):
                 answer += f"\n\n*References: [p. {', '.join(map(str, pages))}]*"

            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."

