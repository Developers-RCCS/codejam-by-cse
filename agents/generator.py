# agents/generator.py
from .base import BaseAgent
from gemini_utils import setup_gemini

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        print("✨ Initializing Gemini model...")
        self.gemini = setup_gemini()
        print("✅ Gemini model initialized.")

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

        # Customize instructions based on query type
        if query_type == "factual":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this factual question accurately based on the provided context above.
- Provide a direct and concise answer with specific facts from the context.
- Cite page numbers in square brackets like this: [p. 42].
- If the answer isn't in the context, clearly state that you don't have that information.
- Answer only what is asked, avoid unnecessary elaboration.
"""
        elif query_type == "causal/analytical":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this analytical question based on the provided context above.
- Provide a structured analysis that explains causes, effects, or developments.
- Organize your response with clear reasoning that connects evidence to conclusions.
- Cite page numbers in square brackets like this: [p. 42].
- Focus on explaining "why" or "how" rather than just listing facts.
- If the context doesn't provide sufficient information, acknowledge limitations in your analysis.
"""
        elif query_type == "comparative":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this comparative question based on the provided context above.
- Structure your answer to clearly compare and contrast the items.
- Organize by points of comparison rather than describing each item separately.
- Present similarities and differences in a balanced way.
- Cite page numbers in square brackets like this: [p. 42].
- If the context lacks information about one of the comparison items, acknowledge this limitation.
"""
        else:  # Default/unknown
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer the question accurately based on the provided context above.
- Be concise and clear in your explanation.
- Cite the relevant page numbers from the context like this: [p. 42].
- If the context doesn't contain the answer, state that clearly and do not guess.
- If multiple pages are relevant, cite them all (e.g., [p. 15, 18]).
"""

        # Add complexity handling
        if complexity == "complex":
            instructions += """
- This is a complex multi-part question. Make sure to address all aspects.
- Structure your answer with clear sections for each part of the question.
- Ensure you're drawing connections between different parts where relevant.
"""

        # Finalize prompt
        complete_prompt = f"{base_prompt}\n{instructions}\n**Answer:**"
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None) -> str:
        """Generates an answer based on the query, context chunks, and query analysis."""
        print("✍️ Generating answer...")
        
        if not query_analysis:
            query_analysis = {"query_type": "unknown", "complexity": "simple"}
            
        # Create appropriate prompt based on query type
        prompt = self.create_prompt(query, context_chunks, query_analysis)
        
        try:
            # Generate response
            response = self.gemini.generate_content(prompt)
            answer = response.text
            print("✅ Answer generated.")
            
            # Simple post-processing to ensure page references are included if not present
            pages = sorted(list(set([chunk["metadata"].get("page", "") for chunk in context_chunks])))
            pages = [p for p in pages if p]  # Filter out empty values
            
            # Check if any page references are already in the answer
            has_page_refs = any(f"[p. {p}]" in answer for p in pages)
            if pages and not has_page_refs:
                answer += f"\n\n*References: [p. {', '.join(map(str, pages))}]*"

            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {e}"

