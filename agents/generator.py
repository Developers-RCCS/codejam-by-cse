# agents/generator.py
from .base import BaseAgent
from gemini_utils import setup_gemini
from datetime import datetime

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
            
        # Separate chunks by source type
        textbook_chunks = []
        web_chunks = []
        
        for chunk in sorted_chunks:
            source_type = chunk["metadata"].get("source_type", "textbook")  # Default to textbook if not specified
            if source_type == "web":
                web_chunks.append(chunk)
            else:
                textbook_chunks.append(chunk)
            
        # Format textbook context
        textbook_context = ""
        if textbook_chunks:
            textbook_context = "**Textbook Content:**"
            for i, chunk in enumerate(textbook_chunks):
                page = chunk["metadata"].get("page", "Unknown")
                section = chunk["metadata"].get("section", "Unknown")
                textbook_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] ---\n"
                textbook_context += chunk["text"]
                textbook_context += "\n"
        
        # Format web context
        web_context = ""
        if web_chunks:
            web_context = "\n**Web Content:**"
            for i, chunk in enumerate(web_chunks):
                url = chunk["metadata"].get("url", "Unknown Source")
                topic = chunk["metadata"].get("topic", "")
                web_context += f"\n--- Web Excerpt {i+1} [Source: {url}] ---\n"
                web_context += chunk["text"]
                web_context += "\n"
        
        # Combine contexts
        formatted_context = textbook_context + web_context
            
        # Get query type and complexity
        query_type = query_analysis.get("query_type", "unknown")
        complexity = query_analysis.get("complexity", "simple")
        
        # Base prompt structure
        base_prompt = f"""**Context Information:**
{formatted_context}

**Question:**
{query}

Current date and time:
{datetime.now().strftime('%Y-%m-%d')}

"""

        # Has web sources flag
        has_web_sources = len(web_chunks) > 0

        # Customize instructions based on query type
        if query_type == "factual":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this factual question accurately based on the provided context above.
- Provide a direct and concise answer with specific facts from the context.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source in your answer (e.g., "According to Wikipedia...").
- If the answer isn't in the context, clearly state that you don't have that information.
- Answer only what is asked, avoid unnecessary elaboration.
"""
        elif query_type == "causal/analytical":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this analytical question based on the provided context above.
- Provide a structured analysis that explains causes, effects, or developments.
- Organize your response with clear reasoning that connects evidence to conclusions.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source in your analysis (e.g., "According to the historical records from...").
- Focus on explaining "why" or "how" rather than just listing facts.
- If the context doesn't provide sufficient information, acknowledge limitations in your analysis.
"""
        elif query_type == "comparative":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this comparative question based on the provided context above.
- Structure your answer to clearly compare and contrast the items.
- Organize by points of comparison rather than describing each item separately.
- Present similarities and differences in a balanced way.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source in your comparison (e.g., "According to...").
- If the context lacks information about one of the comparison items, acknowledge this limitation.
"""
        else:  # Default/unknown
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer the question accurately based on the provided context above.
- Be concise and clear in your explanation.
- For textbook content, cite the relevant page numbers from the context like this: [p. 42].
- For web content, clearly indicate the source of information (e.g., "According to the article from...").
- If the context doesn't contain the answer, state that clearly and do not guess.
- If multiple pages or sources are relevant, cite them all.
"""

        # Add complexity handling
        if complexity == "complex":
            instructions += """
- This is a complex multi-part question. Make sure to address all aspects.
- Structure your answer with clear sections for each part of the question.
- Ensure you're drawing connections between different parts where relevant.
"""

        # Add special instructions for handling multiple source types
        if has_web_sources:
            instructions += """
**Source Attribution Guidelines:**
- Clearly distinguish between information from the textbook and web sources.
- For information from web sources, end the relevant sentence with a source mention, e.g. [Source: Wikipedia].
- When information appears in both textbook and web sources, prioritize the textbook citation.
- At the end of your answer, provide a "References" section listing all sources used.
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
            
            # Post-processing to ensure references are included
            has_references_section = "References:" in answer or "References" in answer.split("\n")[-1]
            
            if not has_references_section:
                # Extract sources for reference
                textbook_pages = []
                web_sources = []
                
                for chunk in context_chunks:
                    source_type = chunk["metadata"].get("source_type", "textbook")
                    if source_type == "web" and "url" in chunk["metadata"]:
                        web_url = chunk["metadata"]["url"]
                        if web_url not in web_sources:
                            web_sources.append(web_url)
                    else:
                        page = chunk["metadata"].get("page", "")
                        if page and page not in textbook_pages:
                            textbook_pages.append(page)
                
                # Add references section
                references = "\n\n**References:**"
                if textbook_pages:
                    references += f"\n- Textbook pages: {', '.join(map(str, sorted(textbook_pages)))}"
                
                for i, url in enumerate(web_sources):
                    references += f"\n- Web source {i+1}: {url}"
                
                answer += references

            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {e}"

