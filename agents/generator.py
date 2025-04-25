# agents/generator.py
from .base import BaseAgent
from gemini_utils import setup_gemini
from datetime import datetime
from urllib.parse import urlparse

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        print("✨ Initializing Gemini model...")
        self.gemini = setup_gemini()
        print("✅ Gemini model initialized.")

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict) -> str:
        """Create an effective prompt based on query type and context with enhanced source attribution."""
        # Extract context text - if chunks have a confidence score, sort by it
        if context_chunks and "confidence" in context_chunks[0]:
            sorted_chunks = sorted(context_chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        else:
            sorted_chunks = context_chunks
            
        # Separate chunks by source type
        textbook_chunks = []
        web_chunks = []
        
        # Check for chunks with conflict resolution metadata
        chunks_with_conflicts = [c for c in sorted_chunks if c.get("metadata", {}).get("conflict_resolution") == "primary"]
        alternative_perspective_chunks = [c for c in sorted_chunks if c.get("metadata", {}).get("conflict_resolution") == "alternative"]
        
        # Group chunks by source
        for chunk in sorted_chunks:
            source_type = chunk["metadata"].get("source_type", "textbook")  # Default to textbook if not specified
            if source_type == "web":
                web_chunks.append(chunk)
            else:
                textbook_chunks.append(chunk)
        
        # Format textbook context with improved structure
        textbook_context = ""
        if textbook_chunks:
            textbook_context = "**Textbook Content:**"
            for i, chunk in enumerate(textbook_chunks):
                page = chunk["metadata"].get("page", "Unknown")
                section = chunk["metadata"].get("section", "Unknown")
                conflict_status = chunk["metadata"].get("conflict_resolution", "")
                
                # Highlight conflicts for the model's awareness
                if conflict_status == "primary":
                    textbook_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] (PRIMARY SOURCE) ---\n"
                elif conflict_status == "alternative":
                    textbook_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] (ALTERNATIVE PERSPECTIVE) ---\n"
                else:
                    textbook_context += f"\n--- Excerpt {i+1} [Page {page}, Section: {section}] ---\n"
                
                textbook_context += chunk["text"]
                textbook_context += "\n"
        
        # Format web context with improved attribution and reliability indicators
        web_context = ""
        if web_chunks:
            web_context = "\n**Web Content:**"
            for i, chunk in enumerate(web_chunks):
                url = chunk["metadata"].get("url", "Unknown Source")
                domain = urlparse(url).netloc if url else "Unknown Domain"
                topic = chunk["metadata"].get("topic", "")
                quality = chunk["metadata"].get("quality_score", 0.5)
                conflict_status = chunk["metadata"].get("conflict_resolution", "")
                
                # Add reliability indicator based on quality score
                reliability = "High reliability" if quality > 0.8 else "Medium reliability" if quality > 0.6 else "Use with caution"
                
                # Highlight conflicts for the model's awareness
                if conflict_status == "primary":
                    web_context += f"\n--- Web Excerpt {i+1} [Source: {domain}] ({reliability}) (PRIMARY SOURCE) ---\n"
                elif conflict_status == "alternative":
                    web_context += f"\n--- Web Excerpt {i+1} [Source: {domain}] ({reliability}) (ALTERNATIVE PERSPECTIVE) ---\n"
                else:
                    web_context += f"\n--- Web Excerpt {i+1} [Source: {domain}] ({reliability}) ---\n"
                    
                web_context += chunk["text"]
                web_context += f"\nSource URL: {url}\n"
        
        # Check if we have conflicting perspectives that need special handling
        has_conflicts = any(c.get("metadata", {}).get("multiple_perspectives", False) for c in context_chunks)
        conflicting_topics = []
        
        if has_conflicts:
            # Identify topics with conflicts
            for chunk in context_chunks:
                if chunk["metadata"].get("multiple_perspectives", False):
                    topic = chunk["metadata"].get("conflict_topic", "")
                    if topic and topic not in conflicting_topics:
                        conflicting_topics.append(topic)
        
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
- For web content, mention the source domain in your answer (e.g., "According to Wikipedia...").
- If the answer isn't in the context, clearly state that you don't have that information.
- Answer only what is asked, avoid unnecessary elaboration.
"""
        elif query_type == "causal/analytical":
            instructions = """**Instructions:**
You are Yuhasa, a helpful AI tutor specializing in Grade 11 History. Answer this analytical question based on the provided context above.
- Provide a structured analysis that explains causes, effects, or developments.
- Organize your response with clear reasoning that connects evidence to conclusions.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source domain in your analysis (e.g., "According to the historical records from...").
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
- For web content, mention the source domain in your comparison (e.g., "According to...").
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

        # Add special instructions for handling multiple perspectives if conflicts detected
        if has_conflicts:
            instructions += """
**Multiple Perspectives Handling:**
- Some information in the sources presents different perspectives on the same topic.
- When discussing topics with conflicting information, present the primary perspective first.
- Then acknowledge alternative perspectives with phrases like "However, according to [source]..." or "An alternative view suggests..."
- Make it clear to the reader when information is contested or when sources disagree.
- For significant factual discrepancies (like dates, numbers, or key events), present both versions and indicate which has stronger source credibility.
"""

        # Add special instructions for handling multiple source types
        if has_web_sources:
            instructions += """
**Source Attribution Guidelines:**
- Clearly distinguish between information from the textbook and web sources.
- For information from web sources, end the relevant sentence with the source domain in parentheses, e.g. (britannica.com).
- When information appears in both textbook and web sources, prioritize the textbook citation.
- At the end of your answer, provide a "References" section listing all sources used.
- Format web references as: domain.com - Brief description of the source's authority/relevance
- Format textbook references as: Textbook, pages: X-Y
"""

        # Finalize prompt
        complete_prompt = f"{base_prompt}\n{instructions}\n**Answer:**"
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None) -> str:
        """Generates an answer based on the query, context chunks, and query analysis with enhanced source attribution."""
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
            
            # Post-processing to ensure references are included and properly formatted
            has_references_section = "References:" in answer or "References" in answer.split("\n")[-1]
            
            if not has_references_section:
                # Extract sources for reference
                textbook_pages = []
                web_sources = {}  # Use dict to track domain -> URL for full references
                
                for chunk in context_chunks:
                    source_type = chunk["metadata"].get("source_type", "textbook")
                    if source_type == "web" and "url" in chunk["metadata"]:
                        web_url = chunk["metadata"]["url"]
                        domain = urlparse(web_url).netloc
                        if domain not in web_sources:
                            web_sources[domain] = web_url
                    else:
                        page = chunk["metadata"].get("page", "")
                        if page and page not in textbook_pages:
                            textbook_pages.append(page)
                
                # Sort sources for consistent presentation
                textbook_pages.sort(key=lambda x: str(x))
                sorted_domains = sorted(web_sources.keys())
                
                # Add formatted references section
                references = "\n\n**References:**"
                if textbook_pages:
                    references += f"\n- Textbook pages: {', '.join(map(str, textbook_pages))}"
                
                for i, domain in enumerate(sorted_domains):
                    url = web_sources[domain]
                    # Extract website name for better formatting
                    website_name = domain
                    if website_name.startswith("www."):
                        website_name = website_name[4:]
                    references += f"\n- {website_name} ({url})"
                
                answer += references

            # Check if the answer properly presents multiple perspectives when needed
            has_multiple_perspectives = any(c.get("metadata", {}).get("multiple_perspectives", False) for c in context_chunks)
            mentions_alternative_perspective = "alternative" in answer.lower() or "however" in answer.lower() or "another perspective" in answer.lower()
            
            if has_multiple_perspectives and not mentions_alternative_perspective:
                # Add a reminder about multiple perspectives
                perspective_note = "\n\nNote: This topic contains multiple perspectives or potentially conflicting information from different sources. Consider consulting the original sources for a more complete understanding."
                
                # Add before the references if they exist, otherwise at the end
                if "**References:**" in answer:
                    parts = answer.split("**References:**")
                    answer = parts[0] + perspective_note + "\n\n**References:**" + parts[1]
                else:
                    answer += perspective_note

            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {e}"

