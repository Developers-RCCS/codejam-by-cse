# agents/generator.py
from .base import BaseAgent
from gemini_utils import setup_gemini
from datetime import datetime
from urllib.parse import urlparse
import random

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating answers using Gemini."""
    def __init__(self):
        print("✨ Initializing Gemini model...")
        self.gemini = setup_gemini()
        print("✅ Gemini model initialized.")
        # Initialize personality traits and conversation starters
        self.initialize_persona()
        
    def initialize_persona(self):
        """Initialize teaching persona traits and conversation elements."""
        # Personality traits
        self.persona_traits = {
            "enthusiasm_level": 0.8,  # 0.0-1.0 scale
            "friendliness": 0.85,
            "formality": 0.5,  # Lower is more casual
            "encouragement": 0.9,
            "humor": 0.6,
        }
        
        # Conversation starters for different topics
        self.topic_starters = {
            "industrial_revolution": [
                "The Industrial Revolution was such a fascinating period of transformation!",
                "Have you ever thought about how different our lives would be without the Industrial Revolution?",
                "The Industrial Revolution completely changed how people lived and worked."
            ],
            "world_war": [
                "Understanding the World Wars helps us recognize patterns that still affect global politics today.",
                "The World Wars were watershed moments that reshaped our modern world.",
                "Studying the World Wars gives us important lessons about conflict and peace."
            ],
            "sri_lanka": [
                "Sri Lanka has such a rich and complex history!",
                "The history of Sri Lanka shows fascinating intersections of culture and colonialism.",
                "Sri Lankan history provides great examples of how geography shapes a nation's development."
            ]
        }
        
        # Enthusiasm expressions
        self.enthusiasm_phrases = [
            "That's a great question!",
            "What a fascinating topic to explore!",
            "I'm excited to help you understand this!",
            "This is actually one of my favorite historical subjects!",
            "That's a really thoughtful question.",
            "I love discussing this period of history!"
        ]
        
        # Encouragement phrases
        self.encouragement_phrases = [
            "You're asking exactly the right questions to understand this topic better.",
            "That's a very perceptive question about a complex topic.",
            "You have a good eye for the important historical details!",
            "You're making excellent connections between these historical events.",
            "Keep thinking critically about history like this!"
        ]
        
        # Follow-up questions to encourage engagement
        self.follow_up_questions = [
            "What aspect of this interests you most?",
            "Does this connect to anything else you've been learning about?",
            "Can you think of any modern parallels to these historical events?",
            "Have you thought about how this might have played out differently?",
            "What do you think was the most significant impact of these events?"
        ]
        
        # Transition phrases between topics
        self.transitions = [
            "Building on that idea...",
            "This connects to another important aspect...",
            "Looking at this from another angle...",
            "What's also interesting about this period is...",
            "To put this in context..."
        ]
        
        # Discourse markers for more natural speech
        self.discourse_markers = [
            "You know,",
            "Actually,",
            "Interestingly,",
            "Of course,",
            "Well,",
            "In fact,"
        ]

    def _get_random_element(self, element_list):
        """Get a random element from a list."""
        if not element_list:
            return ""
        return random.choice(element_list)
        
    def _personalize_response(self, base_response, conversation_context=None):
        """
        Add personality elements to a response based on conversation context.
        
        Args:
            base_response: The educational content of the response
            conversation_context: Optional context about the conversation state and user
            
        Returns:
            str: A more personalized, conversational response
        """
        if not conversation_context:
            conversation_context = {
                "conversation_state": "exploration",
                "rapport_level": 3,
                "interaction_count": 1
            }
        
        # Extract context information    
        state = conversation_context.get("conversation_state", "exploration")
        rapport = conversation_context.get("rapport_level", 3)
        interactions = conversation_context.get("interaction_count", 1)
        recurring_topics = conversation_context.get("topics_of_interest", [])
        
        # Personalize based on conversation state
        personalized_response = ""
        
        # For first interactions or greeting state, use more welcoming language
        if state == "greeting" or interactions <= 2:
            # Add enthusiastic greeting
            if rapport <= 5:
                personalized_response += f"{self._get_random_element(self.enthusiasm_phrases)} "
        
        # Adding the core educational content
        personalized_response += base_response
        
        # Add follow-up or encouragement for ongoing conversations
        if state in ["exploration", "learning"] and rapport >= 3:
            if random.random() < 0.7:  # 70% chance
                personalized_response += f"\n\n{self._get_random_element(self.encouragement_phrases)}"
                
            # Add a follow-up question if we're developing rapport
            if random.random() < 0.5 and rapport >= 5:  # 50% chance when rapport is high
                personalized_response += f" {self._get_random_element(self.follow_up_questions)}"
        
        # If the user has recurring interests, acknowledge them
        if recurring_topics and random.random() < 0.3:  # 30% chance
            topic = recurring_topics[0].replace(" ", "_")
            if topic in self.topic_starters:
                starter = self._get_random_element(self.topic_starters[topic])
                personalized_response = f"{starter} {personalized_response}"
                
        return personalized_response

    def create_prompt(self, query: str, context_chunks: list[dict], query_analysis: dict, conversation_context: dict = None) -> str:
        """Create an effective prompt based on query type and context with enhanced source attribution and personality."""
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

        # Conversation context for personalization
        conversation_state = "exploration"
        if conversation_context:
            conversation_state = conversation_context.get("conversation_state", "exploration")
        
        # Get chat history if available
        chat_history = ""
        if conversation_context and conversation_context.get("chat_history"):
            history = conversation_context["chat_history"]
            if history:
                chat_history = "\n**Recent Conversation:**\n"
                for msg in history[-3:]:  # Last 3 messages
                    sender = "Student" if msg["sender"] == "user" else "Tutor"
                    chat_history += f"{sender}: {msg['message']}\n"

        # Customize instructions based on query type with personality elements
        personality_instructions = """**Personality Guidelines:**
You are Yuhasa, a friendly and enthusiastic history tutor with a passion for making history come alive. You're speaking with a Grade 11 student who's looking for help understanding historical concepts.

- Be warm and encouraging, using a conversational, slightly informal tone.
- Show genuine enthusiasm for history ("The Industrial Revolution was actually a fascinating period!")
- Ask occasional follow-up questions to spark curiosity ("Have you ever wondered why...?")
- Use relatable analogies to explain complex concepts
- Vary your response style-sometimes be energetic, other times thoughtful
- Address the student directly using "you" to create connection
- Occasionally share brief, engaging historical facts beyond the direct question
- Acknowledge when topics might be challenging to understand
- Be supportive when information isn't available ("That's a great question, though I don't have that specific information in my sources.")

Remember to maintain this friendly, encouraging tone while still providing academically accurate information with proper citations. Your goal is to be both an educational resource and an engaging conversation partner.
"""

        if query_type == "factual":
            instructions = f"""**Instructions:**
{personality_instructions}

You are answering a factual history question. 
- Provide an accurate and engaging answer based on the context provided.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source domain in your answer (e.g., "According to Wikipedia...").
- If the answer isn't in the context, clearly state that you don't have that information.
- Use a warm, conversational tone that makes history feel accessible and interesting.
- Include 1-2 interesting related facts if appropriate to build engagement.
"""
        elif query_type == "causal/analytical":
            instructions = f"""**Instructions:**
{personality_instructions}

You are answering an analytical history question that requires explanation of causes, effects, or developments.
- Provide a structured analysis that explains causes, effects, or developments.
- Organize your response with clear reasoning that connects evidence to conclusions.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source domain in your analysis (e.g., "According to the historical records from...").
- Focus on explaining "why" or "how" rather than just listing facts.
- Use a conversational tone that makes complex historical analysis accessible.
- Consider asking a thought-provoking follow-up question at the end to encourage deeper thinking.
"""
        elif query_type == "comparative":
            instructions = f"""**Instructions:**
{personality_instructions}

You are answering a comparative history question that requires analyzing similarities and differences.
- Structure your answer to clearly compare and contrast the items.
- Organize by points of comparison rather than describing each item separately.
- Present similarities and differences in a balanced way.
- For textbook content, cite page numbers in square brackets like this: [p. 42].
- For web content, mention the source domain in your comparison (e.g., "According to...").
- Use analogies or examples that make the comparison more relatable to a Grade 11 student.
- Maintain an engaging, conversational tone throughout your explanation.
"""
        else:  # Default/unknown
            instructions = f"""**Instructions:**
{personality_instructions}

You are answering a history question from a Grade 11 student.
- Provide a clear, accurate, and engaging answer based on the provided context.
- For textbook content, cite the relevant page numbers from the context like this: [p. 42].
- For web content, clearly indicate the source of information (e.g., "According to the article from...").
- If the context doesn't contain the answer, state that clearly and do not guess.
- Use a warm, conversational tone that makes history interesting and accessible.
- If appropriate, ask a thoughtful follow-up question to encourage further exploration.
"""

        # Add complexity handling
        if complexity == "complex":
            instructions += """
- This is a complex multi-part question. Make sure to address all aspects.
- Structure your answer with clear sections for each part of the question.
- Ensure you're drawing connections between different parts where relevant.
- Use a slightly more detailed approach, but maintain the engaging conversational style.
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
- Present these different perspectives in a way that helps students understand that history often involves different interpretations.
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

        # Add conversation context to help generate more contextual responses
        if conversation_context:
            context_str = "\n**Conversation Context:**\n"
            if conversation_context.get("rapport_level"):
                context_str += f"- Rapport level: {conversation_context['rapport_level']}/10\n"
            if conversation_context.get("conversation_state"):
                context_str += f"- Conversation state: {conversation_context['conversation_state']}\n"
            if conversation_context.get("topics_of_interest"):
                context_str += f"- Student's recurring topics of interest: {', '.join(conversation_context['topics_of_interest'])}\n"
            if conversation_context.get("concepts_already_explained"):
                context_str += f"- Concepts already explained: {', '.join(conversation_context['concepts_already_explained'][:5])}\n"
                
            instructions += context_str

        # Add chat history if available
        if chat_history:
            instructions += f"\n{chat_history}"

        # Finalize prompt
        complete_prompt = f"{base_prompt}\n{instructions}\n**Answer:**"
        return complete_prompt

    def run(self, query: str, context_chunks: list[dict], query_analysis: dict = None, conversation_context: dict = None) -> str:
        """Generates an answer based on the query, context chunks, and query analysis with enhanced source attribution and personality."""
        print("✍️ Generating answer...")
        
        if not query_analysis:
            query_analysis = {"query_type": "unknown", "complexity": "simple"}
            
        # Create appropriate prompt based on query type
        prompt = self.create_prompt(query, context_chunks, query_analysis, conversation_context)
        
        try:
            # Generate response
            response = self.gemini.generate_content(prompt)
            answer = response.text
            print("✅ Answer generated.")
            
            # Personalize the response if we have conversation context
            if conversation_context:
                answer = self._personalize_response(answer, conversation_context)
            
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

            # Extract concepts that were explained in this answer for conversation memory
            explained_concepts = []
            # Simple concept extraction from headings or bold text
            import re
            # Match text between ** or # heading markers
            concept_pattern = r'\*\*(.*?)\*\*|# (.*?)\n|## (.*?)\n'
            for match in re.finditer(concept_pattern, answer):
                concept = match.group(1) or match.group(2) or match.group(3)
                if concept and len(concept) > 3 and concept not in ["References"]:
                    explained_concepts.append(concept)

            # Return both the answer and any concepts that were explained
            return {
                "answer": answer,
                "explained_concepts": explained_concepts
            }
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": f"Sorry, I encountered an error while generating the answer: {e}",
                "explained_concepts": []
            }

