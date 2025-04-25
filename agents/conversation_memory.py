# agents/conversation_memory.py
from .base import BaseAgent
import json
import os
import time
from datetime import datetime
import re
from collections import defaultdict

class ConversationMemoryAgent(BaseAgent):
    """Agent responsible for maintaining conversation context and user preferences."""
    
    def __init__(self, memory_file="conversation_memory.json"):
        """Initialize the conversation memory agent."""
        self.memory_file = memory_file
        self.current_session_id = None
        self.session_memory = {}
        self.global_memory = self._load_memory()
        self.conversation_state = "greeting"  # greeting, exploration, learning, wrap-up
        self.rapport_level = 0  # 0-10 scale of relationship development
        
    def _load_memory(self):
        """Load memory from file if available."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading conversation memory: {e}")
        return {
            "global_topics": {},
            "session_history": {},
            "user_preferences": {
                "explanation_style": "balanced",  # detailed, balanced, concise
                "interests": [],
                "difficulty_level": "medium",  # basic, medium, advanced
                "previously_understood_concepts": []
            }
        }
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.global_memory, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation memory: {e}")
    
    def start_new_session(self, session_id=None):
        """Start a new conversation session."""
        if not session_id:
            session_id = f"session_{int(time.time())}"
            
        self.current_session_id = session_id
        self.conversation_state = "greeting"
        self.rapport_level = 0
        
        # Initialize session memory
        self.session_memory = {
            "messages": [],
            "topics_discussed": [],
            "concepts_explained": [],
            "user_questions": [],
            "session_start": datetime.now().isoformat(),
            "interaction_count": 0
        }
        
        # Add to global memory
        if session_id not in self.global_memory["session_history"]:
            self.global_memory["session_history"][session_id] = {
                "start_time": datetime.now().isoformat(),
                "topics": [],
                "summary": ""
            }
        
        return session_id
    
    def add_message(self, sender, message, metadata=None):
        """Add a message to the current conversation history."""
        if not self.current_session_id:
            self.start_new_session()
            
        if not metadata:
            metadata = {}
            
        # Add message to session memory
        message_entry = {
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        self.session_memory["messages"].append(message_entry)
        self.session_memory["interaction_count"] += 1
        
        # Update conversation state
        if len(self.session_memory["messages"]) <= 2:
            self.conversation_state = "greeting"
        elif len(self.session_memory["messages"]) >= 8:
            self.conversation_state = "learning"
        else:
            self.conversation_state = "exploration"
            
        # Update rapport level based on interaction count
        if self.rapport_level < 10:
            self.rapport_level = min(10, self.session_memory["interaction_count"] // 2)
            
        # Analyze user messages for topics and concepts
        if sender == "user":
            # Extract topics from user message
            self._extract_topics(message)
            
            # Track as a question for follow-up management
            if self._is_question(message):
                self.session_memory["user_questions"].append({
                    "question": message, 
                    "timestamp": datetime.now().isoformat()
                })
                
        # Save after updates
        self._save_memory()
        
        return self.conversation_state
    
    def _extract_topics(self, message):
        """Extract potential historical topics from a message."""
        # Simple topic extraction using keywords
        historical_topics = [
            "industrial revolution", "world war", "cold war", "civil war", 
            "french revolution", "american revolution", "sri lanka", "colonization",
            "independence", "monarchy", "democracy", "empire", "ancient", "medieval",
            "renaissance", "enlightenment", "modern history", "war", "treaty", 
            "civilization", "culture", "economy", "politics", "society"
        ]
        
        found_topics = []
        message_lower = message.lower()
        
        for topic in historical_topics:
            if topic in message_lower:
                found_topics.append(topic)
                
                # Add to global topics tracking
                if topic not in self.global_memory["global_topics"]:
                    self.global_memory["global_topics"][topic] = 0
                self.global_memory["global_topics"][topic] += 1
                
        if found_topics:
            self.session_memory["topics_discussed"].extend(found_topics)
            # Remove duplicates
            self.session_memory["topics_discussed"] = list(set(self.session_memory["topics_discussed"]))
    
    def _is_question(self, message):
        """Determine if a message is likely a question."""
        # Check for question marks
        if "?" in message:
            return True
            
        # Check for question words
        question_starters = ["what", "why", "how", "when", "where", "who", "which", "can", "do", "did", "is", "are", "was", "were"]
        message_words = message.lower().split()
        if message_words and message_words[0] in question_starters:
            return True
            
        return False
    
    def record_explained_concept(self, concept):
        """Record that a concept has been explained to the user."""
        if not self.current_session_id:
            return
            
        if concept not in self.session_memory["concepts_explained"]:
            self.session_memory["concepts_explained"].append(concept)
            
        # Add to global user preferences
        if concept not in self.global_memory["user_preferences"]["previously_understood_concepts"]:
            self.global_memory["user_preferences"]["previously_understood_concepts"].append(concept)
            
        self._save_memory()
    
    def has_concept_been_explained(self, concept):
        """Check if a concept has already been explained in this session or before."""
        if not self.current_session_id:
            return False
            
        # Check current session
        if concept in self.session_memory["concepts_explained"]:
            return True
            
        # Check global memory
        return concept in self.global_memory["user_preferences"]["previously_understood_concepts"]
    
    def update_user_preference(self, preference_type, value):
        """Update a user preference."""
        if preference_type in self.global_memory["user_preferences"]:
            if isinstance(self.global_memory["user_preferences"][preference_type], list):
                if value not in self.global_memory["user_preferences"][preference_type]:
                    self.global_memory["user_preferences"][preference_type].append(value)
            else:
                self.global_memory["user_preferences"][preference_type] = value
                
        self._save_memory()
    
    def get_recurring_topics(self):
        """Get topics the user seems most interested in based on frequency."""
        if not self.global_memory["global_topics"]:
            return []
            
        # Sort topics by frequency
        sorted_topics = sorted(
            self.global_memory["global_topics"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [topic for topic, count in sorted_topics if count > 1][:5]
    
    def get_recent_questions(self, count=3):
        """Get the most recent user questions."""
        if not self.current_session_id:
            return []
            
        questions = self.session_memory.get("user_questions", [])
        return [q["question"] for q in questions[-count:]]
    
    def generate_personalized_context(self):
        """Generate conversation context for personalization."""
        context = {
            "conversation_state": self.conversation_state,
            "rapport_level": self.rapport_level,
            "topics_of_interest": self.get_recurring_topics(),
            "recent_questions": self.get_recent_questions(),
            "concepts_already_explained": self.session_memory.get("concepts_explained", []),
            "user_preferences": self.global_memory["user_preferences"],
            "interaction_count": self.session_memory.get("interaction_count", 0)
        }
        
        return context
    
    def get_chat_history(self, max_messages=10):
        """Get recent chat history."""
        if not self.current_session_id or not self.session_memory.get("messages"):
            return []
            
        return self.session_memory["messages"][-max_messages:]
    
    def reset(self):
        """Reset the current session memory."""
        self.current_session_id = None
        self.session_memory = {}
        self.conversation_state = "greeting"
        self.rapport_level = 0
    
    def run(self, action="update", **kwargs):
        """Run the agent with specified action and parameters."""
        if action == "start_session":
            session_id = kwargs.get("session_id")
            return self.start_new_session(session_id)
        elif action == "add_message":
            sender = kwargs.get("sender")
            message = kwargs.get("message")
            metadata = kwargs.get("metadata")
            return self.add_message(sender, message, metadata)
        elif action == "get_context":
            return self.generate_personalized_context()
        elif action == "get_history":
            max_messages = kwargs.get("max_messages", 10)
            return self.get_chat_history(max_messages)
        elif action == "record_concept":
            concept = kwargs.get("concept")
            return self.record_explained_concept(concept)
        elif action == "update_preference":
            preference_type = kwargs.get("preference_type")
            value = kwargs.get("value")
            return self.update_user_preference(preference_type, value)
        else:
            return {"error": "Unknown action"}