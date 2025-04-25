# utils/messages.py
import random
from typing import Literal, Optional

# Not-found messages with caring, witty tone
NOT_FOUND_MESSAGES = [
    "Looks like the textbook is keeping secrets from us! Want to try another question and see if you can stump me again? 😉",
    "You've got some seriously good questions! This one isn't in my notes, but I'm always up for another challenge.",
    "My history book is playing hide-and-seek with that detail. Let's outsmart it together!",
    "Oh, you found a gap in my history book! I love how you're thinking outside the textbook. Ready for another adventure?",
    "Well played! You've found something my textbook doesn't cover. Care to try another angle? I love a good challenge! 📚"
]

# Warm, playful closing remarks
CLOSING_REMARKS = [
    "History is always more fun with you asking the questions!",
    "Ready for another round of time travel? Just ask!",
    "Let's see what other mysteries we can solve together!",
    "Your curiosity is as impressive as the ancient kings! What else shall we discover?",
    "I'm enjoying our historical adventure! What other secrets is history hiding? 😄"
]

# Caring, supportive closing phrases
SUPPORTIVE_CLOSINGS = [
    "What else can I help you explore today? 📚",
    "I'm here to help whenever you're ready for another history question!",
    "Let's keep this learning adventure going - what else interests you?",
    "History is full of surprises—let's find the next one together!",
    "You ask such thoughtful questions! What other historical puzzles can I help you solve?"
]

# Playful check-ins and follow-ups
PLAYFUL_FOLLOWUPS = [
    "Did that help, or should we dig a little deeper?",
    "If you want more details or a fun fact, just ask—I'm always ready!",
    "You're making history fun—keep the questions coming!",
    "Was that what you were looking for, or should we explore more? I'm always up for adventure!",
    "Think you can challenge me with another historical puzzle?"
]

# Greeting messages
GREETING_MESSAGES = [
    "Hello there, history explorer! Ready to dive into the fascinating world of Sri Lankan history? 😄",
    "Welcome back! What historical mystery shall we solve today? 📚",
    "I've been waiting for you! What historical adventure are we embarking on today?",
    "So glad you're here! Ready to make history fun together? 😊"
]

# Message type mapping for get_random_message function
MESSAGE_TYPES = {
    'not_found': NOT_FOUND_MESSAGES,
    'closing': CLOSING_REMARKS,
    'supportive_closing': SUPPORTIVE_CLOSINGS,
    'followup': PLAYFUL_FOLLOWUPS,
    'greeting': GREETING_MESSAGES
}

def get_random_message(message_type: Literal['not_found', 'closing', 'supportive_closing', 'followup', 'greeting']) -> str:
    """
    Returns a random message based on the specified type.
    
    Args:
        message_type: Type of message to return ('not_found', 'closing', 'supportive_closing', 'followup', 'greeting')
        
    Returns:
        A random message string of the specified type
    """
    return random.choice(MESSAGE_TYPES.get(message_type, [""]))

def get_playful_response(response_type: str, custom_messages: Optional[list[str]] = None) -> str:
    """
    Returns a random playful response of the specified type, with option to provide custom messages.
    
    Args:
        response_type: Type of message to return (using MESSAGE_TYPES keys)
        custom_messages: Optional list of custom messages to choose from instead
        
    Returns:
        A random message string of the specified type or from custom messages
    """
    if custom_messages:
        return random.choice(custom_messages)
    return get_random_message(response_type)
