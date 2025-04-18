# utils/messages.py
import random

NOT_FOUND_MESSAGES = [
    "Ooh, that's a tricky one! My textbook doesn't seem to go into detail on that specific point. Maybe we could try phrasing it differently, or ask about something related? 😊",
    "Hmm, stumped me there! Looks like the textbook is a bit quiet on that particular topic. Got another historical mystery for me?",
    "Good question! I scanned my notes (aka the textbook!), but couldn't find the specifics on that. What else is on your mind?",
    "My apologies, but the provided textbook excerpts don't seem to cover that. Is there another angle we could explore?",
    "Interesting question! Unfortunately, the details aren't in the sections I have access to. Perhaps we can focus on a related event mentioned in the book?"
]

CLOSING_REMARKS = [
    "Hope that helps! Ask me another!",
    "Anything else you're curious about?",
    "Happy to help! What's next on your mind? 😉",
    "Let me know if you have more questions!",
    "Was there anything else I can help you with today?"
]

def get_random_message(message_type: str) -> str:
    """Returns a random message based on the specified type."""
    if message_type == 'not_found':
        return random.choice(NOT_FOUND_MESSAGES)
    elif message_type == 'closing':
        return random.choice(CLOSING_REMARKS)
    else:
        return "" # Or raise an error for unknown type
