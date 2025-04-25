# utils/text_utils.py
import re
import random
from typing import List, Optional

def post_process_answer(raw_answer: str) -> str:
    """
    Applies final polishing touches to a generated answer.
    
    Removes boilerplate phrases, robotic language, and apologetic tones.
    Makes the text more conversational, friendly, and witty.
    
    Args:
        raw_answer: The raw answer text from the generator
        
    Returns:
        Processed text that sounds more natural, caring and conversational
    """
    processed = raw_answer

    # Remove common boilerplate leading phrases (case-insensitive)
    boilerplate_starters = [
        r"^based on the (context|information|text|document|excerpt|passage|provided context) provided,?\s*",
        r"^according to the (text|context|information|document|excerpt|passage|provided context),?\s*",
        r"^the (provided context|text|document) (states|indicates|mentions|shows|suggests|talks about|explains) that,?\s*",
        r"^the context suggests that,?\s*",
        r"^in the provided context,?\s*",
        r"^the information (provided|given|presented) (states|indicates|mentions|shows|suggests|talks about|explains) that,?\s*",
        r"^from the (text|context|information|document|excerpt|passage|provided context),?\s*",
        r"^as (mentioned|stated|indicated|shown|presented) in the (text|context|information|document|excerpt|passage|provided context),?\s*",
    ]
    
    for pattern in boilerplate_starters:
        processed = re.sub(pattern, "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove common boilerplate closing phrases (case-insensitive)
    boilerplate_endings = [
        r"in conclusion,?$\s*",
        r"to summarize,?$\s*",
        r"in summary,?$\s*",
        r"overall,?$\s*",
        r"as a result,?$\s*",
        r"therefore,?$\s*",
        r"thus,?$\s*",
        r"hence,?$\s*",
    ]
    
    for pattern in boilerplate_endings:
        processed = re.sub(pattern, "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove apologetic phrases
    apologetic_phrases = [
        r"I apologize,? but ",
        r"sorry,? but ",
        r"I'm afraid ",
        r"Unfortunately, ",
        r"I don't have (enough|specific|detailed|sufficient|the necessary) information ",
        r"The (text|context|information|document|excerpt|passage|provided context) doesn't (provide|mention|contain|include|cover) ",
    ]
    
    for phrase in apologetic_phrases:
        processed = re.sub(phrase, "", processed, flags=re.IGNORECASE).strip()
    
    # Remove robotic phrases
    robotic_phrases = [
        r"As an AI (assistant|model|language model|tutor),?",
        r"Based on my knowledge,?",
        r"I cannot (provide|access|know|determine|confirm|verify|validate),?",
        r"I don't have (access to|information about|knowledge of|data on),?",
        r"Without (more|additional|further|extra) (context|information|details|data),?",
    ]
    
    for phrase in robotic_phrases:
        processed = re.sub(phrase, "", processed, flags=re.IGNORECASE).strip()

    # Break long paragraphs into shorter ones for readability
    if len(processed) > 300:
        sentences = re.split(r'(?<=[.!?])\s+', processed)
        if len(sentences) > 2:
            new_paragraphs = []
            current_paragraph = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 180:  # Target paragraph length
                    if current_paragraph:  # Avoid empty paragraphs
                        new_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = [sentence]
                        current_length = len(sentence)
                else:
                    current_paragraph.append(sentence)
                    current_length += len(sentence)
            
            # Don't forget the last paragraph
            if current_paragraph:
                new_paragraphs.append(' '.join(current_paragraph))
            
            processed = '\n\n'.join(new_paragraphs)
    
    # Trim leading/trailing whitespace again after potential removals
    processed = processed.strip()

    # Remove potential markdown artifacts at the beginning/end
    processed = re.sub(r"^```(python|markdown)?\s*", "", processed).strip()
    processed = re.sub(r"\s*```$", "", processed).strip()
    
    # Add playful interjections to make multi-part answers more engaging
    if '\n\n' in processed:
        paragraphs = processed.split('\n\n')
        if len(paragraphs) >= 2:
            interjections = [
                "Here's where it gets interesting! ",
                "And now for the juicy part... ",
                "This is my favorite part! ",
                "But that's not all! ",
                "Ready for an exciting twist in the story? ",
            ]
            # Add an interjection to a random paragraph (not the first)
            if len(paragraphs) > 2:
                idx = random.randint(1, len(paragraphs) - 1)
                paragraphs[idx] = random.choice(interjections) + paragraphs[idx]
                processed = '\n\n'.join(paragraphs)
    
    # Ensure the response ends with a friendly, witty, or caring line
    friendly_endings = [
        "Let me know if you want to go on another history adventure!",
        "History is full of surprises—let's find the next one together!",
        "What other historical secrets should we uncover together?",
        "History is always more fun with you asking the questions!",
        "Ready for another round of time travel? Just ask!"
    ]
    
    # Check if the response already ends with one of our friendly endings
    has_friendly_ending = any(processed.endswith(ending) for ending in friendly_endings)
    
    if not has_friendly_ending and not processed.endswith('?'):
        # Add a friendly ending if the response doesn't already have one
        if processed and processed[-1] not in [' ', '\n', '\t', '.', '!', '?']:
            processed += ". "
        elif processed and processed[-1] in ['.', '!', '?']:
            processed += " "
            
    return processed

def format_multi_part_answer(answer: str, complexity: str = "simple") -> str:
    """
    Breaks answers into shorter, friendly paragraphs with playful transitions.
    
    Args:
        answer: The answer text to format
        complexity: The complexity of the question ("simple", "complex")
        
    Returns:
        Formatted text with better paragraph breaks and transitions
    """
    if complexity == "simple" or len(answer) < 200:
        return answer
        
    # If already has paragraphs, ensure they're not too long
    if '\n\n' in answer:
        paragraphs = answer.split('\n\n')
        new_paragraphs = []
        
        for i, para in enumerate(paragraphs):
            if len(para) > 250:  # Split long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                mid = len(sentences) // 2
                new_para1 = ' '.join(sentences[:mid])
                new_para2 = ' '.join(sentences[mid:])
                new_paragraphs.append(new_para1)
                new_paragraphs.append(new_para2)
            else:
                new_paragraphs.append(para)
                
        # Add playful transitions between some paragraphs
        transitions = [
            "\n\nNow, here's something fascinating... ",
            "\n\nThis next part is pretty cool! ",
            "\n\nBut wait, there's more to this story! ",
            "\n\nAnd this is where history gets really interesting! ",
            "\n\nHere's a fun fact you might enjoy: "
        ]
        
        # Add supportive or encouraging comments within paragraphs
        supportive_comments = [
            " (I love how history connects the dots here!) ",
            " (This is such an important point to understand!) ",
            " (Isn't history fascinating?) ",
            " (You're learning some really valuable context here!) ",
            " (This is the kind of detail that makes history come alive!) "
        ]
        
        # Add transitions to some paragraphs (not all)
        if len(new_paragraphs) >= 3:
            transition_indices = random.sample(range(1, len(new_paragraphs)), min(2, len(new_paragraphs)-1))
            for idx in sorted(transition_indices, reverse=True):
                transition = random.choice(transitions)
                new_paragraphs[idx] = transition + new_paragraphs[idx].lstrip()
            
            # Add a supportive comment to one paragraph
            comment_idx = random.choice([i for i in range(len(new_paragraphs)) if i not in transition_indices])
            sentences = re.split(r'(?<=[.!?])\s+', new_paragraphs[comment_idx])
            if len(sentences) >= 2:
                insert_idx = random.randint(1, len(sentences)-1)
                sentences.insert(insert_idx, random.choice(supportive_comments))
                new_paragraphs[comment_idx] = ''.join(sentences)
        
        return '\n\n'.join(new_paragraphs)
    else:
        # No paragraphs yet, create some
        return post_process_answer(answer)  # Use the main function to create paragraphs
