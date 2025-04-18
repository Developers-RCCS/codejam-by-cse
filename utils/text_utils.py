# utils/text_utils.py
import re

def post_process_answer(raw_answer: str) -> str:
    """Applies final polishing touches to a generated answer."""
    processed = raw_answer

    # Remove common boilerplate leading phrases (case-insensitive)
    processed = re.sub(r"^based on the context provided,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
    processed = re.sub(r"^according to the text,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
    processed = re.sub(r"^the provided context states that,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
    processed = re.sub(r"^the context suggests that,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
    processed = re.sub(r"^in the provided context,?\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove common boilerplate closing phrases (case-insensitive)
    processed = re.sub(r"in conclusion,?$\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()
    processed = re.sub(r"to summarize,?$\s*", "", processed, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Trim leading/trailing whitespace again after potential removals
    processed = processed.strip()

    # Remove potential markdown artifacts at the beginning/end
    processed = re.sub(r"^```(python|markdown)?\s*", "", processed).strip()
    processed = re.sub(r"\s*```$", "", processed).strip()

    return processed
