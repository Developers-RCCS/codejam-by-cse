# utils/chunk_utils.py
import numpy as np
import re
import logging
from gemini_utils import embed_text

logger = logging.getLogger(__name__)  # Get a logger for this module

def calculate_chunk_similarity(chunks):
    """Calculate similarity between chunks to avoid adding redundant content."""
    if len(chunks) <= 1:
        return []

    # Generate embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        try:
            # Assuming embed_text returns a list or numpy array
            emb = embed_text(chunk["text"])
            # Ensure it's a numpy array for dot product
            embeddings.append(np.array(emb))
        except Exception as e:
            logger.error(f"⚠️ Error embedding chunk: {e}", exc_info=True)
            # Use a zero vector of appropriate dimension if embedding fails
            # Assuming embedding dimension is 768, adjust if different
            embeddings.append(np.zeros(768))

    # Ensure all embeddings are numpy arrays
    embeddings = [np.array(emb) for emb in embeddings]

    # Normalize embeddings for cosine similarity (dot product of normalized vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms[norms == 0] = 1e-9
    normalized_embeddings = embeddings / norms

    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            # Dot product of normalized vectors is cosine similarity
            similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
            similarities.append((i, j, similarity))

    return sorted(similarities, key=lambda x: x[2], reverse=True)

def filter_redundant_chunks(chunks, similarity_threshold=0.85):
    """Remove chunks that are too similar to higher-ranked chunks based on confidence."""
    if len(chunks) <= 1:
        return chunks

    # Ensure chunks are sorted by confidence descending before filtering
    # This makes sure we keep the highest confidence chunk among similar ones
    sorted_chunks = sorted(chunks, key=lambda x: x.get("confidence", 0), reverse=True)

    similarities = calculate_chunk_similarity(sorted_chunks)
    indices_to_remove = set()

    # Mark lower-confidence chunks that are too similar to higher-confidence ones
    for i, j, similarity in similarities:
        # Since list is sorted by confidence (i < j implies confidence[i] >= confidence[j]),
        # if similarity is high, mark the lower confidence chunk (j) for removal.
        if similarity > similarity_threshold:
            # Check if j is already marked for removal to avoid redundant checks
            if j not in indices_to_remove:
                 indices_to_remove.add(j)
                 # logger.debug(f"Marking chunk {j} (conf: {sorted_chunks[j].get('confidence', 0):.2f}) as redundant to chunk {i} (conf: {sorted_chunks[i].get('confidence', 0):.2f}, sim: {similarity:.2f})")

    # Create filtered list
    filtered_chunks = [chunk for i, chunk in enumerate(sorted_chunks) if i not in indices_to_remove]

    logger.info(f"✅ Filtered out {len(sorted_chunks) - len(filtered_chunks)} redundant chunks (threshold > {similarity_threshold}).")
    return filtered_chunks

def simple_keyword_score(text_lower, query_keywords_set):
    """Calculate a simple score based on keyword overlap."""
    text_tokens = set(re.findall(r'\b\w+\b', text_lower))
    common_keywords = text_tokens.intersection(query_keywords_set)
    return len(common_keywords) / len(query_keywords_set) if query_keywords_set else 0.0

def simple_entity_score(text_lower, entities):
    """Calculate score based on simple entity presence."""
    score = 0.0
    if not entities:
        return 0.0
    entities_lower = {e.lower() for e in entities}
    # More robust check: use word boundaries
    for entity in entities_lower:
        if re.search(r'\b' + re.escape(entity) + r'\b', text_lower):
            score += 1
    return score / len(entities)

def section_relevance_score(metadata, query_type):
    """Score chunks based on section relevance to query type."""
    section = metadata.get("section", "").lower()
    score = 0.5 # Default score
    if not section: # No section info, return default
        return score

    # Define keywords for different query types
    factual_keywords = ["overview", "introduction", "summary", "facts", "data", "definition", "timeline"]
    causal_keywords = ["causes", "effects", "impact", "analysis", "consequences", "reasons", "development"]
    comparative_keywords = ["comparison", "versus", "differences", "similarities", "contrast"]

    # Check relevance based on query type
    if query_type == "factual" and any(term in section for term in factual_keywords):
        score = 0.8
    elif query_type == "causal/analytical" and any(term in section for term in causal_keywords):
        score = 0.8
    elif query_type == "comparative" and any(term in section for term in comparative_keywords):
        score = 0.8
    # Add more rules if needed

    return score
