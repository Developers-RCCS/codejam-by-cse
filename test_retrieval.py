import faiss
import pickle
import numpy as np
import logging # Added import
from gemini_utils import embed_text # Use the same embedding function as faiss_store.py

# --- Configuration ---
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.pkl"
TOP_K = 5 # Number of chunks to retrieve

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Load Index and Metadata ---
logger.info(f"Loading FAISS index from {INDEX_PATH}...")
try:
    index = faiss.read_index(INDEX_PATH)
    logger.info(f"Index loaded. Dimension: {index.d}, Total vectors: {index.ntotal}")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}", exc_info=True)
    exit()

logger.info(f"Loading metadata from {METADATA_PATH}...")
try:
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    texts = metadata_store["texts"]
    metadatas = metadata_store["metadatas"]
    logger.info(f"Metadata loaded. Found {len(texts)} text chunks.")
except Exception as e:
    logger.error(f"Error loading metadata: {e}", exc_info=True)
    exit()

# --- Retrieval Function ---
def retrieve_chunks(query: str, k: int = TOP_K):
    """Embeds a query and retrieves the top k chunks from the FAISS index."""
    if not query:
        logger.warning("Query cannot be empty.")
        return

    logger.info(f"\n--- Retrieving top {k} chunks for query: '{query}' ---")

    # 1. Embed the query
    logger.info("Embedding query...")
    try:
        query_embedding = embed_text(query)
        if query_embedding is None or len(query_embedding) == 0:
            logger.error("Error: Failed to generate query embedding.")
            return
        # Convert to numpy array and reshape for FAISS search
        query_embedding_np = np.array([query_embedding]).astype("float32")
        # Normalize the query vector if embeddings were normalized during indexing
        faiss.normalize_L2(query_embedding_np) 
        logger.info(f"Query embedding generated (Dimension: {len(query_embedding)}).")
    except Exception as e:
        logger.error(f"Error during query embedding: {e}", exc_info=True)
        return

    # 2. Search the FAISS index
    logger.info(f"Searching index for top {k} results...")
    try:
        distances, indices = index.search(query_embedding_np, k)
        logger.info("Search complete.")
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return

    # 3. Print results
    logger.info("\n--- Top Results ---")
    if indices.size == 0 or indices[0][0] == -1: # Check if any results were found
        logger.info("No relevant chunks found.")
        return
        
    for i in range(k):
        idx = indices[0][i]
        if idx < 0 or idx >= len(texts): # Check for valid index
             logger.warning(f"Result {i+1}: Invalid index {idx}, skipping.")
             continue
             
        distance = distances[0][i]
        text = texts[idx]
        metadata = metadatas[idx]
        
        logger.info(f"\n--- Result {i+1} (Index: {idx}, Distance: {distance:.4f}) ---")
        logger.info(f"Metadata: {metadata}")
        logger.info(f"Text:\n{text[:300]}...") # Log start of the text

# --- Example Usage ---
if __name__ == "__main__":
    # Example queries (replace or add more as needed)
    sample_queries = [
        "What were the main causes of the French Revolution?",
        "Describe the unification of Germany.",
        "Who was Menelik II?",
        "Explain the concept of the Scramble for Africa.",
        "What was the Industrial Revolution's impact on society?"
    ]

    for q in sample_queries:
        retrieve_chunks(q)
        logger.info("\n==================================================\n")

    # You can also uncomment this to test with user input:
    # while True:
    #     user_query = input("Enter your query (or 'quit' to exit): ")
    #     if user_query.lower() == 'quit':
    #         break
    #     retrieve_chunks(user_query)
    #     logger.info("\n==================================================\n")