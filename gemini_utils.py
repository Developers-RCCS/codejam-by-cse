import google.generativeai as genai
import logging # Added import
from config import GEMINI_API_KEY
from sentence_transformers import SentenceTransformer # Added import
import torch # Added import

logger = logging.getLogger(__name__) # Get a logger for this module

genai.configure(api_key=GEMINI_API_KEY)

# Load the Sentence Transformer model
# Use CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device} for embeddings")
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}", exc_info=True)
    embedding_model = None # Ensure model is None if loading fails

def setup_gemini():
    return genai.GenerativeModel("gemini-1.5-flash")

def embed_text(text):
    if embedding_model is None:
        logger.error("Embedding model not loaded. Cannot embed text.")
        return None # Or raise an exception

    # Use the Sentence Transformer model for embedding
    # Ensure text is not empty or None
    if not text:
        logger.warning("Attempted to embed empty text. Returning zero vector.")
        # Return a zero vector of the expected dimension
        try:
            dim = embedding_model.get_sentence_embedding_dimension()
            return [0.0] * dim
        except Exception as e:
            logger.error(f"Could not get embedding dimension: {e}")
            return None # Indicate failure

    try:
        # Normalize embeddings is often recommended for cosine similarity / L2 distance
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        # Convert numpy array to list for compatibility if needed elsewhere
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error during text embedding: {e}", exc_info=True)
        return None # Indicate failure
