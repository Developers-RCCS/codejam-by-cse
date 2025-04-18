import google.generativeai as genai
from config import GEMINI_API_KEY
from sentence_transformers import SentenceTransformer # Added import
import torch # Added import

genai.configure(api_key=GEMINI_API_KEY)

# Load the Sentence Transformer model
# Use CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} for embeddings")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

def setup_gemini():
    return genai.GenerativeModel("gemini-1.5-flash")

def embed_text(text):
    # Use the Sentence Transformer model for embedding
    # Ensure text is not empty or None
    if not text:
        print("Warning: Attempted to embed empty text. Returning zero vector.")
        # Return a zero vector of the expected dimension
        return [0.0] * embedding_model.get_sentence_embedding_dimension()
        
    # Normalize embeddings is often recommended for cosine similarity / L2 distance
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    # Convert numpy array to list for compatibility if needed elsewhere
    return embedding.tolist()
