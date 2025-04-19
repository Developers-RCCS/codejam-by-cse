# agents/retriever.py
import logging
import faiss
import numpy as np
import pickle
import time
from .base import BaseAgent
from gemini_utils import embed_text

DEFAULT_TOP_K = 5

logger = logging.getLogger(__name__)

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving relevant text chunks using direct FAISS search."""
    def __init__(self, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        init_start_time = time.time()
        logger.info("Initializing Retriever Agent...")
        logger.info("Loading FAISS index and metadata...")
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            self.texts = self.metadata["texts"]
            self.metadatas = self.metadata["metadatas"]
            # Note: Ensure embeddings were normalized during index creation if needed.
            # faiss.normalize_L2(self.index.reconstruct_n(0, self.index.ntotal))
            logger.info(f"FAISS index & metadata loaded in {time.time() - init_start_time:.2f}s ({len(self.texts)} chunks). Index dimension: {self.index.d}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index/metadata: {e}", exc_info=True)
            self.index = None
            self.texts = []
            self.metadatas = []

    def run(self, query: str, top_k: int = DEFAULT_TOP_K):
        """Retrieves top_k chunks using direct semantic search."""
        run_start_time = time.time()
        logger.info(f"Running direct retrieval for: '{query}' (Top K={top_k})")

        if not self.index or not self.texts:
             logger.error("RetrieverAgent not initialized properly (index or texts missing).")
             return []

        embed_start_time = time.time()
        try:
            query_embedding = embed_text(query)
            if query_embedding is None or len(query_embedding) == 0:
                logger.error("Failed to generate query embedding.")
                return []
            query_embedding_np = np.array([query_embedding]).astype("float32")
            # Note: Normalize query vector if embeddings were normalized during indexing.
            # faiss.normalize_L2(query_embedding_np)
            logger.info(f"Query embedding took: {time.time() - embed_start_time:.4f}s")
        except Exception as e:
            logger.error(f"Error during query embedding: {e}", exc_info=True)
            return []

        faiss_start_time = time.time()
        try:
            distances, indices = self.index.search(query_embedding_np, top_k)
            logger.info(f"FAISS search took: {time.time() - faiss_start_time:.4f}s")
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

        results = []
        if indices.size > 0:
            for i in range(top_k):
                idx = indices[0][i]
                # Check for invalid index (-1 can be returned by FAISS)
                if idx < 0 or idx >= len(self.texts):
                    logger.warning(f"Result {i+1}: Invalid index {idx} returned by FAISS, skipping.")
                    continue

                distance = distances[0][i]
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(distance) # Store the raw distance/score
                })
                logger.debug(f"Retrieved chunk {i+1}: Index={idx}, Score={distance:.4f}, Metadata={self.metadatas[idx]}")

        total_run_time = time.time() - run_start_time
        logger.info(f"Direct retrieval complete in {total_run_time:.4f}s. Retrieved {len(results)} chunks.")
        return results
