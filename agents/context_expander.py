# agents/context_expander.py
from .base import BaseAgent

class ContextExpansionAgent(BaseAgent):
    """Agent responsible for assessing and expanding retrieval context."""

    def assess(self, retrieved_chunks: list[dict], min_confidence: float = 0.5, min_chunks: int = 2) -> dict:
        """Assesses if the retrieved context is sufficient."""
        print("🧐 Assessing context sufficiency...")
        if not retrieved_chunks:
            print("⚠️ Assessment: No chunks retrieved, expansion needed (but impossible).")
            return {"needs_expansion": True, "reason": "No chunks retrieved"}

        # Check average confidence (if available)
        avg_confidence = sum(c.get("confidence", 0) for c in retrieved_chunks) / len(retrieved_chunks)
        if avg_confidence < min_confidence:
            print(f"⚠️ Assessment: Low average confidence ({avg_confidence:.2f} < {min_confidence}), expansion recommended.")
            return {"needs_expansion": True, "reason": f"Low confidence ({avg_confidence:.2f})"}

        # Check number of chunks
        if len(retrieved_chunks) < min_chunks:
            print(f"⚠️ Assessment: Too few chunks ({len(retrieved_chunks)} < {min_chunks}), expansion recommended.")
            return {"needs_expansion": True, "reason": f"Too few chunks ({len(retrieved_chunks)})"}

        # Add more sophisticated checks: topic diversity, coverage of query keywords/entities, etc.

        print("✅ Assessment: Context seems sufficient.")
        return {"needs_expansion": False, "reason": "Sufficient confidence and chunk count"}

    def expand(self, retrieved_chunks: list[dict], query_analysis: dict, all_texts: list, all_metadatas: list, index) -> list[dict]:
        """Expands context by adding adjacent or related chunks (Placeholder)."""
        print("➕ Expanding context (placeholder)...")
        # Placeholder logic: Just return original chunks for now.
        # Future implementation could:
        # 1. Find indices of current chunks.
        # 2. Retrieve chunks with adjacent indices (needs mapping back from FAISS index to original order if shuffled).
        # 3. Perform another semantic search with modified query (e.g., focusing on missing aspects).
        # 4. Be mindful of token limits.
        print("✅ Context expansion complete (no changes made).")
        return retrieved_chunks

    def fuse(self, chunks: list[dict]) -> str:
        """Fuses text from multiple chunks coherently."""
        print("🧩 Fusing chunk texts...")
        # Simple concatenation for now
        fused_text = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])
        print("✅ Text fusion complete.")
        return fused_text

    def aggregate_metadata(self, chunks: list[dict]) -> dict:
        """Aggregates metadata (pages, sections) from multiple chunks."""
        print("📊 Aggregating metadata...")
        pages = sorted(list(set(chunk["metadata"].get("page", -1) for chunk in chunks if "metadata" in chunk))) # Deduplicate and sort
        sections = sorted(list(set(chunk["metadata"].get("section", "Unknown") for chunk in chunks if "metadata" in chunk))) # Deduplicate and sort
        # Filter out default/error values if necessary
        pages = [p for p in pages if p != -1]
        sections = [s for s in sections if s != "Unknown Section" and s != "Unknown"]

        aggregated = {"pages": pages, "sections": sections}
        print(f"✅ Metadata aggregated: Pages={pages}, Sections={len(sections)}")
        return aggregated

    def run(self, retrieved_chunks: list[dict], query_analysis: dict, retriever_agent) -> tuple[list[dict], dict]:
        """Assesses, potentially expands, and aggregates metadata."""
        assessment = self.assess(retrieved_chunks)

        if assessment["needs_expansion"]:
            # Pass necessary components for expansion (full text list, metadata, index)
            expanded_chunks = self.expand(
                retrieved_chunks,
                query_analysis,
                retriever_agent.texts, # Access texts from retriever
                retriever_agent.metadatas, # Access metadatas from retriever
                retriever_agent.index # Access index from retriever
            )
            final_chunks = expanded_chunks
        else:
            final_chunks = retrieved_chunks

        # Aggregate metadata from the final set of chunks
        final_metadata = self.aggregate_metadata(final_chunks)

        return final_chunks, final_metadata
