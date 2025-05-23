# agents/context_expander.py
from .base import BaseAgent
import numpy as np
from gemini_utils import embed_text

class ContextExpansionAgent(BaseAgent):
    """Agent responsible for assessing and expanding retrieval context."""

    def assess(self, retrieved_chunks: list[dict]) -> dict:
        """Assess if retrieved context is sufficient."""
        print("🧐 Assessing context sufficiency...")
        
        if not retrieved_chunks:
            print("⚠️ Assessment: No chunks retrieved, expansion needed.")
            return {"needs_expansion": True, "reason": "No chunks retrieved"}
        
        # Check confidence of top chunks
        confidences = [chunk.get("confidence", 0) for chunk in retrieved_chunks]
        avg_confidence = sum(confidences) / len(confidences)
        top_confidence = confidences[0] if confidences else 0
        
        # Calculate context coverage
        total_text_length = sum(len(chunk["text"]) for chunk in retrieved_chunks)
        
        # Separate web and textbook chunks for assessment
        textbook_chunks = [c for c in retrieved_chunks if c["metadata"].get("source_type", "textbook") == "textbook"]
        web_chunks = [c for c in retrieved_chunks if c["metadata"].get("source_type") == "web"]
        
        # Decision logic
        if top_confidence < 0.4:
            print(f"⚠️ Assessment: Low top confidence ({top_confidence:.2f}), expansion needed.")
            return {"needs_expansion": True, "reason": "Low confidence"}
            
        if avg_confidence < 0.3:
            print(f"⚠️ Assessment: Low average confidence ({avg_confidence:.2f}), expansion needed.")
            return {"needs_expansion": True, "reason": "Low average confidence"}
            
        if total_text_length < 500:
            print(f"⚠️ Assessment: Short context ({total_text_length} chars), expansion needed.")
            return {"needs_expansion": True, "reason": "Short context"}
        
        # If we have web chunks but no textbook chunks, expand to try to get textbook context
        if web_chunks and not textbook_chunks:
            print("⚠️ Assessment: Only web chunks available, trying to find textbook content.")
            return {"needs_expansion": True, "reason": "Missing textbook content"}
            
        print(f"✅ Assessment: Context sufficient (Avg conf: {avg_confidence:.2f}, Length: {total_text_length} chars)")
        print(f"   Sources: {len(textbook_chunks)} textbook chunks, {len(web_chunks)} web chunks")
        return {"needs_expansion": False, "reason": "Sufficient confidence and context"}

    def find_contextual_chunks(self, chunks, retriever, max_additional=3):
        """Find chunks that might be contextually related to the given chunks."""
        if not chunks:
            return []
        
        # Only expand textbook chunks using the retriever
        textbook_chunks = [c for c in chunks if c["metadata"].get("source_type", "textbook") == "textbook"]
        if not textbook_chunks:
            print("ℹ️ No textbook chunks to expand from.")
            return []
            
        # Strategy 1: Find adjacent chunks by page numbers
        pages = [chunk["metadata"].get("page", 0) for chunk in textbook_chunks if "metadata" in chunk]
        adjacent_pages = set()
        
        for page in pages:
            if page > 0:
                adjacent_pages.add(page - 1)  # Previous page
                adjacent_pages.add(page + 1)  # Next page
        
        # Filter out pages we already have
        adjacent_pages = adjacent_pages - set(pages)
        
        # Find chunks from adjacent pages
        adjacent_chunks = []
        for i, metadata in enumerate(retriever.metadatas):
            if metadata.get("page", 0) in adjacent_pages:
                adjacent_chunks.append({
                    "text": retriever.texts[i],
                    "metadata": {**metadata, "source_type": "textbook"},  # Ensure source type is set
                    "confidence": 0.4,  # Lower confidence for adjacent chunks
                    "expansion_method": "adjacent_page"
                })
                
        # Strategy 2: Find chunks from same sections
        sections = [chunk["metadata"].get("section", "") for chunk in textbook_chunks if "metadata" in chunk]
        sections = [s for s in sections if s]  # Remove empty sections
        
        section_chunks = []
        if sections:
            for i, metadata in enumerate(retriever.metadatas):
                if metadata.get("section", "") in sections:
                    # Skip if we already have this chunk
                    if any(retriever.texts[i] == c["text"] for c in chunks + adjacent_chunks):
                        continue
                        
                    section_chunks.append({
                        "text": retriever.texts[i],
                        "metadata": {**metadata, "source_type": "textbook"},  # Ensure source type is set
                        "confidence": 0.35,  # Lower confidence for section-based chunks
                        "expansion_method": "same_section"
                    })
        
        # Combine and limit additional chunks
        additional_chunks = (adjacent_chunks + section_chunks)[:max_additional]
        print(f"✅ Found {len(additional_chunks)} additional textbook context chunks.")
        return additional_chunks

    def calculate_chunk_similarity(self, chunks):
        """Calculate similarity between chunks to avoid adding redundant content."""
        if len(chunks) <= 1:
            return []
            
        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            try:
                emb = embed_text(chunk["text"])
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Error embedding chunk: {e}")
                embeddings.append([0] * 768)  # Default empty embedding
                
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                if i != j:
                    similarity = np.dot(embeddings[i], embeddings[j])
                    similarities.append((i, j, similarity))
                    
        return sorted(similarities, key=lambda x: x[2], reverse=True)

    def filter_redundant_chunks(self, chunks, similarity_threshold=0.85):
        """Remove chunks that are too similar to higher-ranked chunks."""
        if len(chunks) <= 1:
            return chunks
            
        # Separate web and textbook chunks for redundancy filtering
        textbook_chunks = [c for c in chunks if c["metadata"].get("source_type", "textbook") == "textbook"]
        web_chunks = [c for c in chunks if c["metadata"].get("source_type") == "web"]
        
        # Filter redundancy within each source type separately
        filtered_textbook = self._filter_source_redundancy(textbook_chunks, similarity_threshold)
        filtered_web = self._filter_source_redundancy(web_chunks, similarity_threshold)
        
        # Combine filtered chunks
        filtered_chunks = filtered_textbook + filtered_web
        
        print(f"✅ Filtered out {len(chunks) - len(filtered_chunks)} redundant chunks.")
        print(f"   Remaining: {len(filtered_textbook)} textbook chunks, {len(filtered_web)} web chunks")
        
        return filtered_chunks
        
    def _filter_source_redundancy(self, chunks, similarity_threshold=0.85):
        """Filter redundancy within a specific source type."""
        if len(chunks) <= 1:
            return chunks
            
        similarities = self.calculate_chunk_similarity(chunks)
        chunks_to_remove = set()
        
        # Mark lower-ranked chunks that are too similar to higher-ranked ones
        for i, j, similarity in similarities:
            if similarity > similarity_threshold:
                # Remove the chunk with lower confidence
                if chunks[i].get("confidence", 0) >= chunks[j].get("confidence", 0):
                    chunks_to_remove.add(j)
                else:
                    chunks_to_remove.add(i)
                    
        # Create filtered list
        filtered_chunks = [chunk for i, chunk in enumerate(chunks) if i not in chunks_to_remove]
        return filtered_chunks

    def fuse_chunks(self, chunks):
        """Fuse chunks into a coherent context, managing token limits."""
        print("🧩 Fusing chunks into coherent context...")
        
        # Sort chunks by confidence
        sorted_chunks = sorted(chunks, key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Separate chunks by source type
        textbook_chunks = []
        web_chunks = []
        
        for chunk in sorted_chunks:
            source_type = chunk["metadata"].get("source_type", "textbook")
            if source_type == "web":
                web_chunks.append(chunk)
            else:
                textbook_chunks.append(chunk)
        
        # Fuse textbook chunks
        textbook_fused = ""
        if textbook_chunks:
            textbook_fused = "**Textbook Content:**"
            for i, chunk in enumerate(textbook_chunks):
                page = chunk["metadata"].get("page", "Unknown")
                section = chunk["metadata"].get("section", "Unknown")
                textbook_fused += f"\n\n--- Excerpt {i+1}: [Page {page}, Section: {section}] ---\n\n"
                textbook_fused += chunk["text"]
        
        # Fuse web chunks
        web_fused = ""
        if web_chunks:
            web_fused = "\n\n**Web Content:**"
            for i, chunk in enumerate(web_chunks):
                url = chunk["metadata"].get("url", "Unknown Source")
                web_fused += f"\n\n--- Web Excerpt {i+1}: [Source: {url}] ---\n\n"
                web_fused += chunk["text"]
        
        # Combine fused text
        fused_text = textbook_fused + web_fused
        
        print(f"✅ Fused {len(textbook_chunks)} textbook chunks and {len(web_chunks)} web chunks into coherent context.")
        return fused_text

    def aggregate_metadata(self, chunks: list[dict]) -> dict:
        """Aggregate metadata from all chunks."""
        print("📊 Aggregating metadata...")
        
        # Extract metadata by source type
        pages = set()
        sections = set()
        web_sources = set()
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_type = metadata.get("source_type", "textbook")
            
            if source_type == "web":
                if "url" in metadata:
                    web_sources.add(metadata["url"])
            else:
                if "page" in metadata and metadata["page"]:
                    pages.add(metadata["page"])
                if "section" in metadata and metadata["section"]:
                    sections.add(metadata["section"])
                
        # Create aggregated metadata dictionary
        aggregated = {
            "pages": sorted(list(pages)),
            "sections": sorted(list(sections)),
            "web_sources": sorted(list(web_sources))
        }
        
        print(f"✅ Metadata aggregated: {len(pages)} pages, {len(sections)} sections, {len(web_sources)} web sources")
        return aggregated

    def run(self, retrieved_chunks: list[dict], query_analysis: dict, retriever_agent) -> tuple[list[dict], dict]:
        """Assess context, expand if needed, filter redundancy, and fuse chunks."""
        # 1. Assess if the context is sufficient
        assessment = self.assess(retrieved_chunks)
        
        final_chunks = retrieved_chunks.copy()
        
        # 2. Expand context if needed
        if assessment["needs_expansion"]:
            print(f"🔍 Expanding context due to: {assessment['reason']}")
            
            # If complex query, consider processing sub-queries separately
            if query_analysis.get("needs_decomposition", False):
                print("📋 Complex query detected, expanding context for multiple aspects.")
                # In a full implementation, we might retrieve for each sub-query
                # For now, just get related chunks to the current results
            
            # Find related chunks (only expands textbook chunks)
            additional_chunks = self.find_contextual_chunks(
                retrieved_chunks, 
                retriever_agent
            )
            
            # Combine original and additional chunks
            expanded_chunks = retrieved_chunks + additional_chunks
            
            # 3. Filter redundant chunks
            final_chunks = self.filter_redundant_chunks(expanded_chunks)
            
            print(f"✅ Context expansion complete: {len(final_chunks)} chunks after filtering.")
        else:
            print("✅ Original context is sufficient, no expansion needed.")
        
        # 4. Aggregate metadata from all included chunks
        aggregated_metadata = self.aggregate_metadata(final_chunks)
        
        # Note: We don't actually fuse the chunks here - that will be handled by the generator
        # when it builds its prompt, using the separate chunks we provide
        
        return final_chunks, aggregated_metadata
