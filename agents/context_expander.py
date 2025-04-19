# agents/context_expander.py
import logging
import spacy
from .base import BaseAgent
from utils.chunk_utils import filter_redundant_chunks

logger = logging.getLogger(__name__)

class ContextExpansionAgent(BaseAgent):
    """Agent responsible for assessing and expanding retrieval context."""

    def __init__(self):
        super().__init__()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error("❌ Error loading spaCy model 'en_core_web_sm'. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.feedback = defaultdict(list)

    def assess(self, retrieved_chunks: list[dict], query_analysis: dict) -> dict:
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

        # Check if we have entities from query in the chunks
        keywords = query_analysis.get("keywords", [])
        entities = query_analysis.get("entities", [])
        search_terms = set([k.lower() for k in keywords] + [e.lower() for e in entities])
        logger.debug(f"Checking context relevance. Search terms: {search_terms}")

        if not search_terms:
            logger.debug("No keywords/entities found in query analysis, assuming context is relevant.")
            return {"needs_expansion": False, "reason": "No keywords/entities found"}

        found_relevant_chunk = False
        for i, chunk in enumerate(retrieved_chunks):
            text_lower = chunk.get("text", "").lower()
            for term in search_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    logger.debug(f"Found relevant term '{term}' in context chunk {i+1}.")
                    found_relevant_chunk = True
                    break
            if found_relevant_chunk:
                break

        if not found_relevant_chunk:
            logger.warning("No relevant terms found in any context chunk.")
            return {"needs_expansion": True, "reason": "No relevant terms found"}

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

        print(f"✅ Assessment: Context sufficient (Avg conf: {avg_confidence:.2f}, Length: {total_text_length} chars)")
        return {"needs_expansion": False, "reason": "Sufficient confidence and context"}

    def find_contextual_chunks(self, chunks, retriever, max_additional=3):
        """Find chunks that might be contextually related to the given chunks."""
        if not chunks:
            return []

        # Strategy 1: Find adjacent chunks by page numbers
        pages = [chunk["metadata"].get("page", 0) for chunk in chunks if "metadata" in chunk]
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
                    "metadata": metadata,
                    "confidence": 0.4,  # Lower confidence for adjacent chunks
                    "expansion_method": "adjacent_page"
                })

        # Strategy 2: Find chunks from same sections
        sections = [chunk["metadata"].get("section", "") for chunk in chunks if "metadata" in chunk]
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
                        "metadata": metadata,
                        "confidence": 0.35,  # Lower confidence for section-based chunks
                        "expansion_method": "same_section"
                    })

        # Strategy 3: Use advanced NLP techniques to find related chunks
        if self.nlp:
            for chunk in chunks:
                doc = self.nlp(chunk["text"].lower())
                chunk_entities = [ent.text.lower() for ent in doc.ents]
                chunk_keywords = [token.text.lower() for token in doc if token.dep_ in ("nsubj", "dobj", "pobj")]

                for i, metadata in enumerate(retriever.metadatas):
                    text_lower = retriever.texts[i].lower()
                    doc = self.nlp(text_lower)
                    entities = [ent.text.lower() for ent in doc.ents]
                    keywords = [token.text.lower() for token in doc if token.dep_ in ("nsubj", "dobj", "pobj")]

                    if any(term in entities or term in keywords for term in chunk_entities + chunk_keywords):
                        section_chunks.append({
                            "text": retriever.texts[i],
                            "metadata": metadata,
                            "confidence": 0.3,  # Lower confidence for NLP-based chunks
                            "expansion_method": "nlp_related"
                        })

        # Combine and limit additional chunks
        additional_chunks = (adjacent_chunks + section_chunks)[:max_additional]
        print(f"✅ Found {len(additional_chunks)} additional context chunks.")
        return additional_chunks

    def fuse_chunks(self, chunks):
        """Fuse chunks into a coherent context, managing token limits."""
        print("🧩 Fusing chunks into coherent context...")

        # Sort chunks by confidence
        sorted_chunks = sorted(chunks, key=lambda x: x.get("confidence", 0), reverse=True)

        # Get metadata for organization
        chunk_metadata = []
        for chunk in sorted_chunks:
            page = chunk["metadata"].get("page", "Unknown")
            section = chunk["metadata"].get("section", "Unknown")
            chunk_metadata.append(f"[Page {page}, Section: {section}]")

        # Combine text with metadata headers
        fused_text = ""
        for i, chunk in enumerate(sorted_chunks):
            fused_text += f"\n\n--- Excerpt {i+1}: {chunk_metadata[i]} ---\n\n"
            fused_text += chunk["text"]

        print(f"✅ Fused {len(sorted_chunks)} chunks into coherent context.")
        return fused_text

    def aggregate_metadata(self, chunks: list[dict]) -> dict:
        """Aggregate metadata from all chunks."""
        print("📊 Aggregating metadata...")

        # Extract page numbers
        pages = set()
        sections = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if "page" in metadata and metadata["page"]:
                pages.add(metadata["page"])
            if "section" in metadata and metadata["section"]:
                sections.add(metadata["section"])

        aggregated = {
            "pages": sorted(list(pages)),
            "sections": sorted(list(sections))
        }

        print(f"✅ Metadata aggregated: {len(pages)} pages, {len(sections)} sections")
        return aggregated

    def _update_feedback(self, query: str, context_chunks: list[dict], relevance: bool):
        """Update feedback loop with user interaction data."""
        self.feedback[query].append({
            "context_chunks": context_chunks,
            "relevance": relevance
        })
        logger.debug(f"Feedback updated for query: '{query}' with relevance: {relevance}")

    def run(self, retrieved_chunks: list[dict], query_analysis: dict, retriever_agent) -> tuple[list[dict], dict]:
        """Assess context, expand if needed, filter redundancy, and fuse chunks."""
        logger.debug(f"Running context expansion/filtering on {len(retrieved_chunks)} chunks.")
        # 1. Assess if the context is sufficient
        assessment = self.assess(retrieved_chunks, query_analysis)

        final_chunks = retrieved_chunks.copy()

        # 2. Expand context if needed
        if assessment["needs_expansion"]:
            print(f"🔍 Expanding context due to: {assessment['reason']}")

            # If complex query, consider processing sub-queries separately
            if query_analysis.get("needs_decomposition", False):
                print("📋 Complex query detected, expanding context for multiple aspects.")
                # In a full implementation, we might retrieve for each sub-query
                # For now, just get related chunks to the current results

            # Find related chunks
            additional_chunks = self.find_contextual_chunks(
                retrieved_chunks,
                retriever_agent
            )

            # Combine original and additional chunks
            expanded_chunks = retrieved_chunks + additional_chunks

            # 3. Filter redundant chunks using the utility function
            final_chunks = filter_redundant_chunks(expanded_chunks)

            print(f"✅ Context expansion complete: {len(final_chunks)} chunks after filtering.")
        else:
            print("✅ Original context is sufficient, no expansion needed.")
            # Still filter original chunks for redundancy
            final_chunks = filter_redundant_chunks(retrieved_chunks)

        # 4. Aggregate metadata from all included chunks
        aggregated_metadata = self.aggregate_metadata(final_chunks)

        logger.debug(f"Context expansion complete. Final chunks: {len(final_chunks)}")
        # Note: We don't actually fuse the chunks here - that will be handled by the generator
        # when it builds its prompt, using the separate chunks we provide

        return final_chunks, aggregated_metadata
