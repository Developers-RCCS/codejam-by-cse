# agents/reference_tracker.py
import logging # Added import
from .base import BaseAgent

logger = logging.getLogger(__name__) # Get a logger for this module

class ReferenceTrackerAgent(BaseAgent):
    """Agent responsible for tracking and formatting references."""
    def run(self, context_chunks: list[dict]) -> dict:
        """Extracts and formats page numbers from context chunks."""
        logger.debug(f"Tracking references for {len(context_chunks)} chunks.")
        print("🔖 Tracking references...")
        pages = sorted(list(set([chunk["metadata"]["page"] for chunk in context_chunks])))
        sections = [] # Placeholder for future section tracking
        logger.debug(f"Reference tracking complete: Pages {pages}, Sections {sections}")
        print(f"✅ References tracked: Pages {pages}")
        return {"pages": pages, "sections": sections} # Return structured data
