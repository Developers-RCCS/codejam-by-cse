# agents/reference_tracker.py
from .base import BaseAgent

class ReferenceTrackerAgent(BaseAgent):
    """Agent responsible for tracking and formatting references."""
    def run(self, context_chunks: list[dict]) -> dict:
        """Extracts and formats page numbers from context chunks."""
        print("🔖 Tracking references...")
        pages = sorted(list(set([chunk["metadata"]["page"] for chunk in context_chunks])))
        sections = [] # Placeholder for future section tracking
        print(f"✅ References tracked: Pages {pages}")
        return {"pages": pages, "sections": sections} # Return structured data
