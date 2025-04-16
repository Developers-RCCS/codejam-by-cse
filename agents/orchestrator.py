# agents/orchestrator.py
from .base import BaseAgent
from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .generator import GeneratorAgent
from .reference_tracker import ReferenceTrackerAgent
# Import the new ContextExpansionAgent
from .context_expander import ContextExpansionAgent

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the QA workflow."""
    def __init__(self):
        print("🚀 Initializing Orchestrator and sub-agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.reference_tracker = ReferenceTrackerAgent()
        self.context_expander = ContextExpansionAgent() # Initialize the new agent
        print("✅ Orchestrator ready.")

    def run(self, query: str, chat_history: list = None) -> dict:
        """Runs the full QA pipeline."""
        print(f"\n🔄 Orchestrating response for query: '{query}'")

        # 1. Analyze Query
        query_analysis = self.query_analyzer.run(query=query)
        refined_query = query_analysis["refined_query"]
        keywords = query_analysis["keywords"]

        # TODO: Integrate chat_history into context/query if needed

        # 2. Retrieve Context (Pass keywords to the updated retriever)
        retrieved_chunks = self.retriever.run(query=refined_query, keywords=keywords)

        # 3. Assess & Expand Context (Prompt 5)
        # Pass the retriever agent itself to allow expander access to full data if needed
        final_context_chunks, aggregated_metadata = self.context_expander.run(
            retrieved_chunks=retrieved_chunks,
            query_analysis=query_analysis,
            retriever_agent=self.retriever
        )

        if not final_context_chunks:
            print("⚠️ No relevant context found after retrieval/expansion.")
            return {
                "answer": "I couldn't find relevant information in the textbook to answer that question.",
                "references": {"pages": [], "sections": []},
                "query_analysis": query_analysis,
                "retrieved_chunks": []
            }

        # 4. Generate Answer (using final context)
        # The generator agent internally joins the text from the chunks
        answer = self.generator.run(query=query_analysis["original_query"], context_chunks=final_context_chunks)

        # 5. Track References (using aggregated metadata from expander)
        # Use the metadata aggregated by the ContextExpansionAgent
        references = aggregated_metadata

        # 6. Format Output
        final_answer = answer
        page_refs = references.get("pages", [])
        section_refs = references.get("sections", []) # Get aggregated sections

        ref_string_parts = []
        if page_refs:
            ref_string_parts.append(f"p. {', '.join(map(str, page_refs))}")
        # Optionally add section display logic here if needed
        # if section_refs:
        #     ref_string_parts.append(f"Sections: {', '.join(section_refs)}")

        ref_string = ", ".join(ref_string_parts)

        # Append references if not already included by the generator
        if ref_string and not any(f"[p. {p}]" in final_answer for p in page_refs):
             final_answer += f"\n\n*References: [{ref_string}]*"

        print("✅ Orchestration complete.")
        return {
            "answer": final_answer,
            "references": references, # Return the aggregated references
            "query_analysis": query_analysis,
            "retrieved_chunks": final_context_chunks # Return final chunks used
        }
