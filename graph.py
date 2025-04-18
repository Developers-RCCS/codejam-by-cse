# graph.py
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import json
import time
import faiss             # Add FAISS import
import numpy as np       # Add numpy import
import pickle            # Add pickle import
import functools         # Add functools for cache
from gemini_utils import embed_text # Import embedding function

# --- Load FAISS Index and Metadata --- 
# (Moved from web.py - ensure these files are accessible)
try:
    index = faiss.read_index("faiss_index.index")
    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    texts = metadata["texts"]
    metadatas = metadata["metadatas"]
    FAISS_LOADED = True
except FileNotFoundError:
    print("ERROR: FAISS index or metadata file not found. Retrieval will fail.")
    index = None
    texts = []
    metadatas = []
    FAISS_LOADED = False
# -------------------------------------

# --- Retrieval Function (Moved from web.py) ---
@functools.lru_cache(maxsize=128) # Keep the cache
def search_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]: # Default to low top_k
    """Performs FAISS search and returns relevant chunks."""
    if not FAISS_LOADED or index is None:
        print("Error: FAISS index not loaded, cannot search.")
        return []
    
    try:
        query_embedding = np.array(embed_text(query), dtype="float32").reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(texts): # Basic bounds check
                continue 
            # Store only essential info: text and page number
            page_num = metadatas[idx].get("page", "Unknown")
            results.append({
                "text": texts[idx],
                "metadata": {"page": page_num},
                "score": float(distances[0][i]) # Optionally store score for validation
            })
        return results
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []
# ---------------------------------------------

# 1. Define the Lean State Schema
class AgentState(TypedDict):
    query: str
    analyzed_query: Optional[str] # Keep simple for now
    retrieved_docs: List[Dict[str, Any]] # List of dicts with 'text' and 'metadata' (e.g., 'page')
    final_context: str # Concatenated context for the generator
    raw_answer: str # Direct output from Gemini
    final_answer: str # Formatted answer for display/output
    references_json: str # JSON string of references
    context_csv: str # Snippet of context for submission
    # Optional fields for potential future use or validation
    validation_passed: Optional[bool]
    confidence_score: Optional[float]
    error_message: Optional[str] # For handling failures

# 2. Implement Efficient Query Analyzer Node
def analyze_query_node(state: AgentState) -> AgentState:
    """Analyzes the input query (lightweight)."""
    start_time = time.time()
    print("--- Analyzing Query ---")
    query = state['query']
    
    # --- Lightweight Analysis Logic --- 
    # For now, just pass the query through. 
    # Future: Could add simple keyword extraction or intent check here.
    # Example: Check if query asks for a definition, summary, comparison etc.
    analyzed_query = query # Keep it simple and fast
    # --------------------------------
    
    state['analyzed_query'] = analyzed_query
    print(f"  Analysis Time: {time.time() - start_time:.4f}s")
    return state

# 3. Implement Optimized Retriever Node
def retrieve_documents_node(state: AgentState) -> AgentState:
    """Retrieves documents using the optimized FAISS search."""
    start_time = time.time()
    print("--- Retrieving Documents ---")
    query_to_use = state.get('analyzed_query') or state['query'] # Use analyzed query if available
    
    # --- Call Optimized Retrieval --- 
    # Using the function defined above with caching and optimal top_k
    # You might adjust top_k based on testing
    optimal_top_k = 5 
    retrieved_docs = search_chunks(query_to_use, top_k=optimal_top_k)
    # --------------------------------
    
    state['retrieved_docs'] = retrieved_docs
    print(f"  Retrieved {len(retrieved_docs)} docs in {time.time() - start_time:.4f}s (top_k={optimal_top_k})")
    return state

def validate_context_node(state: AgentState) -> AgentState:
    print("--- Validating Context ---")
    # Simple validation: check if docs were retrieved
    if state['retrieved_docs'] and len(state['retrieved_docs']) > 0:
        state['validation_passed'] = True
    else:
        state['validation_passed'] = False
    return state

def handle_no_context_node(state: AgentState) -> AgentState:
    print("--- Handling No Context ---")
    state['final_answer'] = "I couldn't find relevant information to answer your question based on the provided documents."
    state['references_json'] = json.dumps({})
    state['context_csv'] = "" # Or some indicator of no context
    state['error_message'] = "No relevant documents found."
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    print("--- Generating Answer ---")
    # Placeholder - concatenate context and call Gemini
    state['final_context'] = "\n\n".join([doc['text'] for doc in state['retrieved_docs']])
    # Simulate Gemini call
    state['raw_answer'] = f"Based on the context about pages {[doc['metadata'].get('page', 'N/A') for doc in state['retrieved_docs']]}, the answer to '{state['query']}' is..."
    return state

def track_references_node(state: AgentState) -> AgentState:
    print("--- Tracking References ---")
    # Placeholder - extract references from retrieved_docs used in final_context
    references = {}
    for doc in state['retrieved_docs']:
        page = doc.get('metadata', {}).get('page', 'Unknown')
        references[f"Page {page}"] = doc['text'][:100] + "..." # Example: page and snippet
    state['references_json'] = json.dumps(references, indent=2)
    return state

def format_output_node(state: AgentState) -> AgentState:
    print("--- Formatting Output ---")
    # Simple formatting for now
    state['final_answer'] = state['raw_answer'] # Can add markdown, etc. later
    # Create a simple CSV snippet
    context_list = [f"Page {doc.get('metadata', {}).get('page', 'N/A')}: {doc['text'][:50]}..." for doc in state['retrieved_docs']]
    state['context_csv'] = "\n".join(context_list)
    return state

# Conditional edge logic
def decide_after_retrieval(state: AgentState) -> str:
    print(f"--- Decision: Context Validation {'Passed' if state['validation_passed'] else 'Failed'} ---")
    if state.get('validation_passed', False):
        return "generate_answer"
    else:
        return "handle_no_context"

# Initialize the graph
workflow = StateGraph(AgentState)

# Add nodes (using placeholder functions for now)
workflow.add_node("analyze_query", analyze_query_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("validate_context", validate_context_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("track_references", track_references_node)
workflow.add_node("handle_no_context", handle_no_context_node)
workflow.add_node("format_output", format_output_node) # Final formatting node

# Define edges
workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "validate_context")

# Conditional edge after validation
workflow.add_conditional_edges(
    "validate_context",
    decide_after_retrieval,
    {
        "generate_answer": "generate_answer",
        "handle_no_context": "handle_no_context",
    }
)

workflow.add_edge("generate_answer", "track_references")
workflow.add_edge("track_references", "format_output")
workflow.add_edge("handle_no_context", "format_output") # Route failure path to final formatting
workflow.add_edge("format_output", END) # End the graph after formatting

# Compile the graph (ready to be used)
app_graph = workflow.compile()

# Example of how to run (for testing purposes, will be integrated later)
if __name__ == '__main__':
    # Ensure FAISS is loaded before testing retrieval
    if FAISS_LOADED:
        inputs = {"query": "What was the main cause of the conflict?"}
        
        # --- Test the full graph path up to retrieval/validation ---
        print("\n--- Testing Graph Path (Successful Retrieval Scenario) ---")
        # Temporarily modify retrieve_documents_node to ensure it returns docs for this test if needed
        # Or just run with a query likely to find results
        result = app_graph.invoke(inputs)
        print("\n--- Final State (Successful Retrieval) ---")
        print(result)
        # -----------------------------------------------------------

        # --- Test the no context path ---
        print("\n--- Testing Graph Path (No Context Scenario) ---")
        # Use a query unlikely to find results
        inputs_no_context = {"query": "Tell me about quantum physics in ancient Rome"}
        result_no_context = app_graph.invoke(inputs_no_context)
        print("\n--- Final State (No Context) ---")
        print(result_no_context)
        # ----------------------------------
    else:
        print("\nSkipping graph execution tests because FAISS index could not be loaded.")

    # Visualize the graph (optional, requires graphviz)
    try:
        # Save graph visualization
        # app_graph.get_graph().draw_mermaid_png(output_file_path="graph_visualization.png")
        # print("\nGraph visualization saved to graph_visualization.png")
        pass # Requires playwright install: pip install playwright; playwright install
    except Exception as e:
        print(f"\nCould not generate graph visualization: {e}")
        print("Install graphviz and potentially playwright: pip install pygraphviz playwright; playwright install")