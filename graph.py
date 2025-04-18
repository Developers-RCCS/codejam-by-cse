# graph.py
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from agents.query_analyzer import QueryAnalyzerAgent

# Import the AgentState definition
from main import AgentState

# Initialize the query analyzer agent
query_analyzer = QueryAnalyzerAgent()

# Define the query analyzer node function
def analyze_query(state: AgentState) -> AgentState:
    """
    First node in the graph. Analyzes the input query to extract entities,
    keywords, and determine query type and complexity.
    
    Args:
        state: The current state dictionary with at least 'initial_query'
        
    Returns:
        Updated state with 'analyzed_query' field populated
    """
    # Extract the query from the state
    query = state["initial_query"]
    
    # Run the query analyzer
    analyzed_query = query_analyzer.run(query)
    
    # Update the state with analyzed query
    return {
        **state,
        "analyzed_query": analyzed_query
    }

# Initialize the LangGraph workflow
def create_workflow() -> StateGraph:
    """
    Creates and returns the LangGraph workflow with all nodes and edges connected.
    
    Returns:
        A configured StateGraph instance
    """
    # Initialize the workflow with AgentState structure
    workflow = StateGraph(AgentState)
    
    # Add the query analyzer node
    workflow.add_node("query_analyzer", analyze_query)
    
    # Set the entry point
    workflow.set_entry_point("query_analyzer")
    
    # For now, query_analyzer leads to the end (more nodes will be added later)
    workflow.add_edge("query_analyzer", END)
    
    # Compile the workflow
    workflow.compile()
    
    return workflow

# Create a stateful graph instance that can be used by the application
workflow = create_workflow()