import logging
from langgraph.graph import StateGraph, END

# Import your nodes and state
from codes.states.bot_state import AgentState
from codes.nodes.bot_nodes import (
    input_node,
    detect_query_type_node,
    secure_rag_node,
    llm_node,
    supervisor_node,
    output_node
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ROUTING FUNCTION ====================
def route_based_on_query_type(state: AgentState) -> str:
    """
    Determine the processing path based on query type.
    Returns either 'supervisor_path' or 'direct_llm_path'
    """
    query_type = state.get("query_type", "general")
    needs_rag = state.get("needs_rag", True)
    
    # Use supervisor for complex queries that don't need RAG
    # or for specific query types that require special handling
    if not needs_rag or query_type in ["complaint", "report", "technical"]:
        logger.info(f"Routing to supervisor for query type: {query_type}")
        return "supervisor_path"
    else:
        logger.info(f"Routing to direct LLM for query type: {query_type}")
        return "direct_llm_path"

# ==================== GRAPH CONSTRUCTION ====================
logger.info("Building workflow graph...")

workflow = StateGraph(AgentState)

# Add nodes in execution order
workflow.add_node("input_node", input_node)
workflow.add_node("detect_query_type", detect_query_type_node)
workflow.add_node("secure_rag", secure_rag_node)
workflow.add_node("llm_node", llm_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("output_node", output_node)

# Define the flow
workflow.set_entry_point("input_node")
workflow.add_edge("input_node", "detect_query_type")

# Conditional routing after query type detection
workflow.add_conditional_edges(
    "detect_query_type",
    route_based_on_query_type,  # This function must be defined!
    {
        "supervisor_path": "supervisor",
        "direct_llm_path": "secure_rag"
    }
)

# Direct LLM path: secure_rag -> llm_node -> output
workflow.add_edge("secure_rag", "llm_node")
workflow.add_edge("llm_node", "output_node")

# Supervisor path: supervisor -> output (bypasses RAG and direct LLM)
workflow.add_edge("supervisor", "output_node")

# Final output to end
workflow.add_edge("output_node", END)

# Compile graph
app = workflow.compile()
logger.info("Workflow graph compiled successfully")