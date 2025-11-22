# codes/graph/bot_graph.py
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import nodes and state
from codes.states.bot_state import AgentState
from codes.nodes.bot_nodes import (
    input_node,
    detect_query_type_node,
    secure_rag_node,
    llm_node,
    supervisor_node,
    output_node,
    escalation_check_node,
    ask_human_node,
    process_human_response_node
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ROUTING FUNCTIONS ====================
def route_based_on_query_type(state: AgentState) -> str:
    """
    Determine the processing path based on query type.
    Returns either 'supervisor_path' or 'direct_llm_path'
    """
    try:
        # Use get with default values to avoid KeyErrors
        query_type = state.get("query_type", "general")
        needs_rag = state.get("needs_rag", True)
        
        # Debug logging
        logger.info(f"Routing decision - query_type: {query_type}, needs_rag: {needs_rag}")
        
        # Use supervisor for complex queries that don't need RAG
        # or for specific query types that require special handling
        supervisor_query_types = ["complaint", "report", "technical", "complex"]
        
        if not needs_rag or query_type in supervisor_query_types:
            logger.info(f"Routing to supervisor for query type: {query_type}")
            return "supervisor_path"
        else:
            logger.info(f"Routing to direct LLM for query type: {query_type}")
            return "direct_llm_path"
            
    except Exception as e:
        logger.error(f"Error in route_based_on_query_type: {e}")
        # Default to supervisor path for safety
        return "supervisor_path"


def route_after_escalation_check(state: AgentState) -> str:
    """
    Determine if we need human decision or can proceed directly
    """
    try:
        requires_human_escalation = state.get("requires_human_escalation", False)
        escalation_score = state.get("escalation_score", 0.0)
        
        logger.info(f"Escalation check - requires_human: {requires_human_escalation}, score: {escalation_score}")
        
        # Validate escalation score range
        if not 0.0 <= escalation_score <= 1.0:
            logger.warning(f"Invalid escalation score: {escalation_score}, defaulting to 0.0")
            escalation_score = 0.0
        
        if requires_human_escalation and escalation_score > 0.7:
            logger.info(f"High escalation score ({escalation_score:.2f}) - requesting human decision")
            return "ask_human"
        elif requires_human_escalation:
            logger.info(f"Moderate escalation ({escalation_score:.2f}) - proceeding with automated escalation")
            return "output"
        else:
            logger.info("No escalation needed - proceeding to output")
            return "output"
            
    except Exception as e:
        logger.error(f"Error in route_after_escalation_check: {e}")
        # Default to output for safety
        return "output"


# ==================== GRAPH CONSTRUCTION ====================
def create_workflow():
    """Create and compile the workflow graph with human-in-the-loop"""
    logger.info("Building enhanced workflow graph with human-in-the-loop...")

    try:
        workflow = StateGraph(AgentState)

        # Add nodes in execution order
        workflow.add_node("input_node", input_node)
        workflow.add_node("detect_query_type", detect_query_type_node)
        workflow.add_node("secure_rag", secure_rag_node)
        workflow.add_node("llm_node", llm_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("escalation_check", escalation_check_node)
        workflow.add_node("ask_human", ask_human_node) 
        workflow.add_node("process_human_response", process_human_response_node)  
        workflow.add_node("output_node", output_node)

        # Define the flow
        workflow.set_entry_point("input_node")
        workflow.add_edge("input_node", "detect_query_type")

        # Conditional routing after query type detection
        workflow.add_conditional_edges(
            "detect_query_type",
            route_based_on_query_type,
            {
                "supervisor_path": "supervisor",
                "direct_llm_path": "secure_rag"
            }
        )

        # Direct LLM path: secure_rag -> llm_node -> escalation_check
        workflow.add_edge("secure_rag", "llm_node")
        workflow.add_edge("llm_node", "escalation_check")

        # Supervisor path: supervisor -> escalation_check
        workflow.add_edge("supervisor", "escalation_check")

        # Conditional routing after escalation check
        workflow.add_conditional_edges(
            "escalation_check",
            route_after_escalation_check,
            {
                "ask_human": "ask_human",
                "output": "output_node"
            }
        )

        # Human decision flow: ask_human -> process_human_response -> output
        workflow.add_edge("ask_human", "process_human_response")
        workflow.add_edge("process_human_response", "output_node")

        # Final output to end
        workflow.add_edge("output_node", END)

        # ==================== COMPILE WITH CHECKPOINTING ====================
        # Set up memory for checkpointing with interrupt
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory, 
            interrupt_before=["ask_human"]  # Enables the human-in-the-loop interrupt
        )

        logger.info("Human-in-the-loop workflow compiled successfully with checkpointing")
        return app
        
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise

# Create the workflow application
try:
    app = create_workflow()
    logger.info("Workflow application created successfully")
except Exception as e:
    logger.error(f"Failed to create workflow application: {e}")
    # Create a minimal fallback app or re-raise based on your needs
    raise