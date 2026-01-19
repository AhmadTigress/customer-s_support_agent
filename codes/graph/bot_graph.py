# codes/graph/bot_graph.py
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import specialized nodes and state
from states.bot_state import AgentState
from nodes.bot_nodes import (
    input_node,
    supervisor_node,
    secure_rag_node,
    technical_support_node,
    billing_agent_node,
    general_inquiry_node,
    escalation_check_node,
    ask_human_node,
    process_human_response_node,
    output_node
)

logger = logging.getLogger(__name__)

# ==================== ROUTING FUNCTIONS ====================

def route_to_specialized_agent(state: AgentState) -> str:
    """
    Routes the graph to the agent chosen by the Supervisor.
    """
    next_agent = state.get("next_agent", "general_inquiry_node")
    logger.info(f"Routing workflow to: {next_agent}")
    return next_agent

def route_after_escalation(state: AgentState) -> str:
    """
    Determines if human-in-the-loop intervention is required.
    """
    if state.get("requires_human_escalation", False) and state.get("escalation_score", 0) > 0.7:
        return "ask_human"
    return "output_node"

# ==================== GRAPH CONSTRUCTION ====================

def create_workflow():
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("input_node", input_node)
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("secure_rag_node", secure_rag_node)

    # Worker Spokes
    workflow.add_node("technical_support_node", technical_support_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("general_inquiry_node", general_inquiry_node)

    # Quality & Human Control
    workflow.add_node("escalation_check_node", escalation_check_node)
    workflow.add_node("ask_human", ask_human_node)
    workflow.add_node("process_human_response", process_human_response_node)
    workflow.add_node("output_node", output_node)

    # 2. Define the Hub-and-Spoke edges
    workflow.set_entry_point("input_node")
    workflow.add_edge("input_node", "supervisor_node")

    # After the Supervisor decides, we fetch RAG context for the chosen agent
    workflow.add_edge("supervisor_node", "secure_rag_node")

    # 3. Conditional Routing from RAG to the chosen Specialist
    workflow.add_conditional_edges(
        "secure_rag_node",
        route_to_specialized_agent,
        {
            "technical_support_node": "technical_support_node",
            "billing_agent_node": "billing_agent_node",
            "general_inquiry_node": "general_inquiry_node",
            "tool_node": "general_inquiry_node" # Placeholder for tool logic
        }
    )

    # 4. All workers flow to the Escalation Check
    workflow.add_edge("technical_support_node", "escalation_check_node")
    workflow.add_edge("billing_agent_node", "escalation_check_node")
    workflow.add_edge("general_inquiry_node", "escalation_check_node")

    # 5. Final output or Human intervention
    workflow.add_conditional_edges(
        "escalation_check_node",
        route_after_escalation,
        {
            "ask_human": "ask_human",
            "output_node": "output_node"
        }
    )

    workflow.add_edge("ask_human", "process_human_response")
    workflow.add_edge("process_human_response", "output_node")
    workflow.add_edge("output_node", END)

    # 6. Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])

app = create_workflow()
