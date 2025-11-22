# ==================== STATE DEFINITION ====================
# bot_state.py (state only - no APIs)
import datetime
from typing import TypedDict, List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator
from typing_extensions import Annotated

class AgentState(TypedDict):
    # User input and conversation
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    response: str
    sender: str
    context: str
    query_type: str
    needs_rag: bool
    
    # Matrix integration
    matrix_room_id: str
    
    # Human escalation system
    requires_human_escalation: bool
    escalation_reason: Optional[str]
    human_handoff_complete: bool
    pending_human_response: bool
    
    # Conversation management
    conversation_context: Dict[str, Any]
    escalation_score: float
    failed_attempts: int
    user_sentiment: str
    complexity_score: float
    
    # Timing and metadata
    timestamp: str
    conversation_id: Optional[str]
    turn_count: int

# Default state initialization
def create_default_state() -> AgentState:
    """Create a properly initialized default state"""
    return {
        # User input and conversation
        "user_input": "",
        "messages": [],
        "response": "",
        "sender": "",
        "context": "",
        "query_type": "general",
        "needs_rag": True,
        
        # Matrix integration
        "matrix_room_id": "",
        
        # Human escalation system
        "requires_human_escalation": False,
        "escalation_reason": None,
        "human_handoff_complete": False,
        "pending_human_response": False,
        
        # Conversation management
        "conversation_context": {},
        "escalation_score": 0.0,
        "failed_attempts": 0,
        "user_sentiment": "neutral",
        "complexity_score": 0.0,
        
        # Timing and metadata
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": None,
        "turn_count": 0
    }

# State validation functions
def validate_state(state: AgentState) -> bool:
    """Validate state integrity"""
    try:
        # Required string fields should not be None
        required_strings = [
            state["user_input"],
            state["response"], 
            state["sender"],
            state["context"],
            state["query_type"],
            state["matrix_room_id"],
            state["user_sentiment"]
        ]
        
        # All required strings should be actual strings
        for field in required_strings:
            if not isinstance(field, str):
                return False
        
        # Boolean fields validation
        required_bools = [
            state["needs_rag"],
            state["requires_human_escalation"],
            state["human_handoff_complete"],
            state["pending_human_response"]
        ]
        
        for field in required_bools:
            if not isinstance(field, bool):
                return False
        
        # Numeric fields validation
        if not isinstance(state["escalation_score"], (int, float)):
            return False
        if not isinstance(state["failed_attempts"], int):
            return False
        if not isinstance(state["complexity_score"], (int, float)):
            return False
        if not isinstance(state["turn_count"], int):
            return False
            
        # List and Dict validation
        if not isinstance(state["messages"], list):
            return False
        if not isinstance(state["conversation_context"], dict):
            return False
            
        return True
        
    except (KeyError, TypeError):
        return False

def update_state_turn(state: AgentState) -> AgentState:
    """Update state for new conversation turn"""
    state["turn_count"] += 1
    state["timestamp"] = datetime.datetime.now().isoformat()
    state["failed_attempts"] = 0  # Reset for new turn
    return state

def reset_escalation_state(state: AgentState) -> AgentState:
    """Reset escalation-related fields"""
    state["requires_human_escalation"] = False
    state["escalation_reason"] = None
    state["human_handoff_complete"] = False
    state["pending_human_response"] = False
    state["escalation_score"] = 0.0
    return state

# Type aliases for better code clarity
StateUpdate = Dict[str, Any]
ConversationTurn = Dict[str, Union[str, List[BaseMessage]]]