# ==================== STATE DEFINITION ====================
# bot_state.py (state only - no APIs)
import datetime
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import add_messages

class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List, add_messages]
    response: str
    sender: str
    context: str
    query_type: str
    needs_rag: bool
    matrix_room_id: str  
    requires_human_escalation: bool  
    escalation_reason: Optional[str]
    human_handoff_complete: bool
    pending_human_response:  bool
    conversation_context: Dict[str, Any]
    escalation_score: float
    failed_attempts: int
    user_sentiment: str
    complexity_score: float    
