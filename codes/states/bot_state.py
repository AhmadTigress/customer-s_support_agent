# ==================== STATE DEFINITION ====================
# bot_state.py (state only - no APIs)
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List, add_messages]
    response: str
    sender: str
    context: str
    query_type: str
    needs_rag: bool
    # Add these instead of API instances:
    matrix_room_id: str  # Room ID for Matrix
    requires_human_escalation: bool  # Flag for Matrix escalation