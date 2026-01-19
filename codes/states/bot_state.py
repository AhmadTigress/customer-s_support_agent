import datetime
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from typing_extensions import Annotated

class ConversationMetadata(BaseModel):
    """Automated validation for metadata to replace manual checks"""
    user_sentiment: str = "neutral"
    complexity_score: float = 0.0
    turn_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    escalation_score: float = 0.0

class AgentState(TypedDict):
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    response: str
    sender: str
    context: str  # RAG context
    query_type: str
    next_agent: str

    # Use the Pydantic model for structured data
    metadata: ConversationMetadata

    requires_human_escalation: bool
    escalation_reason: Optional[str]
    requires_human_decision: bool

def get_history_str(state: AgentState) -> str:
    """
    Centralized helper to derive history from messages.
    Ensures 'Single Source of Truth'.
    """
    messages = state.get("messages", [])
    history = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)
