import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

# Mocking the imports before they are used in bot_nodes
# This prevents the actual RAG or LLM from initializing during tests
with patch('codes.nodes.bot_nodes.TigressTechRAG'), \
     patch('codes.nodes.bot_nodes.PromptManager'), \
     patch('codes.nodes.bot_nodes.EscalationEvaluator'):
    from codes.nodes.bot_nodes import (
        input_node,
        supervisor_node,
        secure_rag_node,
        technical_support_node,
        escalation_check_node
    )

# --- Fixtures ---

@pytest.fixture
def mock_state():
    """Provides a baseline state for node testing."""
    return {
        "user_input": "How do I fix my router?",
        "messages": [],
        "context": "",
        "metadata": MagicMock(turn_count=0),
        "query_type": "technical"
    }

# --- Test Cases ---

def test_input_node_initialization(mock_state):
    """Verifies that input_node increments turns and initializes messages."""
    result = input_node(mock_state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["metadata"].turn_count == 1
    assert result["context"] == ""  #

@patch('codes.nodes.bot_nodes.get_supervisor')
def test_supervisor_routing(mock_get_supervisor, mock_state):
    """Tests if the supervisor correctly sets the next agent."""
    mock_supervisor = MagicMock()
    mock_supervisor.route_request.return_value = {
        "next_node": "technical_agent",
        "intent": "repair"
    }
    mock_get_supervisor.return_value = mock_supervisor

    result = supervisor_node(mock_state)

    assert result["next_agent"] == "technical_agent"
    assert result["query_type"] == "repair"

@patch('codes.nodes.bot_nodes.rag')
def test_secure_rag_retrieval(mock_rag, mock_state):
    """Ensures RAG node updates the state context."""
    mock_rag.query_knowledge_base.return_value = "Mocked policy details."

    result = secure_rag_node(mock_state)

    assert result["context"] == "Mocked policy details."
    mock_rag.query_knowledge_base.assert_called_once_with(mock_state["user_input"])

@patch('codes.nodes.bot_nodes.huggingface_completion')
@patch('codes.nodes.bot_nodes.validate_response')
@patch('codes.nodes.bot_nodes.get_history_str')
def test_technical_support_node(mock_history, mock_validate, mock_hf, mock_state):
    """Tests the full worker cycle: LLM call -> Guardrail -> Result."""
    mock_history.return_value = "User: Hello"
    mock_hf.return_value = {"response": "Try rebooting."}
    mock_validate.return_value = "Safe: Try rebooting."

    result = technical_support_node(mock_state)

    assert result["sender"] == "technical_agent"
    assert result["response"] == "Safe: Try rebooting."
    assert isinstance(result["messages"][0], AIMessage)

@patch('codes.nodes.bot_nodes.escalation_evaluator')
def test_escalation_check_positive(mock_evaluator, mock_state):
    """Verifies escalation detection when the user is frustrated."""
    mock_evaluator.evaluate_escalation_need.return_value = (True, "Frustration", 0.9)

    # Simulate an AI response in state
    mock_state["response"] = "I cannot help you."

    result = escalation_check_node(mock_state)

    assert result["requires_human_escalation"] is True
    assert result["escalation_reason"] == "Frustration"
