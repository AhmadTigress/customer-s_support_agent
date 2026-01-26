"""
Simple mock test for bot_graph.py
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import HumanMessage, AIMessage
from fastapi.testclient import TestClient

# Import the components to test
from app import app
from nodes.bot_nodes import input_node, supervisor_node, technical_support_node, escalation_check_node
from states.bot_state import AgentState, ConversationMetadata

client = TestClient(app)

# ==================== FIXTURES ====================

@pytest.fixture
def mock_agent_state():
    """Initializes a standard state for testing nodes."""
    return {
        "user_input": "My internet is slow",
        "messages": [],
        "metadata": ConversationMetadata(turn_count=0),
        "context": "",
        "response": ""
    }

# ==================== NODE UNIT TESTS ====================

def test_input_node(mock_agent_state):
    """Verifies that the input node initializes messages and increments turns."""
    result = input_node(mock_agent_state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["metadata"].turn_count == 1
    assert result["context"] == ""  # Ensures RAG leakage protection

@patch("nodes.bot_nodes.get_supervisor")
def test_supervisor_node(mock_get_supervisor, mock_agent_state):
    """Tests if the supervisor correctly routes to a technical agent."""
    mock_sup = MagicMock()
    mock_sup.route_request.return_value = {"next_node": "technical_support_node", "intent": "technical"}
    mock_get_supervisor.return_value = mock_sup

    result = supervisor_node(mock_agent_state)

    assert result["next_agent"] == "technical_support_node"
    assert result["query_type"] == "technical"

@patch("nodes.bot_nodes.huggingface_completion")
@patch("nodes.bot_nodes.validate_response")
def test_technical_support_node(mock_validate, mock_hf, mock_agent_state):
    """Verifies the worker node calls the LLM and validates the output."""
    mock_hf.return_value = {"response": "Please restart your router."}
    mock_validate.return_value = "Validated: Please restart your router."
    mock_agent_state["context"] = "Router troubleshooting guide contents."

    result = technical_support_node(mock_agent_state)

    assert result["sender"] == "tech_support"
    assert "Validated" in result["response"]
    assert isinstance(result["messages"][0], AIMessage)

@patch("nodes.bot_nodes.escalation_evaluator")
def test_escalation_check_node(mock_eval, mock_agent_state):
    """Checks if frustration triggers the escalation flag."""
    mock_eval.evaluate_escalation_need.return_value = (True, "Frustrated user", 0.85)
    mock_agent_state["response"] = "I cannot help you."

    result = escalation_check_node(mock_agent_state)

    assert result["requires_human_escalation"] is True
    assert result["escalation_reason"] == "Frustrated user"
    assert result["metadata"].escalation_score == 0.85

# ==================== GRAPH INTEGRATION TESTS ====================

@patch("nodes.bot_nodes.huggingface_completion")
@patch("nodes.bot_nodes.rag")
@patch("nodes.bot_nodes.get_supervisor")
def test_graph_flow(mock_sup_func, mock_rag, mock_hf):
    """
    Tests the full LangGraph flow from input to output.
    Mocks all external AI services to verify the internal logic.
    """
    from codes.graph.bot_graph import app as workflow

    # Setup Mocks
    mock_sup = MagicMock()
    mock_sup.route_request.return_value = {"next_node": "technical_support_node", "intent": "tech"}
    mock_sup_func.return_value = mock_sup

    mock_rag.query_knowledge_base.return_value = "Mocked Context"
    mock_hf.return_value = {"response": "I solved it!"}

    inputs = {"user_input": "Help with my bill"}
    config = {"configurable": {"thread_id": "test_thread"}}

    # Execute graph
    final_state = workflow.invoke(inputs, config=config)

    assert "response" in final_state
    assert final_state["metadata"].turn_count > 0
    # Verify the supervisor was consulted
    mock_sup.route_request.assert_called()

# ==================== API ENDPOINT TESTS ====================

def test_api_chat_endpoint():
    """Tests the FastAPI /chat endpoint."""
    # Note: We patch the underlying graph call to avoid complex setup
    with patch("app.langgraph_app.ainvoke") as mock_invoke:
        mock_invoke.return_value = {
            "response": "Hello from the API",
            "requires_human_escalation": False,
            "sender": "general_agent"
        }

        response = client.post("/chat", json={"user_input": "Hello", "thread_id": "api_test"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello from the API"
        assert data["thread_id"] == "api_test"

def test_health_check():
    """Verifies the health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
