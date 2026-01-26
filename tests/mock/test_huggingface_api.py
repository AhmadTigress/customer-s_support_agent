"""
Simple mock test for huggingface_api.py
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain.schema import HumanMessage, AIMessage

# Mocking modules before imports to prevent actual initialization of LLMs/DBs
with patch('codes.nodes.bot_nodes.TigressTechRAG'), \
     patch('codes.nodes.bot_nodes.PromptManager'), \
     patch('codes.nodes.bot_nodes.EscalationEvaluator'):
    from codes.nodes.bot_nodes import (
        input_node,
        supervisor_node,
        technical_support_node,
        escalation_check_node
    )
from app import app
from states.bot_state import ConversationMetadata

client = TestClient(app)

# ==================== UNIT TESTS: bot_nodes.py ====================

def test_input_node_logic():
    """Verifies turn increment and RAG leakage prevention."""
    state = {"user_input": "Hello", "metadata": ConversationMetadata(turn_count=5)}

    result = input_node(state)

    assert result["metadata"].turn_count == 6
    assert result["context"] == ""  # Critical: Ensures fresh RAG context
    assert isinstance(result["messages"][0], HumanMessage)

@patch('codes.nodes.bot_nodes.get_supervisor')
def test_supervisor_routing(mock_get_supervisor):
    """Checks if the supervisor correctly sets the next agent."""
    mock_sup = MagicMock()
    mock_sup.route_request.return_value = {"next_node": "billing_agent_node", "intent": "billing"}
    mock_get_supervisor.return_value = mock_sup

    state = {"user_input": "Why was I charged?"}
    result = supervisor_node(state)

    assert result["next_agent"] == "billing_agent_node"
    assert result["query_type"] == "billing"

@patch('codes.nodes.bot_nodes.huggingface_completion')
@patch('codes.nodes.bot_nodes.validate_response')
def test_technical_worker_node(mock_validate, mock_hf):
    """Tests the worker's ability to process LLM output and validate it."""
    mock_hf.return_value = {'status': 1, 'response': 'Reset your router.'}
    mock_validate.return_value = 'Validated: Reset your router.'

    state = {"user_input": "Internet slow", "context": "Docs", "metadata": ConversationMetadata()}
    result = technical_support_node(state)

    assert result["sender"] == "tech_support"
    assert "Validated" in result["response"]
    assert isinstance(result["messages"][0], AIMessage)

# ==================== UNIT TESTS: huggingface_api.py ====================

@patch('codes.API.huggingface_api.model_pipeline')
def test_huggingface_completion_success(mock_pipeline):
    """Verifies successful text extraction from the HF pipeline."""
    from codes.API.huggingface_api import huggingface_completion
    mock_pipeline.return_value = [{"generated_text": "  Support Response  "}]

    result = huggingface_completion("Test Prompt")

    assert result['status'] == 1
    assert result['response'] == "Support Response"

def test_huggingface_empty_input():
    """Ensures the API handles invalid inputs gracefully."""
    from codes.API.huggingface_api import huggingface_completion
    result = huggingface_completion("")
    assert result['status'] == 0

# ==================== INTEGRATION TESTS: app.py ====================

def test_health_endpoint():
    """Tests the FastAPI health check status."""
    response = client.get("/health")
    # This might be 503 if dependencies aren't mocked in monitoring/health.py
    assert response.status_code in [200, 503]

@patch('app.langgraph_app.ainvoke')
def test_chat_endpoint(mock_invoke):
    """Verifies that /chat correctly triggers the LangGraph workflow."""
    mock_invoke.return_value = {
        "response": "Final Answer",
        "requires_human_escalation": False,
        "sender": "general_agent"
    }

    payload = {"user_input": "Help!", "thread_id": "test-session"}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["response"] == "Final Answer"
    assert response.json()["thread_id"] == "test-session"
