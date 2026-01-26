"""
Simple mock test for supervisor.py
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain.schema import HumanMessage, AIMessage

# Mocking internal modules before imports to prevent actual initialization of LLMs/DBs
with patch('codes.nodes.bot_nodes.TigressTechRAG'), \
     patch('codes.nodes.bot_nodes.PromptManager'), \
     patch('codes.nodes.bot_nodes.EscalationEvaluator'):
    from codes.nodes.bot_nodes import (
        input_node,
        supervisor_node,
        technical_support_node,
        escalation_check_node
    )
from supervisor import Supervisor
from app import app
from states.bot_state import ConversationMetadata

client = TestClient(app)

# ==================== UNIT TESTS: supervisor.py ====================

@patch('supervisor.huggingface_completion')
def test_supervisor_routing_logic(mock_hf):
    """Verifies the Supervisor correctly classifies intent and routes."""
    # Setup
    mock_prompt_mgr = MagicMock()
    mock_rag = MagicMock()
    supervisor = Supervisor(mock_prompt_mgr, mock_rag)

    # Test Technical Routing
    mock_hf.return_value = {'response': 'TECHNICAL'}
    result = supervisor.route_request("My router is broken")
    assert result["next_node"] == "technical_support_node"

    # Test Billing Routing
    mock_hf.return_value = {'response': 'BILLING'}
    result = supervisor.route_request("Why was I charged twice?")
    assert result["next_node"] == "billing_agent_node"

# ==================== UNIT TESTS: bot_nodes.py ====================

def test_input_node_clears_context():
    """Ensures input_node prevents RAG leakage by resetting context."""
    state = {
        "user_input": "Help",
        "context": "Old Data",
        "metadata": ConversationMetadata(turn_count=1)
    }
    result = input_node(state)

    assert result["context"] == ""
    assert result["metadata"].turn_count == 2
    assert isinstance(result["messages"][0], HumanMessage)

@patch('codes.nodes.bot_nodes.huggingface_completion')
@patch('codes.nodes.bot_nodes.validate_response')
def test_technical_worker_node(mock_validate, mock_hf):
    """Tests the worker's ability to process LLM output and validate it."""
    mock_hf.return_value = {'status': 1, 'response': 'Reboot it.'}
    mock_validate.return_value = 'Safe: Reboot it.'

    state = {
        "user_input": "Slow net",
        "context": "Manual text",
        "messages": [],
        "metadata": ConversationMetadata()
    }
    result = technical_support_node(state)

    assert result["sender"] == "tech_support"
    assert result["response"] == "Safe: Reboot it."
    assert isinstance(result["messages"][0], AIMessage)

# ==================== UNIT TESTS: huggingface_api.py ====================

@patch('codes.API.huggingface_api.model_pipeline')
def test_huggingface_completion_success(mock_pipeline):
    """Verifies extraction logic from the HF text-generation pipeline."""
    from codes.API.huggingface_api import huggingface_completion
    mock_pipeline.return_value = [{"generated_text": "  Clean Response  "}]

    result = huggingface_completion("Prompt")

    assert result['status'] == 1
    assert result['response'] == "Clean Response"

# ==================== INTEGRATION TESTS: app.py ====================

def test_health_endpoint():
    """Verifies health check status code from app.py."""
    # Since health.py isn't provided, we check for standard responses
    response = client.get("/health")
    assert response.status_code in [200, 503]

@patch('app.langgraph_app.ainvoke')
def test_chat_endpoint_success(mock_invoke):
    """Verifies the FastAPI /chat endpoint triggers the graph and returns JSON."""
    mock_invoke.return_value = {
        "response": "Final Answer",
        "requires_human_escalation": False,
        "sender": "tech_support"
    }

    thread_id = str(uuid.uuid4())
    payload = {"user_input": "I need help", "thread_id": thread_id}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Final Answer"
    assert data["thread_id"] == thread_id
    assert data["metadata"]["sender"] == "tech_support"
