"""
Simple mock test for bot_nodes.py
"""

import pytest
from unittest.mock import Mock, patch
from codes.nodes.bot_nodes import input_node, detect_query_type_node, output_node

def test_input_node_basic():
    """Test input node with basic state"""
    state = {"user_input": "hello", "sender": "test_user"}
    result = input_node(state)
    assert "messages" in result
    assert "needs_rag" in result
    assert result["requires_human_escalation"] == False

def test_input_node_empty():
    """Test input node with empty input"""
    state = {"user_input": "", "sender": "test_user"}
    result = input_node(state)
    assert "error" in result

@patch('codes.nodes.bot_nodes.prompt_manager')
def test_detect_query_type_node(mock_prompt):
    """Test query type detection with mock"""
    mock_prompt.detect_query_type.return_value = "general"
    state = {"user_input": "test question"}
    result = detect_query_type_node(state)
    assert "query_type" in result
    assert "needs_rag" in result

def test_output_node_basic():
    """Test output node returns state unchanged"""
    state = {"response": "test response", "requires_human_escalation": False}
    result = output_node(state)
    assert result == state

def test_output_node_with_escalation():
    """Test output node with escalation"""
    state = {
        "response": "test", 
        "requires_human_escalation": True,
        "escalation_reason": "complex issue"
    }
    result = output_node(state)
    assert "response" in result

@patch('codes.nodes.bot_nodes.rag')
def test_secure_rag_node_no_rag(mock_rag):
    """Test RAG node when RAG not needed"""
    from codes.nodes.bot_nodes import secure_rag_node
    state = {"needs_rag": False, "user_input": "test"}
    result = secure_rag_node(state)
    assert "context" in result

if __name__ == "__main__":
    print("Testing bot_nodes...")
    
    test_input_node_basic()
    print("✓ Input node basic test")
    
    test_input_node_empty()
    print("✓ Input node empty test")
    
    test_detect_query_type_node()
    print("✓ Query type detection test")
    
    test_output_node_basic()
    print("✓ Output node basic test")
    
    test_output_node_with_escalation()
    print("✓ Output node escalation test")
    
    test_secure_rag_node_no_rag()
    print("✓ RAG node test")
    
    print("✓ All bot_nodes tests passed")