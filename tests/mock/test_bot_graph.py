"""
Simple mock test for bot_graph.py
"""

import pytest
from unittest.mock import Mock, patch
from codes.graph.bot_graph import create_workflow, route_based_on_query_type, route_after_escalation_check

def test_route_based_on_query_type():
    """Test query type routing logic"""
    # Test supervisor path for complaint
    state = {"query_type": "complaint", "needs_rag": True}
    result = route_based_on_query_type(state)
    assert result == "supervisor_path"
    
    # Test direct path for general
    state = {"query_type": "general", "needs_rag": True}
    result = route_based_on_query_type(state)
    assert result == "direct_llm_path"

def test_route_after_escalation_check():
    """Test escalation routing logic"""
    # Test human escalation
    state = {"requires_human_escalation": True, "escalation_score": 0.8}
    result = route_after_escalation_check(state)
    assert result == "ask_human"
    
    # Test no escalation
    state = {"requires_human_escalation": False, "escalation_score": 0.0}
    result = route_after_escalation_check(state)
    assert result == "output"

@patch('codes.graph.bot_graph.StateGraph')
@patch('codes.graph.bot_graph.MemorySaver')
def test_create_workflow_success(mock_memory, mock_graph):
    """Test workflow creation with mocks"""
    # Mock the graph components
    mock_graph_instance = Mock()
    mock_graph.return_value = mock_graph_instance
    mock_memory_instance = Mock()
    mock_memory.return_value = mock_memory_instance
    
    # Mock node functions
    with patch('codes.graph.bot_graph.input_node'), \
         patch('codes.graph.bot_graph.detect_query_type_node'), \
         patch('codes.graph.bot_graph.secure_rag_node'):
        
        app = create_workflow()
        assert app is not None
        assert mock_graph_instance.add_node.called
        assert mock_graph_instance.add_edge.called

def test_workflow_imports():
    """Test that all required imports are available"""
    from codes.graph.bot_graph import (
        route_based_on_query_type,
        route_after_escalation_check, 
        create_workflow
    )
    assert callable(route_based_on_query_type)
    assert callable(route_after_escalation_check)
    assert callable(create_workflow)

if __name__ == "__main__":
    print("Testing bot_graph...")
    
    test_route_based_on_query_type()
    print("✓ Query routing works")
    
    test_route_after_escalation_check()
    print("✓ Escalation routing works")
    
    test_workflow_imports()
    print("✓ All imports available")
    
    test_create_workflow_success()
    print("✓ Workflow creation works")
    
    print("✓ All bot_graph tests passed")