"""
Simple mock test for supervisor.py
"""

import pytest
from unittest.mock import Mock, patch
from supervisor import Supervisor

def test_supervisor_creation():
    """Test Supervisor initialization"""
    mock_prompt = Mock()
    mock_rag = Mock()
    supervisor = Supervisor(mock_prompt, mock_rag)
    assert supervisor.prompt_manager == mock_prompt
    assert supervisor.rag_system == mock_rag
    print("✓ Supervisor creation")

@patch('supervisor.huggingface_completion')
@patch('supervisor.get_all_tools')
def test_handle_query_valid(mock_tools, mock_llm):
    """Test handle_query with valid input"""
    mock_prompt = Mock()
    mock_rag = Mock()
    mock_prompt.detect_query_type.return_value = "general"
    mock_rag.query_knowledge_base.return_value = "test context"
    mock_llm.return_value = {'status': 1, 'response': 'test response'}
    
    supervisor = Supervisor(mock_prompt, mock_rag)
    result = supervisor.handle_query("test question")
    
    assert result['status'] == 'test response'
    assert result['agent_type'] == 'general'
    print("✓ Handle query valid")

def test_handle_query_empty():
    """Test handle_query with empty input"""
    mock_prompt = Mock()
    mock_rag = Mock()
    supervisor = Supervisor(mock_prompt, mock_rag)
    result = supervisor.handle_query("")
    
    assert "Please provide a valid input" in result['response']
    print("✓ Handle query empty")

def test_is_calculation_request():
    """Test calculation detection"""
    mock_prompt = Mock()
    mock_rag = Mock()
    supervisor = Supervisor(mock_prompt, mock_rag)
    
    assert supervisor._is_calculation_request("calculate 2+2") == True
    assert supervisor._is_calculation_request("hello") == False
    print("✓ Calculation detection")

def test_is_appointment_request():
    """Test appointment detection"""
    mock_prompt = Mock()
    mock_rag = Mock()
    supervisor = Supervisor(mock_prompt, mock_rag)
    
    assert supervisor._is_appointment_request("schedule appointment") == True
    assert supervisor._is_appointment_request("hello") == False
    print("✓ Appointment detection")

@patch('supervisor.calculator')
def test_try_use_tools_calc(mock_calc):
    """Test tool usage for calculator"""
    mock_prompt = Mock()
    mock_rag = Mock()
    mock_calc.invoke.return_value = "4"
    
    supervisor = Supervisor(mock_prompt, mock_rag)
    result = supervisor._try_use_tools("calculate 2+2")
    
    assert "Calculation result" in result
    print("✓ Tool usage calculator")

if __name__ == "__main__":
    print("Testing supervisor...")
    
    test_supervisor_creation()
    test_handle_query_valid()
    test_handle_query_empty()
    test_is_calculation_request()
    test_is_appointment_request()
    test_try_use_tools_calc()
    
    print("✓ All supervisor tests passed")