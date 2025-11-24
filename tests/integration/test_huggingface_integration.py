# test_huggingface_integration.py
import pytest
from huggingface_api import huggingface_completion


@pytest.mark.integration
def test_huggingface_completion_basic():
    """Test basic text completion functionality"""
    # Test with a simple prompt
    prompt = "What is the capital of France?"
    
    result = huggingface_completion(prompt)
    
    # Check that the API call was successful
    assert result['status'] == 1, f"API call failed. Response: {result}"
    
    # Check that we got a non-empty response
    assert result['response'] != "", "Response should not be empty"
    assert len(result['response']) > 0, "Response should have content"
    
    # Basic sanity check - response should contain relevant information
    # Note: We use case-insensitive check since the model might respond in different cases
    assert "paris" in result['response'].lower(), f"Expected 'Paris' in response, got: {result['response']}"


@pytest.mark.integration
def test_huggingface_completion_empty_prompt():
    """Test handling of empty prompts"""
    result = huggingface_completion("")
    
    # Should fail gracefully with status 0
    assert result['status'] == 0, "Empty prompt should return status 0"
    assert result['response'] == "", "Empty prompt should return empty response"


@pytest.mark.integration
def test_huggingface_completion_whitespace_prompt():
    """Test handling of whitespace-only prompts"""
    result = huggingface_completion("   ")
    
    # Should fail gracefully with status 0
    assert result['status'] == 0, "Whitespace prompt should return status 0"
    assert result['response'] == "", "Whitespace prompt should return empty response"


@pytest.mark.integration
def test_huggingface_completion_simple_question():
    """Test with another simple question to verify consistency"""
    prompt = "What is 2 + 2?"
    
    result = huggingface_completion(prompt)
    
    # Check successful response
    assert result['status'] == 1, f"API call failed. Response: {result}"
    assert result['response'] != "", "Response should not be empty"
    
    # The response should contain the answer (might be in different formats)
    response_lower = result['response'].lower()
    # Check for common ways the answer might appear
    assert any(keyword in response_lower for keyword in ['4', 'four']), \
        f"Expected answer about '4' in response, got: {result['response']}"


@pytest.mark.integration
def test_huggingface_completion_longer_prompt():
    """Test with a slightly longer prompt"""
    prompt = """Please write a very short greeting message. 
    Keep it to one sentence only."""
    
    result = huggingface_completion(prompt)
    
    # Check successful response
    assert result['status'] == 1, f"API call failed. Response: {result}"
    assert result['response'] != "", "Response should not be empty"
    
    # Basic check for greeting-like content
    response_lower = result['response'].lower()
    greeting_indicators = ['hello', 'hi', 'greeting', 'welcome', 'hey']
    assert any(indicator in response_lower for indicator in greeting_indicators), \
        f"Expected greeting in response, got: {result['response']}"