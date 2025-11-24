"""
Simple mock test for huggingface_api.py
"""

import torch
import pytest
from unittest.mock import Mock, patch
from huggingface_api import huggingface_completion

def test_huggingface_completion_empty_prompt():
    """Test with empty prompt"""
    result = huggingface_completion("")
    assert result['status'] == 0
    assert result['response'] == ''

def test_huggingface_completion_valid_prompt():
    """Test with valid prompt using mocks"""
    with patch('huggingface_api.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('huggingface_api.AutoModelForCausalLM.from_pretrained') as mock_model, \
         patch('huggingface_api.pipeline') as mock_pipeline:
        
        # Mock the pipeline response
        mock_pipe_instance = Mock()
        mock_pipe_instance.return_value = [{"generated_text": "Mocked response"}]
        mock_pipeline.return_value = mock_pipe_instance
        
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "eos"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        result = huggingface_completion("test prompt")
        assert result['status'] == 1
        assert 'Mocked response' in result['response']

def test_huggingface_completion_error_handling():
    """Test error handling"""
    with patch('huggingface_api.AutoTokenizer.from_pretrained', side_effect=Exception("Test error")):
        result = huggingface_completion("test prompt")
        assert result['status'] == 0
        assert result['response'] == ''

def test_huggingface_completion_memory_error():
    """Test memory error handling"""
    with patch('huggingface_api.AutoTokenizer.from_pretrained', side_effect=torch.cuda.OutOfMemoryError):
        result = huggingface_completion("test prompt")
        assert result['status'] == 0
        assert result['response'] == ''

def test_function_exists():
    """Test that the main function exists"""
    assert callable(huggingface_completion)

if __name__ == "__main__":
    print("Testing huggingface_api...")
    
    test_huggingface_completion_empty_prompt()
    print("✓ Empty prompt handling")
    
    test_huggingface_completion_valid_prompt()
    print("✓ Valid prompt with mocks")
    
    test_huggingface_completion_error_handling()
    print("✓ Error handling")
    
    test_function_exists()
    print("✓ Function exists")
    
    print("✓ All huggingface_api tests passed")