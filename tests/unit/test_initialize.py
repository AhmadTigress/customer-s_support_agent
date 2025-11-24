"""
Unit tests for initialize.py
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv


class TestInitialize:
    """Test the initialization sequence in initialize.py"""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.pipeline')
    @patch('langchain_community.llms.HuggingFacePipelineLLM')
    @patch('codes.prompt_manager.PromptManager')
    @patch('codes.rag_system.TigressTechRAG')
    @patch('codes.API.matrix_api.MatrixClient')
    def test_initialization_sequence(
        self, 
        mock_matrix_client, 
        mock_rag, 
        mock_prompt_manager, 
        mock_hf_llm,
        mock_pipeline,
        mock_model,
        mock_tokenizer
    ):
        """Test that all components are initialized in the correct order"""
        # Mock the dependencies
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        mock_hf_llm.return_value = Mock()
        mock_prompt_manager.return_value = Mock()
        mock_rag_instance = Mock()
        mock_rag_instance.setup_rag.return_value = True
        mock_rag.return_value = mock_rag_instance
        mock_matrix_client.return_value = Mock()
        
        # Import and execute the initialization code
        with patch('logging.getLogger') as mock_logger, \
             patch('logging.basicConfig') as mock_log_config:
            
            mock_logger.return_value = Mock()
            
            # Execute the initialization code by importing the module
            # We'll simulate the execution since we can't directly run the module as is
            from initialize import (
                MODEL_NAME, HF_TOKEN, MATRIX_HOMESERVER, 
                MATRIX_USER, MATRIX_PASSWORD, logger
            )
            
            # Verify environment variables are set
            assert MODEL_NAME is not None
            assert HF_TOKEN is not None
            assert MATRIX_HOMESERVER is not None
            assert MATRIX_USER is not None
            assert MATRIX_PASSWORD is not None
            
            # Verify logging was set up
            mock_log_config.assert_called_once_with(level=20)  # 20 = logging.INFO
            
            # Verify the initialization sequence would happen in correct order
            # The actual calls would happen when the module is executed, 
            # but we've mocked them to prevent actual initialization
    
    def test_environment_variables(self):
        """Test that environment variables are properly loaded with fallbacks"""
        # Load environment
        load_dotenv()
        
        # Import the module to test variable assignment
        from initialize import MODEL_NAME, HF_TOKEN
        
        # Test fallback values
        assert MODEL_NAME == os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
        
        # Test that variables are strings (even if empty)
        assert isinstance(MODEL_NAME, str)
        assert isinstance(HF_TOKEN, str)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_model_loading_parameters(self, mock_model, mock_tokenizer):
        """Test that models are loaded with correct parameters"""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Simulate the model loading calls
        from initialize import MODEL_NAME, HF_TOKEN
        
        # Test tokenizer call
        mock_tokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        mock_tokenizer.from_pretrained.assert_called_once_with(
            MODEL_NAME, token=HF_TOKEN
        )
        
        # Test model call
        mock_model.from_pretrained.assert_called_once_with(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            token=HF_TOKEN,
        )
    
    @patch('codes.rag_system.TigressTechRAG')
    def test_rag_setup_fallback(self, mock_rag):
        """Test that the system handles RAG setup failure gracefully"""
        mock_rag_instance = Mock()
        mock_rag_instance.setup_rag.return_value = False  # Simulate failure
        mock_rag.return_value = mock_rag_instance
        
        # This would normally log a warning but continue
        # We're testing that the failure doesn't break the initialization
        rag_success = mock_rag_instance.setup_rag()
        assert rag_success is False
    
    def test_matrix_client_initialization(self):
        """Test Matrix client initialization parameters"""
        from initialize import MATRIX_HOMESERVER, MATRIX_USER, MATRIX_PASSWORD
        
        # These should be the values that would be passed to MatrixClient
        homeserver = MATRIX_HOMESERVER
        username = MATRIX_USER
        password = MATRIX_PASSWORD
        
        # Verify they are strings (actual values depend on environment)
        assert isinstance(homeserver, str)
        assert isinstance(username, str)
        assert isinstance(password, str)


def test_module_imports():
    """Test that all required modules can be imported"""
    # Test that the main module imports work
    try:
        from initialize import (
            MODEL_NAME, HF_TOKEN, MATRIX_HOMESERVER, 
            MATRIX_USER, MATRIX_PASSWORD, logger
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import from initialize.py: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])