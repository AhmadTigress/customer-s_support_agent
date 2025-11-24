"""
Simple unit tests for main.py
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the codes directory to Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'codes'))

class TestMainComponents:
    """Basic tests for main.py components"""
    
    def test_validate_user_input_safe_input(self):
        """Test input validation with safe input"""
        from main import validate_user_input
        
        is_valid, cleaned = validate_user_input("Hello, how are you?")
        assert is_valid == True
        assert cleaned == "Hello, how are you?"
    
    def test_validate_user_input_dangerous_input(self):
        """Test input validation with dangerous input"""
        from main import validate_user_input
        
        # Test dangerous patterns
        is_valid, cleaned = validate_user_input("exec('rm -rf')")
        assert is_valid == False
        
        is_valid, cleaned = validate_user_input("__import__('os')")
        assert is_valid == False
    
    def test_validate_user_input_too_long(self):
        """Test input validation with long input"""
        from main import validate_user_input
        
        long_text = "a" * 2001
        is_valid, cleaned = validate_user_input(long_text)
        assert is_valid == False
    
    def test_human_request_manager_basic(self):
        """Test HumanRequestManager basic functionality"""
        from main import HumanRequestManager
        
        manager = HumanRequestManager(max_requests=5, request_timeout=10)
        
        # Test adding request
        result = manager.add_request("room1", {"tool_call_id": "test123"})
        assert result == True
        
        # Test retrieving request
        request = manager.get_request("room1")
        assert request is not None
        assert request["tool_call_id"] == "test123"
        
        # Test removing request
        request = manager.get_and_remove_request("room1")
        assert request is not None
        assert manager.get_request("room1") is None
    
    def test_signal_handler_import(self):
        """Test that signal handler function exists"""
        from main import signal_handler
        
        # Just check it's callable
        assert callable(signal_handler)
    
    def test_cleanup_pending_requests_import(self):
        """Test that cleanup function exists"""
        from main import cleanup_pending_requests
        
        # Just check it's callable
        assert callable(cleanup_pending_requests)

class TestMockedComponents:
    """Tests with mocked dependencies"""
    
    @patch('main.MatrixClient')
    @patch('main.PromptManager')
    @patch('main.TigressTechRAG')
    def test_run_bot_initialization(self, mock_rag, mock_prompt, mock_matrix):
        """Test bot initialization with mocked components"""
        from main import run_bot
        
        # Setup mocks
        mock_matrix_instance = Mock()
        mock_matrix.return_value = mock_matrix_instance
        mock_rag_instance = Mock()
        mock_rag_instance.setup_rag.return_value = True
        mock_rag.return_value = mock_rag_instance
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'MATRIX_HOMESERVER': 'test-server',
            'MATRIX_USER': 'test-user', 
            'MATRIX_PASSWORD': 'test-pass',
            'MATRIX_ROOM_ID': 'test-room'
        }):
            # This will start the bot but we'll interrupt it
            with patch('main.matrix_client') as mock_global_client:
                mock_global_client.stop_listening = Mock()
                
                # Mock the listen_for_messages to avoid blocking
                mock_matrix_instance.listen_for_messages = Mock(side_effect=KeyboardInterrupt)
                
                try:
                    run_bot()
                    # If we get here, bot ran without crashing
                    assert True
                except KeyboardInterrupt:
                    # Expected behavior
                    assert True
    
    @patch('main.matrix_client')
    def test_enhanced_process_message_basic(self, mock_client):
        """Test message processing with basic input"""
        from main import enhanced_process_message
        
        # Create mock message
        mock_message = {
            "content": {"body": "Hello bot"},
            "sender": "test_user",
            "room_id": "test_room"
        }
        
        # Mock app
        mock_app = Mock()
        
        # Mock handle_human_decision to return False (not a human response)
        with patch('main.handle_human_decision', return_value=False):
            enhanced_process_message(mock_message, mock_client, mock_app)
            
            # Should not crash with basic input
            assert True

def test_module_imports():
    """Test that all required modules can be imported"""
    try:
        from main import (
            validate_user_input,
            HumanRequestManager,
            handle_human_decision,
            enhanced_process_message,
            cleanup_pending_requests,
            signal_handler,
            run_bot
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import from main.py: {e}")

def test_environment_variables():
    """Test that required environment variables are referenced"""
    import os
    from main import ROOM_ID
    
    # Just check that the variable is accessed
    # Actual value depends on environment
    assert ROOM_ID == os.getenv("MATRIX_ROOM_ID")

if __name__ == "__main__":
    print("Running main.py tests...")
    
    # Run basic tests
    test_instance = TestMainComponents()
    
    try:
        test_instance.test_validate_user_input_safe_input()
        print("✓ Input validation (safe) test passed")
    except Exception as e:
        print(f"✗ Input validation (safe) test failed: {e}")
    
    try:
        test_instance.test_validate_user_input_dangerous_input()
        print("✓ Input validation (dangerous) test passed")
    except Exception as e:
        print(f"✗ Input validation (dangerous) test failed: {e}")
    
    try:
        test_instance.test_human_request_manager_basic()
        print("✓ Human request manager test passed")
    except Exception as e:
        print(f"✗ Human request manager test failed: {e}")
    
    try:
        test_instance.test_signal_handler_import()
        print("✓ Signal handler test passed")
    except Exception as e:
        print(f"✗ Signal handler test failed: {e}")
    
    try:
        test_module_imports()
        print("✓ Module imports test passed")
    except Exception as e:
        print(f"✗ Module imports test failed: {e}")
    
    try:
        test_environment_variables()
        print("✓ Environment variables test passed")
    except Exception as e:
        print(f"✗ Environment variables test failed: {e}")
    
    print("\n🎉 All main.py tests completed!")
