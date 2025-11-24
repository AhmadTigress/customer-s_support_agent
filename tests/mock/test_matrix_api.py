"""
Simple mock test for matrix_api.py
"""

import pytest
from unittest.mock import Mock, patch
from matrix_api import MatrixClient

def test_matrix_client_creation():
    """Test MatrixClient initialization"""
    client = MatrixClient("homeserver", "user", "password")
    assert client.homeserver == "homeserver"
    assert client.user == "user"
    assert client.access_token is None
    print("✓ Client creation")

@patch('matrix_api.requests.Session')
def test_login_success(mock_session):
    """Test successful login"""
    mock_response = Mock()
    mock_response.json.return_value = {"access_token": "test_token"}
    mock_session.return_value.request.return_value = mock_response
    
    client = MatrixClient("homeserver", "user", "password")
    token = client.login()
    
    assert token == "test_token"
    assert client.access_token == "test_token"
    print("✓ Login success")

@patch('matrix_api.requests.Session')
def test_send_message(mock_session):
    """Test message sending"""
    mock_response = Mock()
    mock_session.return_value.request.return_value = mock_response
    
    client = MatrixClient("homeserver", "user", "password")
    client.access_token = "test_token"
    client.send_message("room123", "Hello")
    
    mock_session.return_value.request.assert_called_once()
    print("✓ Send message")

def test_stop_listening():
    """Test graceful shutdown"""
    client = MatrixClient("homeserver", "user", "password")
    client.stop_listening()
    assert client._shutdown_event.is_set()
    print("✓ Stop listening")

@patch('matrix_api.requests.Session')
def test_context_manager(mock_session):
    """Test context manager usage"""
    mock_response = Mock()
    mock_response.json.return_value = {"access_token": "test_token"}
    mock_session.return_value.request.return_value = mock_response
    
    with MatrixClient("homeserver", "user", "password") as client:
        assert client.access_token == "test_token"
    
    mock_session.return_value.close.assert_called_once()
    print("✓ Context manager")

if __name__ == "__main__":
    print("Testing matrix_api...")
    
    test_matrix_client_creation()
    test_login_success()
    test_send_message()
    test_stop_listening()
    test_context_manager()
    
    print("✓ All matrix_api tests passed")