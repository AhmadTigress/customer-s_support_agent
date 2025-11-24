# test_matrix_integration.py
import pytest
import os
import time
import threading
from matrix_api import MatrixClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def matrix_client():
    """Fixture to provide a logged-in Matrix client"""
    homeserver = os.getenv("MATRIX_HOMESERVER")
    user = os.getenv("MATRIX_USER")
    password = os.getenv("MATRIX_PASSWORD")
    
    if not all([homeserver, user, password]):
        pytest.skip("Matrix environment variables not set")
    
    client = MatrixClient(homeserver, user, password)
    client.login()
    yield client
    client.stop_listening()
    client._session.close()


@pytest.mark.integration
def test_matrix_login(matrix_client):
    """Test that we can successfully log into Matrix"""
    # The fixture already logs in, so we just verify we have an access token
    assert matrix_client.access_token is not None
    assert len(matrix_client.access_token) > 0
    assert isinstance(matrix_client.access_token, str)


@pytest.mark.integration
def test_matrix_sync_basic(matrix_client):
    """Test basic sync functionality"""
    next_batch, messages = matrix_client.sync_messages()
    
    # Should get a sync token (next_batch)
    assert next_batch is not None
    assert isinstance(next_batch, str)
    assert len(next_batch) > 0
    
    # Messages should be a list
    assert isinstance(messages, list)


@pytest.mark.integration
def test_matrix_context_manager():
    """Test the context manager functionality"""
    homeserver = os.getenv("MATRIX_HOMESERVER")
    user = os.getenv("MATRIX_USER")
    password = os.getenv("MATRIX_PASSWORD")
    
    if not all([homeserver, user, password]):
        pytest.skip("Matrix environment variables not set")
    
    with MatrixClient(homeserver, user, password) as client:
        # Should be logged in automatically
        assert client.access_token is not None
        assert len(client.access_token) > 0
    
    # Client should be cleaned up after context manager exit


@pytest.mark.integration
def test_matrix_get_invited_rooms(matrix_client):
    """Test checking for invited rooms"""
    invited_rooms = matrix_client.get_invited_rooms()
    
    # Should return a list (might be empty if no invitations)
    assert isinstance(invited_rooms, list)


@pytest.mark.integration
def test_matrix_message_listener_start_stop(matrix_client):
    """Test starting and stopping the message listener"""
    # Track if callback was called
    callback_called = threading.Event()
    
    def test_callback(message):
        callback_called.set()
    
    # Start listening in a separate thread
    listener_thread = threading.Thread(
        target=matrix_client.listen_for_messages,
        args=(test_callback,),
        daemon=True
    )
    listener_thread.start()
    
    # Let it run for a moment
    time.sleep(2)
    
    # Stop listening
    matrix_client.stop_listening()
    
    # Wait for thread to finish
    listener_thread.join(timeout=5)
    
    # Thread should have stopped
    assert not listener_thread.is_alive()


@pytest.mark.integration
def test_matrix_request_retry_mechanism(matrix_client):
    """Test that the retry mechanism works for failed requests"""
    # Try to make a request to an invalid endpoint to trigger retries
    with pytest.raises(Exception):  # Should eventually raise after retries
        matrix_client._make_request(
            "GET", 
            f"{matrix_client.homeserver}/_matrix/client/v3/invalid_endpoint"
        )


@pytest.mark.integration
def test_matrix_session_management(matrix_client):
    """Test that session management is working"""
    # Verify session is created and reused
    assert matrix_client._session is not None
    assert hasattr(matrix_client._session, 'get')
    assert hasattr(matrix_client._session, 'post')


@pytest.mark.integration
def test_matrix_shutdown_event(matrix_client):
    """Test the shutdown event functionality"""
    assert not matrix_client._shutdown_event.is_set()
    
    matrix_client.stop_listening()
    
    assert matrix_client._shutdown_event.is_set()


@pytest.mark.integration
def test_matrix_error_count_reset(matrix_client):
    """Test that error count resets on successful requests"""
    initial_count = matrix_client._error_count
    
    # Make a successful request (sync should work)
    matrix_client.sync_messages()
    
    # Error count should be reset to 0 after successful request
    assert matrix_client._error_count == 0