"""
Simple unit tests for main.py
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from main import production_message_processor, process_with_retry, event_id_cache

@pytest.mark.asyncio
async def test_process_with_retry_success():
    """Verifies that the retry wrapper calls the graph invoke correctly."""
    mock_app = MagicMock()
    mock_app.invoke.return_value = {"response": "Success"}

    input_data = {"user_input": "Hello"}
    config = {"configurable": {"thread_id": "123"}}

    with patch("main.app", mock_app):
        result = await process_with_retry(input_data, config)
        assert result["response"] == "Success"
        assert mock_app.invoke.called

@pytest.mark.asyncio
async def test_idempotency_cache():
    """Ensures the same message event_id is not processed twice."""
    mock_client = MagicMock()
    mock_event = MagicMock()
    mock_event.event_id = "unique_event_123"
    mock_event.sender = "@user:matrix.org"
    mock_event.body = "Hello"

    # Manually add to cache to simulate a previous process
    event_id_cache[mock_event.event_id] = True

    # Call the processor
    await production_message_processor(mock_event, mock_client)

    # The client should NOT have sent a "typing" notice because it skipped processing
    mock_client.room_typing.assert_not_called()

@pytest.mark.asyncio
async def test_message_processor_flow():
    """Tests the full flow from receiving a Matrix message to sending a response."""
    mock_client = MagicMock()
    mock_client.send_text = AsyncMock()
    mock_client.room_typing = AsyncMock()

    mock_event = MagicMock()
    mock_event.event_id = "new_event_456"
    mock_event.room_id = "!room:matrix.org"
    mock_event.body = "Help me"
    mock_event.sender = "@customer:matrix.org"

    # Clear cache for this test
    if mock_event.event_id in event_id_cache:
        del event_id_cache[mock_event.event_id]

    # Mock the graph execution
    mock_result = {"response": "I am Tigra, how can I help?"}

    with patch("main.process_with_retry", AsyncMock(return_value=mock_result)):
        await production_message_processor(mock_event, mock_client)

        # Verify Matrix interactions
        mock_client.room_typing.assert_called_with(mock_event.room_id, typing=True)
        mock_client.send_text.assert_called_with(
            mock_event.room_id,
            "I am Tigra, how can I help?"
        )
        # Verify it was added to cache
        assert mock_event.event_id in event_id_cache
