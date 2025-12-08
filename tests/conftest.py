"""Test configuration and fixtures for bruno-llm tests."""

import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock

from bruno_core.models import Message, MessageRole


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello!"),
    ]


@pytest.fixture
def mock_response():
    """Mock LLM response."""
    return "Hello! How can I help you today?"


@pytest.fixture
async def mock_stream_response():
    """Mock streaming response."""
    async def _stream():
        for chunk in ["Hello", "! ", "How ", "can ", "I ", "help?"]:
            yield chunk
    return _stream
