"""Tests for Ollama provider."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from bruno_core.models import Message, MessageRole

from bruno_llm.exceptions import (
    LLMError,
    ModelNotFoundError,
    StreamError,
    InvalidResponseError,
)
from bruno_llm.providers.ollama import OllamaConfig, OllamaProvider


@pytest.fixture
def ollama_config():
    """Create test Ollama configuration."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="llama2",
        timeout=30.0,
    )


@pytest.fixture
def ollama_provider():
    """Create test Ollama provider."""
    return OllamaProvider(model="llama2")


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
    ]


def test_ollama_config_default():
    """Test default Ollama configuration."""
    config = OllamaConfig()
    
    assert config.base_url == "http://localhost:11434"
    assert config.model == "llama2"
    assert config.timeout == 30.0
    assert config.temperature == 0.7


def test_ollama_config_custom():
    """Test custom Ollama configuration."""
    config = OllamaConfig(
        base_url="http://192.168.1.100:11434",
        model="mistral",
        temperature=0.5,
        top_p=0.8,
    )
    
    assert config.base_url == "http://192.168.1.100:11434"
    assert config.model == "mistral"
    assert config.temperature == 0.5
    assert config.top_p == 0.8


def test_ollama_provider_init():
    """Test Ollama provider initialization."""
    provider = OllamaProvider(model="llama2")
    
    assert provider.model == "llama2"
    assert provider.config.base_url == "http://localhost:11434"


def test_ollama_provider_format_messages(ollama_provider, sample_messages):
    """Test message formatting."""
    formatted = ollama_provider._format_messages(sample_messages)
    
    assert len(formatted) == 2
    assert formatted[0] == {"role": "system", "content": "You are helpful"}
    assert formatted[1] == {"role": "user", "content": "Hello"}


def test_ollama_provider_build_request(ollama_provider, sample_messages):
    """Test request building."""
    request = ollama_provider._build_request(sample_messages, stream=False)
    
    assert request["model"] == "llama2"
    assert request["stream"] is False
    assert len(request["messages"]) == 2
    assert "options" in request


@pytest.mark.asyncio
async def test_generate_success(ollama_provider, sample_messages):
    """Test successful generation."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Hello! How can I help you?"}
    }
    
    with patch.object(ollama_provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        response = await ollama_provider.generate(sample_messages)
        
        assert response == "Hello! How can I help you?"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_generate_model_not_found(ollama_provider, sample_messages):
    """Test generation with model not found."""
    mock_response = Mock()
    mock_response.status_code = 404
    
    with patch.object(ollama_provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError(
            "Not found",
            request=Mock(),
            response=mock_response,
        )
        
        with pytest.raises(ModelNotFoundError):
            await ollama_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_timeout(ollama_provider, sample_messages):
    """Test generation timeout."""
    from bruno_llm.exceptions import TimeoutError as LLMTimeoutError
    
    with patch.object(ollama_provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        
        with pytest.raises(LLMTimeoutError):
            await ollama_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_connection_error(ollama_provider, sample_messages):
    """Test generation connection error."""
    with patch.object(ollama_provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection failed")
        
        with pytest.raises(LLMError):
            await ollama_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_invalid_response(ollama_provider, sample_messages):
    """Test generation with invalid response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Invalid"}
    
    with patch.object(ollama_provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with pytest.raises(InvalidResponseError):
            await ollama_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_stream_success(ollama_provider, sample_messages):
    """Test successful streaming."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    
    # Simulate streaming chunks
    chunks = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " there"}, "done": False}),
        json.dumps({"message": {"content": "!"}, "done": False}),
        json.dumps({"done": True}),
    ]
    
    async def mock_aiter_lines():
        for chunk in chunks:
            yield chunk
    
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = Mock()
    
    with patch.object(ollama_provider._client, "stream") as mock_stream:
        mock_stream.return_value.__aenter__.return_value = mock_response
        
        result = []
        async for chunk in ollama_provider.stream(sample_messages):
            result.append(chunk)
        
        assert result == ["Hello", " there", "!"]


@pytest.mark.asyncio
async def test_stream_model_not_found(ollama_provider, sample_messages):
    """Test streaming with model not found."""
    mock_response = Mock()
    mock_response.status_code = 404
    
    with patch.object(ollama_provider._client, "stream") as mock_stream:
        mock_stream.return_value.__aenter__.side_effect = httpx.HTTPStatusError(
            "Not found",
            request=Mock(),
            response=mock_response,
        )
        
        with pytest.raises(ModelNotFoundError):
            async for _ in ollama_provider.stream(sample_messages):
                pass


@pytest.mark.asyncio
async def test_list_models_success(ollama_provider):
    """Test listing models."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
            {"name": "codellama"},
        ]
    }
    
    with patch.object(ollama_provider._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        models = await ollama_provider.list_models()
        
        assert models == ["llama2", "mistral", "codellama"]
        mock_get.assert_called_once_with("/api/tags")


@pytest.mark.asyncio
async def test_list_models_error(ollama_provider):
    """Test listing models with error."""
    with patch.object(ollama_provider._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.RequestError("Connection failed")
        
        with pytest.raises(LLMError):
            await ollama_provider.list_models()


@pytest.mark.asyncio
async def test_check_connection_success(ollama_provider):
    """Test successful connection check."""
    mock_response = Mock()
    mock_response.status_code = 200
    
    with patch.object(ollama_provider._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        result = await ollama_provider.check_connection()
        
        assert result is True


@pytest.mark.asyncio
async def test_check_connection_failure(ollama_provider):
    """Test failed connection check."""
    with patch.object(ollama_provider._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.RequestError("Connection failed")
        
        result = await ollama_provider.check_connection()
        
        assert result is False


def test_get_token_count(ollama_provider):
    """Test token counting."""
    text = "Hello world"
    tokens = ollama_provider.get_token_count(text)
    
    assert tokens >= 2  # Should have at least 2 tokens


def test_get_model_info(ollama_provider):
    """Test getting model info."""
    info = ollama_provider.get_model_info()
    
    assert info["provider"] == "ollama"
    assert info["model"] == "llama2"
    assert "base_url" in info
    assert "temperature" in info


@pytest.mark.asyncio
async def test_context_manager():
    """Test provider as context manager."""
    async with OllamaProvider(model="llama2") as provider:
        assert provider.model == "llama2"
    
    # Client should be closed after context exit
    # (httpx raises RuntimeError if used after close)
