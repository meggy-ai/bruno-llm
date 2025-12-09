"""Tests for OpenAI provider."""

from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from bruno_core.models import Message, MessageRole

from bruno_llm.exceptions import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    StreamError,
)
from bruno_llm.providers.openai import OpenAIConfig, OpenAIProvider


@pytest.fixture
def openai_config():
    """Create test OpenAI configuration."""
    return OpenAIConfig(
        api_key="sk-test-key-123",
        model="gpt-4",
        timeout=30.0,
    )


@pytest.fixture
def openai_provider():
    """Create test OpenAI provider."""
    return OpenAIProvider(api_key="sk-test-key-123", model="gpt-4", track_cost=False)


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
    ]


def test_openai_config_default():
    """Test default OpenAI configuration."""
    config = OpenAIConfig(api_key="sk-test-key")
    
    assert config.model == "gpt-4"
    assert config.timeout == 30.0
    assert config.temperature == 0.7
    assert config.base_url == "https://api.openai.com/v1"


def test_openai_config_custom():
    """Test custom OpenAI configuration."""
    config = OpenAIConfig(
        api_key="sk-test-key",
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1000,
        organization="org-123",
    )
    
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
    assert config.organization == "org-123"


def test_openai_provider_init():
    """Test OpenAI provider initialization."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4")
    
    assert provider.model == "gpt-4"
    assert provider.config.base_url == "https://api.openai.com/v1"


def test_openai_provider_format_messages(openai_provider, sample_messages):
    """Test message formatting."""
    formatted = openai_provider._format_messages(sample_messages)
    
    assert len(formatted) == 2
    assert formatted[0] == {"role": "system", "content": "You are helpful"}
    assert formatted[1] == {"role": "user", "content": "Hello"}


def test_openai_provider_build_request_params(openai_provider):
    """Test request parameter building."""
    params = openai_provider._build_request_params()
    
    assert params["model"] == "gpt-4"
    assert params["temperature"] == 0.7
    assert params["top_p"] == 1.0


@pytest.mark.asyncio
async def test_generate_success(openai_provider, sample_messages):
    """Test successful generation."""
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Hello! How can I help you?"
    mock_completion.usage = None
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_completion
        
        response = await openai_provider.generate(sample_messages)
        
        assert response == "Hello! How can I help you?"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_with_usage_tracking():
    """Test generation with cost tracking."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4", track_cost=True)
    
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Test response"
    mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=5)
    
    with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_completion
        
        messages = [Message(role=MessageRole.USER, content="Hello")]
        await provider.generate(messages)
        
        # Check cost tracking
        assert provider.cost_tracker.get_request_count() == 1
        assert provider.cost_tracker.get_total_cost() > 0


@pytest.mark.asyncio
async def test_generate_authentication_error(openai_provider, sample_messages):
    """Test generation with authentication error."""
    from openai import AuthenticationError as OpenAIAuthError
    from httpx import Response, Request
    
    mock_response = Response(status_code=401, request=Request("POST", "https://api.openai.com/v1"))
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = OpenAIAuthError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        
        with pytest.raises(AuthenticationError):
            await openai_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_rate_limit_error(openai_provider, sample_messages):
    """Test generation with rate limit error."""
    from openai import RateLimitError as OpenAIRateLimitError
    from httpx import Response, Request
    
    mock_response = Response(status_code=429, request=Request("POST", "https://api.openai.com/v1"))
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        with pytest.raises(RateLimitError):
            await openai_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_model_not_found(openai_provider, sample_messages):
    """Test generation with model not found."""
    from openai import NotFoundError
    from httpx import Response, Request
    
    mock_response = Response(status_code=404, request=Request("POST", "https://api.openai.com/v1"))
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = NotFoundError(
            "Model not found",
            response=mock_response,
            body={"error": {"message": "Model not found"}}
        )
        
        with pytest.raises(ModelNotFoundError):
            await openai_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_timeout(openai_provider, sample_messages):
    """Test generation timeout."""
    from openai import APITimeoutError
    from bruno_llm.exceptions import TimeoutError as LLMTimeoutError
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = APITimeoutError("Timeout")
        
        with pytest.raises(LLMTimeoutError):
            await openai_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_generate_empty_response(openai_provider, sample_messages):
    """Test generation with empty response."""
    from bruno_llm.exceptions import InvalidResponseError
    
    mock_completion = Mock()
    mock_completion.choices = []
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_completion
        
        with pytest.raises(InvalidResponseError):
            await openai_provider.generate(sample_messages)


@pytest.mark.asyncio
async def test_stream_success(openai_provider, sample_messages):
    """Test successful streaming."""
    # Create mock chunks
    chunks = []
    for content in ["Hello", " there", "!"]:
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta = Mock(content=content)
        chunks.append(chunk)
    
    # Create async iterator
    async def mock_stream():
        for chunk in chunks:
            yield chunk
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_stream()
        
        result = []
        async for chunk in openai_provider.stream(sample_messages):
            result.append(chunk)
        
        assert result == ["Hello", " there", "!"]


@pytest.mark.asyncio
async def test_stream_authentication_error(openai_provider, sample_messages):
    """Test streaming with authentication error."""
    from openai import AuthenticationError as OpenAIAuthError
    from httpx import Response, Request
    
    mock_response = Response(status_code=401, request=Request("POST", "https://api.openai.com/v1"))
    
    with patch.object(openai_provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = OpenAIAuthError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        
        with pytest.raises(AuthenticationError):
            async for _ in openai_provider.stream(sample_messages):
                pass


@pytest.mark.asyncio
async def test_list_models_success(openai_provider):
    """Test listing models."""
    mock_response = Mock()
    mock_response.data = [
        Mock(id="gpt-4"),
        Mock(id="gpt-3.5-turbo"),
        Mock(id="gpt-4-turbo"),
    ]
    
    with patch.object(openai_provider._client.models, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = mock_response
        
        models = await openai_provider.list_models()
        
        assert models == ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_list_models_error(openai_provider):
    """Test listing models with error."""
    from openai import OpenAIError
    
    with patch.object(openai_provider._client.models, "list", new_callable=AsyncMock) as mock_list:
        mock_list.side_effect = OpenAIError("API error")
        
        with pytest.raises(LLMError):
            await openai_provider.list_models()


@pytest.mark.asyncio
async def test_check_connection_success(openai_provider):
    """Test successful connection check."""
    mock_response = Mock()
    mock_response.data = []
    
    with patch.object(openai_provider._client.models, "list", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = mock_response
        
        result = await openai_provider.check_connection()
        
        assert result is True


@pytest.mark.asyncio
async def test_check_connection_failure(openai_provider):
    """Test failed connection check."""
    from openai import OpenAIError
    
    with patch.object(openai_provider._client.models, "list", new_callable=AsyncMock) as mock_list:
        mock_list.side_effect = OpenAIError("Connection failed")
        
        result = await openai_provider.check_connection()
        
        assert result is False


def test_get_token_count(openai_provider):
    """Test token counting."""
    text = "Hello world"
    tokens = openai_provider.get_token_count(text)
    
    assert tokens >= 2  # Should have at least 2 tokens


def test_get_model_info(openai_provider):
    """Test getting model info."""
    info = openai_provider.get_model_info()
    
    assert info["provider"] == "openai"
    assert info["model"] == "gpt-4"
    assert "base_url" in info
    assert "temperature" in info


def test_get_model_info_with_cost_tracking():
    """Test getting model info with cost tracking enabled."""
    provider = OpenAIProvider(api_key="sk-test", track_cost=True)
    
    info = provider.get_model_info()
    
    assert "cost_tracking" in info
    assert info["cost_tracking"]["enabled"] is True
    assert "total_cost" in info["cost_tracking"]


@pytest.mark.asyncio
async def test_context_manager():
    """Test provider as context manager."""
    async with OpenAIProvider(api_key="sk-test", model="gpt-4") as provider:
        assert provider.model == "gpt-4"
    
    # Client should be closed after context exit
