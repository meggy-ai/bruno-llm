"""Tests for bruno-llm exception hierarchy."""

import pytest
from bruno_llm.exceptions import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthExceededError,
    StreamError,
    ConfigurationError,
    TimeoutError,
    InvalidResponseError,
)


def test_llm_error_base():
    """Test base LLMError exception."""
    error = LLMError("Test error", provider="test-provider")
    assert "Test error" in str(error)
    assert "[test-provider]" in str(error)
    assert error.provider == "test-provider"


def test_llm_error_with_original():
    """Test LLMError with original exception."""
    original = ValueError("Original error")
    error = LLMError("Wrapper error", provider="test", original_error=original)
    assert "Wrapper error" in str(error)
    assert "ValueError" in str(error)
    assert error.original_error == original


def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError("Invalid API key", provider="openai")
    assert isinstance(error, LLMError)
    assert "Invalid API key" in str(error)
    assert "[openai]" in str(error)


def test_rate_limit_error():
    """Test RateLimitError with retry_after."""
    error = RateLimitError("Rate limit exceeded", provider="openai", retry_after=60)
    assert isinstance(error, LLMError)
    assert error.retry_after == 60
    assert "Rate limit exceeded" in str(error)


def test_model_not_found_error():
    """Test ModelNotFoundError."""
    error = ModelNotFoundError("Model 'gpt-5' not found", provider="openai")
    assert isinstance(error, LLMError)
    assert "gpt-5" in str(error)


def test_context_length_exceeded_error():
    """Test ContextLengthExceededError with token counts."""
    error = ContextLengthExceededError(
        "Context too long",
        provider="openai",
        max_tokens=4096,
        actual_tokens=5000,
    )
    assert isinstance(error, LLMError)
    assert error.max_tokens == 4096
    assert error.actual_tokens == 5000


def test_stream_error():
    """Test StreamError."""
    error = StreamError("Stream interrupted", provider="openai")
    assert isinstance(error, LLMError)
    assert "Stream interrupted" in str(error)


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError("Missing base_url", provider="ollama")
    assert isinstance(error, LLMError)
    assert "Missing base_url" in str(error)


def test_timeout_error():
    """Test TimeoutError with timeout value."""
    error = TimeoutError("Request timeout", provider="openai", timeout=30.0)
    assert isinstance(error, LLMError)
    assert error.timeout == 30.0


def test_invalid_response_error():
    """Test InvalidResponseError."""
    error = InvalidResponseError("Missing content field", provider="openai")
    assert isinstance(error, LLMError)
    assert "Missing content field" in str(error)


def test_exception_inheritance():
    """Test that all exceptions inherit from LLMError."""
    exceptions = [
        AuthenticationError("test"),
        RateLimitError("test"),
        ModelNotFoundError("test"),
        ContextLengthExceededError("test"),
        StreamError("test"),
        ConfigurationError("test"),
        TimeoutError("test"),
        InvalidResponseError("test"),
    ]
    
    for exc in exceptions:
        assert isinstance(exc, LLMError)
        assert isinstance(exc, Exception)
