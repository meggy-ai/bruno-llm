"""
Exception hierarchy for bruno-llm.

Defines custom exceptions for LLM provider errors, enabling
consistent error handling across all providers.
"""

from typing import Optional


class LLMError(Exception):
    """
    Base exception for all LLM-related errors.

    All custom exceptions in bruno-llm inherit from this class.

    Args:
        message: Error message
        provider: Name of the provider that raised the error
        original_error: Original exception if available
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with provider context."""
        parts = []
        if self.provider:
            parts.append(f"[{self.provider}]")
        parts.append(self.message)
        if self.original_error:
            parts.append(
                f"(caused by: {type(self.original_error).__name__}: {self.original_error})"
            )
        return " ".join(parts)


class AuthenticationError(LLMError):
    """
    Raised when authentication with LLM provider fails.

    Common causes:
    - Invalid API key
    - Expired API key
    - Missing API key
    - Invalid organization ID

    Example:
        >>> raise AuthenticationError("Invalid API key", provider="openai")
    """

    pass


class RateLimitError(LLMError):
    """
    Raised when API rate limits are exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided)

    Example:
        >>> raise RateLimitError("Rate limit exceeded", provider="openai", retry_after=60)
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, provider, original_error)
        self.retry_after = retry_after


class ModelNotFoundError(LLMError):
    """
    Raised when requested model is not found or not available.

    Common causes:
    - Model name misspelled
    - Model not available in region
    - Model access not granted
    - Model has been deprecated

    Example:
        >>> raise ModelNotFoundError("Model 'gpt-5' not found", provider="openai")
    """

    pass


class ContextLengthExceededError(LLMError):
    """
    Raised when input exceeds model's context length.

    Attributes:
        max_tokens: Maximum tokens allowed
        actual_tokens: Actual token count in request

    Example:
        >>> raise ContextLengthExceededError(
        ...     "Context length exceeded",
        ...     provider="openai",
        ...     max_tokens=8192,
        ...     actual_tokens=10000
        ... )
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None,
    ):
        super().__init__(message, provider, original_error)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class StreamError(LLMError):
    """
    Raised when streaming response encounters an error.

    Common causes:
    - Network connection lost
    - Server-side error during streaming
    - Invalid chunk format
    - Stream interrupted

    Example:
        >>> raise StreamError("Stream connection lost", provider="openai")
    """

    pass


class ConfigurationError(LLMError):
    """
    Raised when provider configuration is invalid.

    Common causes:
    - Missing required configuration
    - Invalid configuration values
    - Conflicting configuration options

    Example:
        >>> raise ConfigurationError("Missing base_url", provider="ollama")
    """

    pass


class TimeoutError(LLMError):
    """
    Raised when request times out.

    Attributes:
        timeout: Timeout value in seconds

    Example:
        >>> raise TimeoutError("Request timed out after 30s", provider="openai", timeout=30)
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(message, provider, original_error)
        self.timeout = timeout


class InvalidResponseError(LLMError):
    """
    Raised when provider returns invalid or unexpected response.

    Common causes:
    - Malformed JSON response
    - Missing required fields
    - Unexpected response structure

    Example:
        >>> raise InvalidResponseError("Missing 'content' field", provider="openai")
    """

    pass


class ProviderNotFoundError(LLMError):
    """
    Raised when requested provider is not registered.

    Example:
        >>> raise ProviderNotFoundError("Provider 'unknown' not found")
    """

    pass
