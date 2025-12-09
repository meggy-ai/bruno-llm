"""
Middleware system for LLM providers.

Provides hooks for pre/post-processing of requests and responses.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from bruno_core.models import Message


class Middleware(ABC):
    """
    Base class for provider middleware.

    Middleware can intercept and modify:
    - Request messages before sending to provider
    - Response text after receiving from provider
    - Streaming chunks as they arrive
    - Request parameters (temperature, max_tokens, etc.)

    Example:
        >>> class LoggingMiddleware(Middleware):
        ...     async def before_request(self, messages, **kwargs):
        ...         print(f"Sending {len(messages)} messages")
        ...         return messages, kwargs
        ...
        ...     async def after_response(self, messages, response, **kwargs):
        ...         print(f"Received {len(response)} chars")
        ...         return response
    """

    @abstractmethod
    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """
        Process messages and parameters before request.

        Args:
            messages: Input messages
            **kwargs: Request parameters

        Returns:
            Tuple of (modified_messages, modified_kwargs)
        """
        pass

    @abstractmethod
    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """
        Process response after receiving.

        Args:
            messages: Original input messages
            response: Provider response
            **kwargs: Request parameters used

        Returns:
            Modified response
        """
        pass

    async def on_stream_chunk(self, chunk: str, **kwargs: Any) -> str:
        """
        Process individual stream chunks.

        Args:
            chunk: Stream chunk
            **kwargs: Request parameters

        Returns:
            Modified chunk
        """
        return chunk

    async def on_error(self, error: Exception, messages: list[Message], **kwargs: Any) -> None:  # noqa: B027
        """
        Handle errors during request.

        Args:
            error: The exception that occurred
            messages: Messages that were being processed
            **kwargs: Request parameters
        """
        pass


class LoggingMiddleware(Middleware):
    """
    Log all requests and responses.

    Args:
        logger: Logger instance (defaults to structlog)
        log_messages: Whether to log full message content

    Example:
        >>> middleware = LoggingMiddleware(log_messages=False)
        >>> provider = MiddlewareProvider(base_provider, [middleware])
    """

    def __init__(self, logger=None, log_messages: bool = False):
        """
        Initialize logging middleware.

        Args:
            logger: Logger instance
            log_messages: Whether to log message content
        """
        self.log_messages = log_messages

        if logger is None:
            import structlog

            self.logger = structlog.get_logger(__name__)
        else:
            self.logger = logger

    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """Log before request."""
        log_data = {
            "event": "llm_request",
            "message_count": len(messages),
            "params": {k: v for k, v in kwargs.items() if k not in ["api_key"]},
        }

        if self.log_messages:
            log_data["messages"] = [{"role": m.role.value, "content": m.content} for m in messages]

        self.logger.info("LLM request", **log_data)
        return messages, kwargs

    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """Log after response."""
        log_data = {
            "event": "llm_response",
            "response_length": len(response),
        }

        if self.log_messages:
            log_data["response"] = response

        self.logger.info("LLM response", **log_data)
        return response

    async def on_error(self, error: Exception, messages: list[Message], **kwargs: Any) -> None:
        """Log errors."""
        self.logger.error(
            "LLM error",
            event="llm_error",
            error=str(error),
            error_type=type(error).__name__,
            message_count=len(messages),
        )


class CachingMiddleware(Middleware):
    """
    Cache responses using ResponseCache.

    Args:
        cache: ResponseCache instance
        cache_streaming: Whether to cache streaming responses

    Example:
        >>> from bruno_llm.base.cache import ResponseCache
        >>> cache = ResponseCache(max_size=100, ttl=300)
        >>> middleware = CachingMiddleware(cache)
    """

    def __init__(self, cache, cache_streaming: bool = True):
        """
        Initialize caching middleware.

        Args:
            cache: ResponseCache instance
            cache_streaming: Whether to cache streaming responses
        """
        self.cache = cache
        self.cache_streaming = cache_streaming
        self._current_stream_chunks: Optional[list[str]] = None

    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """Check cache before request."""
        # Cache lookup is handled externally
        return messages, kwargs

    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """Cache response after receiving."""
        self.cache.set(messages, response, **kwargs)
        return response

    async def on_stream_chunk(self, chunk: str, **kwargs: Any) -> str:
        """Collect stream chunks for caching."""
        if self.cache_streaming:
            if self._current_stream_chunks is None:
                self._current_stream_chunks = []
            self._current_stream_chunks.append(chunk)

        return chunk


class ValidationMiddleware(Middleware):
    """
    Validate messages and parameters.

    Args:
        max_message_length: Max length for individual messages
        allowed_roles: Allowed message roles
        required_params: Required parameter names

    Example:
        >>> middleware = ValidationMiddleware(
        ...     max_message_length=10000,
        ...     allowed_roles=["user", "assistant", "system"]
        ... )
    """

    def __init__(
        self,
        max_message_length: Optional[int] = None,
        allowed_roles: Optional[list[str]] = None,
        required_params: Optional[list[str]] = None,
    ):
        """
        Initialize validation middleware.

        Args:
            max_message_length: Maximum message length
            allowed_roles: Allowed message roles
            required_params: Required parameters
        """
        self.max_message_length = max_message_length
        self.allowed_roles = allowed_roles
        self.required_params = required_params or []

    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """Validate before request."""
        # Validate message lengths
        if self.max_message_length:
            for msg in messages:
                if len(msg.content) > self.max_message_length:
                    raise ValueError(
                        f"Message content exceeds max length "
                        f"({len(msg.content)} > {self.max_message_length})"
                    )

        # Validate roles
        if self.allowed_roles:
            for msg in messages:
                if msg.role.value not in self.allowed_roles:
                    raise ValueError(
                        f"Invalid message role: {msg.role.value}. Allowed: {self.allowed_roles}"
                    )

        # Validate required parameters
        for param in self.required_params:
            if param not in kwargs:
                raise ValueError(f"Required parameter missing: {param}")

        return messages, kwargs

    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """No validation after response."""
        return response


class RetryMiddleware(Middleware):
    """
    Add retry logic with exponential backoff.

    Note: This is typically handled by BaseProvider's retry logic,
    but can be used for additional retry layers.

    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds

    Example:
        >>> middleware = RetryMiddleware(max_retries=3, base_delay=1.0)
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry middleware.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0

    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """Reset retry count before request."""
        self.retry_count = 0
        return messages, kwargs

    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """No processing after successful response."""
        return response

    async def on_error(self, error: Exception, messages: list[Message], **kwargs: Any) -> None:
        """Handle retry logic on error."""
        import asyncio

        self.retry_count += 1

        if self.retry_count <= self.max_retries:
            delay = self.base_delay * (2 ** (self.retry_count - 1))
            await asyncio.sleep(delay)


class MiddlewareChain:
    """
    Chain multiple middleware together.

    Executes middleware in order for before_request,
    and in reverse order for after_response.

    Args:
        middlewares: List of middleware instances

    Example:
        >>> chain = MiddlewareChain([
        ...     LoggingMiddleware(),
        ...     ValidationMiddleware(),
        ...     CachingMiddleware(cache),
        ... ])
        >>> messages, kwargs = await chain.before_request(messages, **kwargs)
    """

    def __init__(self, middlewares: list[Middleware]):
        """
        Initialize middleware chain.

        Args:
            middlewares: List of middleware to chain
        """
        self.middlewares = middlewares

    async def before_request(
        self, messages: list[Message], **kwargs: Any
    ) -> tuple[list[Message], dict[str, Any]]:
        """Execute all middleware before_request in order."""
        for middleware in self.middlewares:
            messages, kwargs = await middleware.before_request(messages, **kwargs)
        return messages, kwargs

    async def after_response(self, messages: list[Message], response: str, **kwargs: Any) -> str:
        """Execute all middleware after_response in reverse order."""
        for middleware in reversed(self.middlewares):
            response = await middleware.after_response(messages, response, **kwargs)
        return response

    async def on_stream_chunk(self, chunk: str, **kwargs: Any) -> str:
        """Execute all middleware on_stream_chunk in order."""
        for middleware in self.middlewares:
            chunk = await middleware.on_stream_chunk(chunk, **kwargs)
        return chunk

    async def on_error(self, error: Exception, messages: list[Message], **kwargs: Any) -> None:
        """Execute all middleware on_error."""
        for middleware in self.middlewares:
            await middleware.on_error(error, messages, **kwargs)
