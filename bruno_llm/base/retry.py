"""
Retry logic with exponential backoff.

Provides configurable retry mechanisms for handling transient failures
in API calls.
"""

import asyncio
import random
from typing import Any, Callable, Optional, Type, TypeVar

from bruno_llm.exceptions import LLMError, RateLimitError

T = TypeVar("T")


class RetryConfig:
    """
    Configuration for retry behavior.

    Example:
        >>> config = RetryConfig(
        ...     max_retries=5,
        ...     initial_delay=1.0,
        ...     max_delay=60.0,
        ...     exponential_base=2.0
        ... )
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[tuple[Type[Exception], ...]] = None,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retry_on: Tuple of exception types to retry on (None = all)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or (Exception,)

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given retry attempt.

        Uses exponential backoff with optional jitter.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            exception: Exception that was raised
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exhausted retries
        if attempt >= self.max_retries:
            return False

        # Check if exception type is retryable
        if not isinstance(exception, self.retry_on):
            return False

        # Special handling for rate limit errors
        if isinstance(exception, RateLimitError):
            return True

        return True


async def retry_async(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception: Last exception if all retries fail

    Example:
        >>> async def api_call():
        ...     # May fail transiently
        ...     return await external_api()
        >>> result = await retry_async(api_call, config=RetryConfig(max_retries=5))
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if we should retry
            if not config.should_retry(e, attempt):
                raise

            # Calculate and wait
            if attempt < config.max_retries:
                delay = config.calculate_delay(attempt)

                # Special handling for rate limit with retry_after
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)

                await asyncio.sleep(delay)

    # Should not reach here, but for safety
    if last_exception:
        raise last_exception
    raise LLMError("Retry loop exited unexpectedly")


class RetryDecorator:
    """
    Decorator for adding retry logic to async functions.

    Example:
        >>> @RetryDecorator(max_retries=5)
        ... async def api_call():
        ...     return await external_api()
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry decorator.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter
        """
        self.config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Wrap function with retry logic.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(func, *args, config=self.config, **kwargs)

        return wrapper
