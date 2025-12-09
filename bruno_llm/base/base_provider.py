"""
Base provider implementation with common functionality.

This module provides a base class that all LLM provider implementations
can extend to get common functionality like retry logic, rate limiting,
and cost tracking.
"""

import asyncio
from abc import ABC
from typing import Any, Callable, Optional, TypeVar

from bruno_core.interfaces import LLMInterface
from bruno_core.models import Message
from bruno_llm.exceptions import LLMError

T = TypeVar("T")


class BaseProvider(LLMInterface, ABC):
    """
    Base class for LLM provider implementations.

    Provides common functionality that all providers can use:
    - Retry logic with exponential backoff
    - Rate limiting
    - Cost tracking
    - Error handling patterns

    Subclasses must implement the LLMInterface methods:
    - generate()
    - stream()
    - get_token_count()
    - check_connection()
    - list_models()
    - get_model_info()
    - set_system_prompt()
    - get_system_prompt()

    Example:
        >>> class MyProvider(BaseProvider):
        ...     async def generate(self, messages, **kwargs):
        ...         return await self._with_retry(
        ...             self._generate_impl(messages, **kwargs)
        ...         )
    """

    def __init__(
        self,
        provider_name: str,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs: Any,
    ):
        """
        Initialize base provider.

        Args:
            provider_name: Name of the provider (e.g., "ollama", "openai")
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific configuration
        """
        self.provider_name = provider_name
        self.max_retries = max_retries
        self.timeout = timeout
        self._system_prompt: Optional[str] = None
        self._config = kwargs

    async def _with_retry(
        self,
        coro: Callable[[], T],
        max_retries: Optional[int] = None,
    ) -> T:
        """
        Execute coroutine with retry logic.

        Implements exponential backoff with jitter for retries.

        Args:
            coro: Async function to execute
            max_retries: Override default max_retries

        Returns:
            Result from coroutine

        Raises:
            LLMError: If all retries are exhausted
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error = None

        for attempt in range(retries + 1):
            try:
                return await coro()
            except Exception as e:
                last_error = e

                if attempt < retries:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + (asyncio.get_event_loop().time() % 1)
                    await asyncio.sleep(delay)
                    continue

                # All retries exhausted
                if isinstance(e, LLMError):
                    raise
                raise LLMError(
                    f"Request failed after {retries + 1} attempts",
                    provider=self.provider_name,
                    original_error=e,
                ) from e

        # Should never reach here, but for type safety
        raise LLMError(
            "Unexpected retry loop exit",
            provider=self.provider_name,
            original_error=last_error,
        )

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set system prompt for the provider.

        Args:
            prompt: System prompt text
        """
        self._system_prompt = prompt

    def get_system_prompt(self) -> Optional[str]:
        """
        Get current system prompt.

        Returns:
            Current system prompt or None
        """
        return self._system_prompt

    def _add_system_prompt(self, messages: list[Message]) -> list[Message]:
        """
        Add system prompt to messages if set.

        Args:
            messages: Original messages

        Returns:
            Messages with system prompt prepended if set
        """
        if not self._system_prompt:
            return messages

        from bruno_core.models import MessageRole

        # Check if first message is already a system message
        if messages and messages[0].role == MessageRole.SYSTEM:
            return messages

        # Prepend system prompt
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=self._system_prompt,
        )
        return [system_message] + messages

    def get_model_info(self) -> dict[str, Any]:
        """
        Get provider and configuration information.

        Returns:
            Dict with provider information
        """
        return {
            "provider": self.provider_name,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "system_prompt": self._system_prompt,
            **self._config,
        }
