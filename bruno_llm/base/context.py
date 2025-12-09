"""
Context window management for LLM providers.

Manages token limits, message truncation, and context overflow handling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from bruno_core.models import Message, MessageRole
from bruno_llm.base.token_counter import TokenCounter, create_token_counter
from bruno_llm.exceptions import ContextLengthExceededError


class TruncationStrategy(Enum):
    """Strategy for truncating messages when context limit is exceeded."""

    OLDEST_FIRST = "oldest_first"  # Remove oldest messages first
    MIDDLE_OUT = "middle_out"  # Keep first and last, remove middle
    SLIDING_WINDOW = "sliding_window"  # Keep most recent N messages
    SMART = "smart"  # Keep system + important messages + recent


@dataclass
class ContextLimits:
    """
    Context window limits for a model.

    Attributes:
        max_tokens: Maximum total tokens (input + output)
        max_input_tokens: Maximum input tokens
        max_output_tokens: Maximum output tokens
        warning_threshold: Warn when this % of limit is reached (0.0-1.0)
    """

    max_tokens: int
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    warning_threshold: float = 0.9

    def __post_init__(self):
        """Validate limits after initialization."""
        if self.max_input_tokens is None:
            self.max_input_tokens = self.max_tokens
        if self.max_output_tokens is None:
            self.max_output_tokens = self.max_tokens // 4  # Default 25% for output


# Common model limits
MODEL_LIMITS = {
    # OpenAI models
    "gpt-4": ContextLimits(max_tokens=8192),
    "gpt-4-32k": ContextLimits(max_tokens=32768),
    "gpt-4-turbo": ContextLimits(max_tokens=128000),
    "gpt-3.5-turbo": ContextLimits(max_tokens=4096),
    "gpt-3.5-turbo-16k": ContextLimits(max_tokens=16384),
    # Ollama models (approximate)
    "llama2": ContextLimits(max_tokens=4096),
    "llama2:13b": ContextLimits(max_tokens=4096),
    "llama2:70b": ContextLimits(max_tokens=4096),
    "mistral": ContextLimits(max_tokens=8192),
    "mixtral": ContextLimits(max_tokens=32768),
    # Claude models
    "claude-2": ContextLimits(max_tokens=100000),
    "claude-3-opus": ContextLimits(max_tokens=200000),
    "claude-3-sonnet": ContextLimits(max_tokens=200000),
    "claude-3-haiku": ContextLimits(max_tokens=200000),
}


class ContextWindowManager:
    """
    Manage context windows and message truncation.

    Handles:
    - Token counting for messages
    - Context limit checking
    - Automatic message truncation
    - Warning when approaching limits

    Args:
        model: Model name for context limits
        token_counter: Token counter instance
        limits: Custom context limits (overrides model defaults)
        strategy: Truncation strategy to use

    Example:
        >>> manager = ContextWindowManager(model="gpt-4")
        >>>
        >>> # Check if messages fit
        >>> if manager.check_limit(messages):
        ...     response = await provider.generate(messages)
        ... else:
        ...     # Truncate messages
        ...     truncated = manager.truncate(messages)
        ...     response = await provider.generate(truncated)
    """

    def __init__(
        self,
        model: str,
        token_counter: Optional[TokenCounter] = None,
        limits: Optional[ContextLimits] = None,
        strategy: TruncationStrategy = TruncationStrategy.SLIDING_WINDOW,
    ):
        """
        Initialize context window manager.

        Args:
            model: Model name
            token_counter: Token counter instance
            limits: Custom context limits
            strategy: Truncation strategy
        """
        self.model = model
        self.token_counter = token_counter or create_token_counter(model)
        self.limits = limits or self._get_model_limits(model)
        self.strategy = strategy
        self._warning_callback: Optional[Callable[[int, int], None]] = None

    def _get_model_limits(self, model: str) -> ContextLimits:
        """
        Get context limits for a model.

        Args:
            model: Model name

        Returns:
            Context limits for the model
        """
        # Try exact match first
        if model in MODEL_LIMITS:
            return MODEL_LIMITS[model]

        # Try partial match
        for model_name, limits in MODEL_LIMITS.items():
            if model.startswith(model_name):
                return limits

        # Default conservative limit
        return ContextLimits(max_tokens=4096)

    def count_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in messages.

        Args:
            messages: List of messages

        Returns:
            Total token count
        """
        return self.token_counter.count_messages_tokens(messages)

    def check_limit(
        self,
        messages: list[Message],
        max_output_tokens: Optional[int] = None,
    ) -> bool:
        """
        Check if messages fit within context limit.

        Args:
            messages: List of messages
            max_output_tokens: Expected output tokens

        Returns:
            True if messages fit, False otherwise
        """
        input_tokens = self.count_tokens(messages)
        output_tokens = max_output_tokens or self.limits.max_output_tokens
        total_tokens = input_tokens + output_tokens

        # Check warning threshold
        if input_tokens / self.limits.max_input_tokens >= self.limits.warning_threshold:
            if self._warning_callback:
                self._warning_callback(input_tokens, self.limits.max_input_tokens)

        return total_tokens <= self.limits.max_tokens

    def get_available_tokens(self, messages: list[Message]) -> int:
        """
        Get number of tokens available for output.

        Args:
            messages: List of messages

        Returns:
            Available tokens for output
        """
        input_tokens = self.count_tokens(messages)
        return max(0, self.limits.max_tokens - input_tokens)

    def truncate(
        self,
        messages: list[Message],
        max_output_tokens: Optional[int] = None,
    ) -> list[Message]:
        """
        Truncate messages to fit within context limit.

        Args:
            messages: List of messages
            max_output_tokens: Expected output tokens

        Returns:
            Truncated message list

        Raises:
            ContextLengthExceededError: If messages can't be truncated enough
        """
        output_tokens = max_output_tokens or self.limits.max_output_tokens
        target_input_tokens = self.limits.max_tokens - output_tokens

        if target_input_tokens <= 0:
            raise ContextLengthExceededError(
                f"Output tokens ({output_tokens}) exceed total limit ({self.limits.max_tokens})"
            )

        if self.strategy == TruncationStrategy.OLDEST_FIRST:
            return self._truncate_oldest_first(messages, target_input_tokens)
        elif self.strategy == TruncationStrategy.MIDDLE_OUT:
            return self._truncate_middle_out(messages, target_input_tokens)
        elif self.strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._truncate_sliding_window(messages, target_input_tokens)
        elif self.strategy == TruncationStrategy.SMART:
            return self._truncate_smart(messages, target_input_tokens)
        else:
            return self._truncate_oldest_first(messages, target_input_tokens)

    def _truncate_oldest_first(
        self,
        messages: list[Message],
        target_tokens: int,
    ) -> list[Message]:
        """Remove oldest messages first (keep system message)."""
        # Always keep system messages
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        # Start with system messages
        result = system_messages[:]
        current_tokens = self.count_tokens(result)

        # Add messages from newest to oldest
        for message in reversed(other_messages):
            message_tokens = self.token_counter.count_message_tokens(message)
            if current_tokens + message_tokens <= target_tokens:
                result.insert(len(system_messages), message)
                current_tokens += message_tokens
            else:
                break

        # Re-order to maintain chronological order (except system at start)
        return system_messages + list(reversed(result[len(system_messages) :]))

    def _truncate_middle_out(
        self,
        messages: list[Message],
        target_tokens: int,
    ) -> list[Message]:
        """Keep first and last messages, remove middle."""
        if len(messages) <= 2:
            return messages

        # Keep system messages and last message
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not other_messages:
            return messages

        result = system_messages + [other_messages[-1]]
        current_tokens = self.count_tokens(result)

        # Add messages from the start
        for message in other_messages[:-1]:
            message_tokens = self.token_counter.count_message_tokens(message)
            if current_tokens + message_tokens <= target_tokens:
                result.insert(len(system_messages), message)
                current_tokens += message_tokens
            else:
                break

        return result

    def _truncate_sliding_window(
        self,
        messages: list[Message],
        target_tokens: int,
    ) -> list[Message]:
        """Keep most recent N messages."""
        # Always keep system messages
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        result = system_messages[:]
        current_tokens = self.count_tokens(result)

        # Add messages from newest to oldest
        for message in reversed(other_messages):
            message_tokens = self.token_counter.count_message_tokens(message)
            if current_tokens + message_tokens <= target_tokens:
                result.append(message)
                current_tokens += message_tokens
            else:
                break

        # Keep system messages at start, reverse others
        return system_messages + list(reversed(result[len(system_messages) :]))

    def _truncate_smart(
        self,
        messages: list[Message],
        target_tokens: int,
    ) -> list[Message]:
        """
        Smart truncation: keep system + important messages + recent.

        Priority:
        1. System messages (always keep)
        2. Last 2 messages (recent context)
        3. Messages with high token count (likely important)
        4. Fill remaining space with recent messages
        """
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not other_messages:
            return messages

        # Start with system messages
        result = system_messages[:]
        current_tokens = self.count_tokens(result)

        # Always include last 2 messages (most recent context)
        priority_messages = other_messages[-2:]
        for message in priority_messages:
            message_tokens = self.token_counter.count_message_tokens(message)
            if current_tokens + message_tokens <= target_tokens:
                result.append(message)
                current_tokens += message_tokens

        # Fill remaining space with other messages (newest first)
        remaining = list(other_messages[:-2])
        for message in reversed(remaining):
            message_tokens = self.token_counter.count_message_tokens(message)
            if current_tokens + message_tokens <= target_tokens:
                result.insert(len(system_messages), message)
                current_tokens += message_tokens
            else:
                break

        return result

    def set_warning_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Set callback for context limit warnings.

        Args:
            callback: Function (current_tokens, max_tokens) -> None
        """
        self._warning_callback = callback

    def get_stats(self, messages: list[Message]) -> dict:
        """
        Get statistics about context usage.

        Args:
            messages: List of messages

        Returns:
            Dictionary with context statistics
        """
        input_tokens = self.count_tokens(messages)
        available_tokens = self.get_available_tokens(messages)
        usage_percent = (input_tokens / self.limits.max_input_tokens) * 100

        return {
            "model": self.model,
            "input_tokens": input_tokens,
            "max_input_tokens": self.limits.max_input_tokens,
            "available_output_tokens": available_tokens,
            "max_output_tokens": self.limits.max_output_tokens,
            "total_limit": self.limits.max_tokens,
            "usage_percent": usage_percent,
            "within_limit": self.check_limit(messages),
            "message_count": len(messages),
        }
