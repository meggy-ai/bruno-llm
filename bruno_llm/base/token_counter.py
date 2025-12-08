"""
Token counting utilities for LLM providers.

Provides token counting implementations for different providers.
Default implementation uses simple word counting, but can be extended
with provider-specific tokenizers (e.g., tiktoken for OpenAI).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from bruno_core.models import Message


class TokenCounter(ABC):
    """
    Abstract base class for token counting.
    
    Different providers may have different tokenization methods.
    Subclasses should implement provider-specific counting logic.
    """
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens
        """
        pass
    
    def count_message_tokens(self, message: Message) -> int:
        """
        Count tokens in a message.
        
        Args:
            message: Message to count tokens for
        
        Returns:
            Number of tokens
        """
        return self.count_tokens(message.content)
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """
        Count tokens in multiple messages.
        
        Args:
            messages: List of messages
        
        Returns:
            Total number of tokens
        """
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
            # Add overhead for message formatting (role, etc.)
            total += 4  # Approximate overhead per message
        return total


class SimpleTokenCounter(TokenCounter):
    """
    Simple token counter using word splitting.
    
    This is a fallback implementation that approximates token count
    by counting words. Not as accurate as provider-specific tokenizers
    but works universally.
    
    Example:
        >>> counter = SimpleTokenCounter()
        >>> tokens = counter.count_tokens("Hello world!")
        >>> print(tokens)  # Approximately 2-3
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        """
        Initialize simple token counter.
        
        Args:
            chars_per_token: Average characters per token (default: 4)
        """
        self.chars_per_token = chars_per_token
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using character-based estimation.
        
        Uses the common approximation that 1 token ≈ 4 characters
        in English text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))


class TikTokenCounter(TokenCounter):
    """
    Token counter using OpenAI's tiktoken library.
    
    Provides accurate token counting for OpenAI models.
    Falls back to SimpleTokenCounter if tiktoken is not available.
    
    Example:
        >>> counter = TikTokenCounter(model="gpt-4")
        >>> tokens = counter.count_tokens("Hello world!")
        >>> print(tokens)
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize tiktoken-based counter.
        
        Args:
            model: Model name for tiktoken encoding
        """
        self.model = model
        self._encoding = None
        self._fallback = SimpleTokenCounter()
        
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
        except ImportError:
            # tiktoken not available, will use fallback
            pass
        except Exception:
            # Model not found or other error, use fallback
            pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken or fallback.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Accurate token count (if tiktoken available) or estimate
        """
        if not text:
            return 0
        
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        
        # Fallback to simple counting
        return self._fallback.count_tokens(text)


def create_token_counter(
    provider: str = "simple",
    model: Optional[str] = None,
) -> TokenCounter:
    """
    Factory function to create appropriate token counter.
    
    Args:
        provider: Provider name ("simple", "openai", "tiktoken")
        model: Optional model name for provider-specific counting
    
    Returns:
        TokenCounter instance
    
    Example:
        >>> counter = create_token_counter("openai", model="gpt-4")
        >>> tokens = counter.count_tokens("Hello!")
    """
    if provider in ("openai", "tiktoken"):
        if model:
            return TikTokenCounter(model=model)
        return TikTokenCounter()
    
    return SimpleTokenCounter()
