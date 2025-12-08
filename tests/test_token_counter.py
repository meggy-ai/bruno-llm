"""Tests for token counting utilities."""

import pytest
from bruno_core.models import Message, MessageRole

from bruno_llm.base.token_counter import (
    SimpleTokenCounter,
    TikTokenCounter,
    create_token_counter,
)


def test_simple_token_counter():
    """Test simple token counter."""
    counter = SimpleTokenCounter()
    
    # Empty string
    assert counter.count_tokens("") == 0
    
    # Short text
    tokens = counter.count_tokens("Hello world")
    assert tokens >= 2  # At least 2 tokens for 11 characters
    
    # Longer text
    text = "The quick brown fox jumps over the lazy dog"
    tokens = counter.count_tokens(text)
    assert tokens > 5  # Should have multiple tokens


def test_simple_token_counter_with_custom_ratio():
    """Test simple token counter with custom chars_per_token."""
    counter = SimpleTokenCounter(chars_per_token=2.0)
    
    # Should count more tokens with smaller ratio
    tokens = counter.count_tokens("Hello world")
    assert tokens >= 5  # 11 chars / 2 = 5.5 tokens


def test_simple_token_counter_messages():
    """Test counting tokens in messages."""
    counter = SimpleTokenCounter()
    
    message = Message(role=MessageRole.USER, content="Hello world")
    tokens = counter.count_message_tokens(message)
    assert tokens >= 2


def test_simple_token_counter_multiple_messages():
    """Test counting tokens in multiple messages."""
    counter = SimpleTokenCounter()
    
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
        Message(role=MessageRole.USER, content="Hello"),
    ]
    
    tokens = counter.count_messages_tokens(messages)
    # Should include message content + overhead (4 per message)
    assert tokens >= 10


def test_tiktoken_counter_fallback():
    """Test TikTokenCounter falls back to simple counting when tiktoken unavailable."""
    # Create counter (may or may not have tiktoken)
    counter = TikTokenCounter(model="gpt-4")
    
    # Should work regardless of whether tiktoken is installed
    tokens = counter.count_tokens("Hello world")
    assert tokens >= 1
    
    # Empty string
    assert counter.count_tokens("") == 0


def test_create_token_counter_simple():
    """Test factory function creates simple counter."""
    counter = create_token_counter("simple")
    
    assert isinstance(counter, SimpleTokenCounter)
    tokens = counter.count_tokens("Hello")
    assert tokens >= 1


def test_create_token_counter_tiktoken():
    """Test factory function creates tiktoken counter."""
    counter = create_token_counter("openai", model="gpt-4")
    
    assert isinstance(counter, TikTokenCounter)
    tokens = counter.count_tokens("Hello")
    assert tokens >= 1


def test_create_token_counter_default():
    """Test factory function defaults to simple counter."""
    counter = create_token_counter("unknown")
    
    assert isinstance(counter, SimpleTokenCounter)
