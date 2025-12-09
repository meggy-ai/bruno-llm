"""
Tests for context window management.
"""

import pytest

from bruno_core.models import Message, MessageRole
from bruno_llm.base.context import (
    MODEL_LIMITS,
    ContextLimits,
    ContextWindowManager,
    TruncationStrategy,
)
from bruno_llm.base.token_counter import SimpleTokenCounter
from bruno_llm.exceptions import ContextLengthExceededError


@pytest.fixture
def simple_counter():
    """Create a simple token counter."""
    return SimpleTokenCounter(chars_per_token=4)


@pytest.fixture
def context_manager(simple_counter):
    """Create a context window manager."""
    limits = ContextLimits(max_tokens=100, max_output_tokens=20)
    return ContextWindowManager(
        model="test-model",
        token_counter=simple_counter,
        limits=limits,
    )


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),  # ~4 tokens
        Message(role=MessageRole.USER, content="Hello, how are you?"),  # ~5 tokens
        Message(role=MessageRole.ASSISTANT, content="I'm great!"),  # ~3 tokens
        Message(role=MessageRole.USER, content="Tell me more."),  # ~3 tokens
    ]


def test_context_limits():
    """Test ContextLimits dataclass."""
    limits = ContextLimits(max_tokens=1000, warning_threshold=0.8)

    assert limits.max_tokens == 1000
    assert limits.max_input_tokens == 1000
    assert limits.max_output_tokens == 250  # 25% default
    assert limits.warning_threshold == 0.8


def test_context_limits_custom():
    """Test ContextLimits with custom values."""
    limits = ContextLimits(
        max_tokens=1000,
        max_input_tokens=800,
        max_output_tokens=200,
    )

    assert limits.max_input_tokens == 800
    assert limits.max_output_tokens == 200


def test_model_limits_predefined():
    """Test predefined model limits."""
    assert "gpt-4" in MODEL_LIMITS
    assert "gpt-3.5-turbo" in MODEL_LIMITS
    assert "claude-3-opus" in MODEL_LIMITS
    assert "llama2" in MODEL_LIMITS

    gpt4_limits = MODEL_LIMITS["gpt-4"]
    assert gpt4_limits.max_tokens == 8192


def test_context_manager_count_tokens(context_manager, sample_messages):
    """Test token counting."""
    count = context_manager.count_tokens(sample_messages)

    # ~15 tokens total (4+5+3+3)
    assert count > 0
    assert count < 100


def test_context_manager_check_limit_pass(context_manager, sample_messages):
    """Test check_limit when within limit."""
    # Messages are ~15 tokens, limit is 100, should pass
    assert context_manager.check_limit(sample_messages, max_output_tokens=20)


def test_context_manager_check_limit_fail(context_manager):
    """Test check_limit when exceeding limit."""
    # Create many large messages
    large_messages = [
        Message(role=MessageRole.USER, content="A" * 400)  # ~100 tokens each
        for _ in range(5)
    ]

    # 500 tokens exceeds limit of 100
    assert not context_manager.check_limit(large_messages)


def test_context_manager_get_available_tokens(context_manager, sample_messages):
    """Test getting available output tokens."""
    available = context_manager.get_available_tokens(sample_messages)

    # Total is 100, messages use some tokens, so we have available space
    assert available > 0
    assert available < 100


def test_context_manager_truncate_oldest_first(simple_counter):
    """Test OLDEST_FIRST truncation strategy."""
    limits = ContextLimits(max_tokens=50, max_output_tokens=10)
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
        strategy=TruncationStrategy.OLDEST_FIRST,
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),  # Keep
        Message(role=MessageRole.USER, content="Old message 1" * 5),  # Remove
        Message(role=MessageRole.ASSISTANT, content="Old response 1" * 5),  # Remove
        Message(role=MessageRole.USER, content="Recent message"),  # Keep
        Message(role=MessageRole.ASSISTANT, content="Recent response"),  # Keep
    ]

    truncated = manager.truncate(messages)

    # Should keep system, and most recent messages
    assert any(m.content == "System prompt" for m in truncated)
    assert any(m.content == "Recent message" for m in truncated)
    assert len(truncated) < len(messages)


def test_context_manager_truncate_sliding_window(simple_counter):
    """Test SLIDING_WINDOW truncation strategy."""
    limits = ContextLimits(max_tokens=50, max_output_tokens=10)
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
        strategy=TruncationStrategy.SLIDING_WINDOW,
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="System"),
        Message(role=MessageRole.USER, content="Message 1" * 10),
        Message(role=MessageRole.ASSISTANT, content="Response 1" * 10),
        Message(role=MessageRole.USER, content="Message 2"),
        Message(role=MessageRole.ASSISTANT, content="Response 2"),
    ]

    truncated = manager.truncate(messages)

    # Should keep system + most recent messages
    assert truncated[0].role == MessageRole.SYSTEM
    assert len(truncated) < len(messages)


def test_context_manager_truncate_smart(simple_counter):
    """Test SMART truncation strategy."""
    limits = ContextLimits(max_tokens=60, max_output_tokens=10)
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
        strategy=TruncationStrategy.SMART,
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),
        Message(role=MessageRole.USER, content="Old message" * 10),
        Message(role=MessageRole.ASSISTANT, content="Old response" * 10),
        Message(role=MessageRole.USER, content="Recent 1"),
        Message(role=MessageRole.ASSISTANT, content="Recent 2"),
    ]

    truncated = manager.truncate(messages)

    # Should keep system + last 2 messages
    assert truncated[0].role == MessageRole.SYSTEM
    assert truncated[-1].content == "Recent 2"
    assert truncated[-2].content == "Recent 1"


def test_context_manager_truncate_impossible(simple_counter):
    """Test truncation when output tokens exceed total limit."""
    limits = ContextLimits(max_tokens=50, max_output_tokens=60)  # Invalid!
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
    )

    messages = [Message(role=MessageRole.USER, content="Test")]

    with pytest.raises(ContextLengthExceededError):
        manager.truncate(messages)


def test_context_manager_get_stats(context_manager, sample_messages):
    """Test getting context statistics."""
    stats = context_manager.get_stats(sample_messages)

    assert "model" in stats
    assert "input_tokens" in stats
    assert "max_input_tokens" in stats
    assert "available_output_tokens" in stats
    assert "usage_percent" in stats
    assert "within_limit" in stats
    assert "message_count" in stats

    assert stats["model"] == "test-model"
    assert stats["message_count"] == 4
    assert stats["within_limit"] is True


def test_context_manager_warning_callback(context_manager):
    """Test warning callback when approaching limit."""
    warnings = []

    def warning_cb(current, max_tokens):
        warnings.append((current, max_tokens))

    context_manager.set_warning_callback(warning_cb)

    # Create messages that exceed warning threshold (90%)
    large_messages = [
        Message(role=MessageRole.USER, content="A" * 360)  # ~90 tokens
    ]

    context_manager.check_limit(large_messages)

    # Should have triggered warning
    assert len(warnings) == 1
    assert warnings[0][0] >= context_manager.limits.max_input_tokens * 0.9


def test_context_manager_default_model_limits():
    """Test default limits for unknown model."""
    manager = ContextWindowManager(model="unknown-model-xyz")

    # Should use conservative default
    assert manager.limits.max_tokens == 4096


def test_context_manager_partial_model_match():
    """Test partial model name matching."""
    manager = ContextWindowManager(model="gpt-4-0613-custom")

    # Should match "gpt-4" prefix
    assert manager.limits.max_tokens == 8192


def test_truncation_preserves_system_messages(simple_counter):
    """Test that system messages are always preserved."""
    limits = ContextLimits(max_tokens=30, max_output_tokens=10)
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
        strategy=TruncationStrategy.OLDEST_FIRST,
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="System instruction"),
        Message(role=MessageRole.USER, content="User message" * 10),
        Message(role=MessageRole.ASSISTANT, content="Assistant response" * 10),
    ]

    truncated = manager.truncate(messages)

    # System message should always be preserved
    assert any(m.role == MessageRole.SYSTEM for m in truncated)
    assert any(m.content == "System instruction" for m in truncated)


def test_context_manager_empty_messages(context_manager):
    """Test handling of empty message list."""
    count = context_manager.count_tokens([])
    assert count == 0

    assert context_manager.check_limit([])

    stats = context_manager.get_stats([])
    assert stats["input_tokens"] == 0
    assert stats["message_count"] == 0


def test_truncation_strategy_middle_out(simple_counter):
    """Test MIDDLE_OUT truncation strategy."""
    limits = ContextLimits(max_tokens=40, max_output_tokens=10)
    manager = ContextWindowManager(
        model="test",
        token_counter=simple_counter,
        limits=limits,
        strategy=TruncationStrategy.MIDDLE_OUT,
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="System"),
        Message(role=MessageRole.USER, content="First" * 10),
        Message(role=MessageRole.ASSISTANT, content="Middle" * 10),
        Message(role=MessageRole.USER, content="Last"),
    ]

    truncated = manager.truncate(messages)

    # Should keep system, last, and as much from start as fits
    assert truncated[0].role == MessageRole.SYSTEM
    assert truncated[-1].content == "Last"
