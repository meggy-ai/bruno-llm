"""Tests for retry logic."""

import pytest

from bruno_llm.base.retry import RetryConfig, RetryDecorator, retry_async
from bruno_llm.exceptions import RateLimitError


@pytest.mark.asyncio
async def test_retry_config_basic():
    """Test basic retry configuration."""
    config = RetryConfig(max_retries=3)

    assert config.max_retries == 3
    assert config.initial_delay == 1.0
    assert config.exponential_base == 2.0


@pytest.mark.asyncio
async def test_retry_config_calculate_delay():
    """Test delay calculation."""
    config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)

    # Should follow exponential pattern
    assert config.calculate_delay(0) == 1.0  # 1 * 2^0
    assert config.calculate_delay(1) == 2.0  # 1 * 2^1
    assert config.calculate_delay(2) == 4.0  # 1 * 2^2


@pytest.mark.asyncio
async def test_retry_config_max_delay():
    """Test maximum delay cap."""
    config = RetryConfig(initial_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)

    # Should be capped at max_delay
    assert config.calculate_delay(10) == 5.0


@pytest.mark.asyncio
async def test_retry_config_should_retry():
    """Test retry decision logic."""
    config = RetryConfig(max_retries=3)

    # Should retry within limit
    assert config.should_retry(Exception("test"), 0) is True
    assert config.should_retry(Exception("test"), 2) is True

    # Should not retry after max
    assert config.should_retry(Exception("test"), 3) is False


@pytest.mark.asyncio
async def test_retry_async_success():
    """Test successful execution without retries."""
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await retry_async(func)

    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_async_with_retry():
    """Test execution with retries."""
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    config = RetryConfig(max_retries=5, initial_delay=0.01)
    result = await retry_async(func, config=config)

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_async_exhausted():
    """Test retries exhausted."""
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Persistent failure")

    config = RetryConfig(max_retries=2, initial_delay=0.01)

    with pytest.raises(ValueError):
        await retry_async(func, config=config)

    assert call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_retry_decorator():
    """Test retry decorator."""
    call_count = 0

    @RetryDecorator(max_retries=3, initial_delay=0.01)
    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Failure")
        return "success"

    result = await func()

    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_with_rate_limit_error():
    """Test retry with RateLimitError."""
    call_count = 0

    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RateLimitError("Rate limited", retry_after=0.01)
        return "success"

    config = RetryConfig(max_retries=3, initial_delay=0.01)
    result = await retry_async(func, config=config)

    assert result == "success"
    assert call_count == 2
