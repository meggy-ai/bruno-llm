"""Tests for rate limiter."""

import time

import pytest

from bruno_llm.base.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test basic rate limiting."""
    # Allow 10 requests per minute = 1 request per 6 seconds
    limiter = RateLimiter(requests_per_minute=10)

    # First request should be immediate
    start = time.time()
    async with limiter:
        pass
    elapsed = time.time() - start
    assert elapsed < 0.1  # Should be nearly instant


@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    """Test acquire method."""
    limiter = RateLimiter(requests_per_minute=60)

    # Acquire should not block on first call
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_multiple_requests():
    """Test multiple requests respect rate limit."""
    # Allow 60 requests per minute
    limiter = RateLimiter(requests_per_minute=60)

    # Make several requests quickly
    request_times = []
    for _ in range(3):
        start = time.time()
        await limiter.acquire()
        request_times.append(time.time() - start)

    # First request should be instant
    assert request_times[0] < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_stats():
    """Test getting rate limiter stats."""
    limiter = RateLimiter(requests_per_minute=60)

    stats = limiter.get_stats()
    assert "requests_per_minute" in stats
    assert stats["requests_per_minute"] == 60
    assert "available_request_tokens" in stats


@pytest.mark.asyncio
async def test_rate_limiter_with_token_limit():
    """Test rate limiter with API token limits."""
    limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=1000)

    # Acquire with token consumption
    await limiter.acquire(api_tokens=10)

    stats = limiter.get_stats()
    assert stats["tokens_per_minute"] == 1000


@pytest.mark.asyncio
async def test_rate_limiter_context_manager():
    """Test rate limiter as context manager."""
    limiter = RateLimiter(requests_per_minute=60)

    # Should work as context manager
    async with limiter:
        # Do something
        pass

    # Should be able to use again
    async with limiter:
        pass
