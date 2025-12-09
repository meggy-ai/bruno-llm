"""
Rate limiting utilities for API calls.

Provides async rate limiting to prevent exceeding API rate limits.
Uses token bucket algorithm for smooth rate limiting.
"""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Controls the rate of API calls to prevent exceeding provider limits.
    Thread-safe and supports multiple concurrent requests.

    Example:
        >>> limiter = RateLimiter(requests_per_minute=60)
        >>> async with limiter:
        ...     # Make API call
        ...     response = await api_call()
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            tokens_per_minute: Maximum tokens allowed per minute (optional)
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Calculate minimum interval between requests
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

        # Token bucket for requests
        self._request_tokens = float(requests_per_minute)
        self._max_request_tokens = float(requests_per_minute)
        self._last_update = time.time()

        # Token bucket for API tokens (if specified)
        self._api_tokens = float(tokens_per_minute) if tokens_per_minute else None
        self._max_api_tokens = float(tokens_per_minute) if tokens_per_minute else None

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def _refill_tokens(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update

        if elapsed <= 0:
            return

        # Refill request tokens
        tokens_to_add = (elapsed * self.requests_per_minute) / 60.0
        self._request_tokens = min(self._max_request_tokens, self._request_tokens + tokens_to_add)

        # Refill API tokens if applicable
        if self._api_tokens is not None and self.tokens_per_minute:
            api_tokens_to_add = (elapsed * self.tokens_per_minute) / 60.0
            self._api_tokens = min(self._max_api_tokens or 0, self._api_tokens + api_tokens_to_add)

        self._last_update = now

    async def acquire(self, api_tokens: int = 0) -> None:
        """
        Acquire permission to make a request.

        Blocks until rate limit allows the request.

        Args:
            api_tokens: Number of API tokens the request will consume
        """
        async with self._lock:
            while True:
                await self._refill_tokens()

                # Check if we have enough request tokens
                if self._request_tokens < 1:
                    # Calculate wait time
                    wait_time = (1 - self._request_tokens) * (60.0 / self.requests_per_minute)
                    await asyncio.sleep(wait_time)
                    continue

                # Check if we have enough API tokens (if applicable)
                if self._api_tokens is not None and api_tokens > 0:
                    if self._api_tokens < api_tokens:
                        wait_time = (api_tokens - self._api_tokens) * (
                            60.0 / (self.tokens_per_minute or 1)
                        )
                        await asyncio.sleep(wait_time)
                        continue

                # Consume tokens
                self._request_tokens -= 1
                if self._api_tokens is not None and api_tokens > 0:
                    self._api_tokens -= api_tokens

                break

    async def __aenter__(self) -> "RateLimiter":
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dict with current token levels and limits
        """
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "available_request_tokens": self._request_tokens,
            "available_api_tokens": self._api_tokens,
            "last_update": self._last_update,
        }
