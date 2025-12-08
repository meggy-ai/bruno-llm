"""
Base utilities and common functionality for LLM providers.

This module provides shared utilities that all provider implementations
can use, including token counting, rate limiting, retry logic, and cost tracking.
"""

from bruno_llm.base.base_provider import BaseProvider
from bruno_llm.base.cost_tracker import (
    CostTracker,
    UsageRecord,
    PRICING_OPENAI,
    PRICING_CLAUDE,
    PRICING_OLLAMA,
)
from bruno_llm.base.rate_limiter import RateLimiter
from bruno_llm.base.retry import RetryConfig, retry_async, RetryDecorator
from bruno_llm.base.token_counter import (
    TokenCounter,
    SimpleTokenCounter,
    TikTokenCounter,
    create_token_counter,
)

__all__ = [
    # Base provider
    "BaseProvider",
    # Token counting
    "TokenCounter",
    "SimpleTokenCounter",
    "TikTokenCounter",
    "create_token_counter",
    # Rate limiting
    "RateLimiter",
    # Retry logic
    "RetryConfig",
    "retry_async",
    "RetryDecorator",
    # Cost tracking
    "CostTracker",
    "UsageRecord",
    "PRICING_OPENAI",
    "PRICING_CLAUDE",
    "PRICING_OLLAMA",
]
