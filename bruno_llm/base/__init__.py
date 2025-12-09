"""
Base utilities and common functionality for LLM providers.

This module provides shared utilities that all provider implementations
can use, including token counting, rate limiting, retry logic, cost tracking,
caching, streaming, context management, and middleware.
"""

from bruno_llm.base.base_provider import BaseProvider
from bruno_llm.base.cache import ResponseCache, CacheEntry
from bruno_llm.base.context import (
    ContextWindowManager,
    ContextLimits,
    TruncationStrategy,
    MODEL_LIMITS,
)
from bruno_llm.base.cost_tracker import (
    CostTracker,
    UsageRecord,
    PRICING_OPENAI,
    PRICING_CLAUDE,
    PRICING_OLLAMA,
)
from bruno_llm.base.middleware import (
    Middleware,
    MiddlewareChain,
    LoggingMiddleware,
    CachingMiddleware,
    ValidationMiddleware,
    RetryMiddleware,
)
from bruno_llm.base.rate_limiter import RateLimiter
from bruno_llm.base.retry import RetryConfig, retry_async, RetryDecorator
from bruno_llm.base.streaming import (
    StreamBuffer,
    StreamStats,
    StreamAggregator,
    StreamProcessor,
    stream_with_timeout,
)
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
    # Caching
    "ResponseCache",
    "CacheEntry",
    # Context management
    "ContextWindowManager",
    "ContextLimits",
    "TruncationStrategy",
    "MODEL_LIMITS",
    # Streaming utilities
    "StreamBuffer",
    "StreamStats",
    "StreamAggregator",
    "StreamProcessor",
    "stream_with_timeout",
    # Middleware
    "Middleware",
    "MiddlewareChain",
    "LoggingMiddleware",
    "CachingMiddleware",
    "ValidationMiddleware",
    "RetryMiddleware",
]
