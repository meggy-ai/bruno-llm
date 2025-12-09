"""
Response caching for LLM providers.

Provides memory-based caching to avoid redundant API calls for identical requests.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bruno_core.models import Message


@dataclass
class CacheEntry:
    """
    Cache entry for a response.

    Attributes:
        response: The cached response
        timestamp: When the entry was cached
        hit_count: Number of times this entry was accessed
        tokens: Token count for the response (if available)
    """

    response: str
    timestamp: float
    hit_count: int = 0
    tokens: Optional[int] = None


class ResponseCache:
    """
    LRU cache for LLM responses with TTL support.

    Caches responses to avoid redundant API calls. Uses message content
    and parameters as cache keys. Includes TTL (time-to-live) to ensure
    responses don't become stale.

    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration
    - Hit/miss statistics
    - Thread-safe operations (async-safe)

    Args:
        max_size: Maximum number of entries to cache (default: 1000)
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)

    Example:
        >>> cache = ResponseCache(max_size=100, ttl=300)
        >>>
        >>> # Check if response is cached
        >>> response = cache.get(messages, temperature=0.7)
        >>> if response is None:
        ...     response = await provider.generate(messages)
        ...     cache.set(messages, response, temperature=0.7)
    """

    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds for cached entries
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _generate_key(self, messages: List[Message], **kwargs: Any) -> str:
        """
        Generate cache key from messages and parameters.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Returns:
            Cache key as hex string
        """
        # Convert messages to dict representation
        messages_dict = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        # Create deterministic JSON representation
        key_data = {"messages": messages_dict, "params": {k: v for k, v in sorted(kwargs.items())}}

        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def get(self, messages: List[Message], **kwargs: Any) -> Optional[str]:
        """
        Get cached response if available and not expired.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(messages, **kwargs)

        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        current_time = time.time()

        # Check if entry has expired
        if current_time - entry.timestamp > self._ttl:
            # Remove expired entry
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1

        return entry.response

    def set(
        self, messages: List[Message], response: str, tokens: Optional[int] = None, **kwargs: Any
    ) -> None:
        """
        Cache a response.

        Args:
            messages: List of conversation messages
            response: The response to cache
            tokens: Token count for the response (optional)
            **kwargs: Additional generation parameters
        """
        key = self._generate_key(messages, **kwargs)

        # Create cache entry
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            hit_count=0,
            tokens=tokens,
        )

        # Add to cache
        self._cache[key] = entry
        self._cache.move_to_end(key)

        # Evict oldest entry if max size exceeded
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def invalidate(self, messages: List[Message], **kwargs: Any) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters

        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._generate_key(messages, **kwargs)

        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl": self._ttl,
        }

    def get_size_bytes(self) -> int:
        """
        Estimate cache size in bytes.

        Returns:
            Approximate cache size in bytes
        """
        total_size = 0
        for key, entry in self._cache.items():
            # Key size
            total_size += len(key.encode())
            # Response size
            total_size += len(entry.response.encode())
            # Overhead for entry metadata (~100 bytes)
            total_size += 100

        return total_size

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items() if current_time - entry.timestamp > self._ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_top_entries(self, n: int = 10) -> List[Tuple[str, CacheEntry]]:
        """
        Get top N most frequently accessed entries.

        Args:
            n: Number of entries to return

        Returns:
            List of (key, entry) tuples sorted by hit count
        """
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].hit_count, reverse=True)
        return sorted_entries[:n]
