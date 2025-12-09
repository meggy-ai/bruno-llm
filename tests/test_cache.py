"""
Tests for response cache functionality.
"""

import time

import pytest

from bruno_core.models import Message, MessageRole
from bruno_llm.base.cache import CacheEntry, ResponseCache


@pytest.fixture
def cache():
    """Create a response cache for testing."""
    return ResponseCache(max_size=10, ttl=2)  # Small cache with 2s TTL


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello, world!"),
    ]


def test_cache_entry():
    """Test CacheEntry dataclass."""
    entry = CacheEntry(
        response="test response",
        timestamp=time.time(),
        hit_count=5,
        tokens=100,
    )

    assert entry.response == "test response"
    assert entry.hit_count == 5
    assert entry.tokens == 100


def test_cache_set_and_get(cache, sample_messages):
    """Test basic cache set and get operations."""
    response = "Hello! How can I help you?"

    # Set cache entry
    cache.set(sample_messages, response, temperature=0.7)

    # Get from cache
    cached = cache.get(sample_messages, temperature=0.7)
    assert cached == response

    # Stats should show 1 hit
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["size"] == 1


def test_cache_miss(cache, sample_messages):
    """Test cache miss."""
    # Try to get non-existent entry
    result = cache.get(sample_messages, temperature=0.7)
    assert result is None

    # Stats should show 1 miss
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_cache_key_sensitivity(cache, sample_messages):
    """Test that cache keys are sensitive to parameters."""
    response1 = "Response 1"
    response2 = "Response 2"

    # Cache with different parameters
    cache.set(sample_messages, response1, temperature=0.7)
    cache.set(sample_messages, response2, temperature=0.9)

    # Should get different responses
    assert cache.get(sample_messages, temperature=0.7) == response1
    assert cache.get(sample_messages, temperature=0.9) == response2

    # Cache should have 2 entries
    assert cache.get_stats()["size"] == 2


def test_cache_ttl_expiration(cache, sample_messages):
    """Test cache entry expiration."""
    response = "Test response"

    # Set cache entry
    cache.set(sample_messages, response)

    # Should be available immediately
    assert cache.get(sample_messages) == response

    # Wait for TTL to expire
    time.sleep(2.5)

    # Should be expired now
    result = cache.get(sample_messages)
    assert result is None

    # Stats should show 1 hit, 1 miss
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_max_size_eviction(cache):
    """Test LRU eviction when max size is reached."""
    messages_list = []

    # Fill cache to max (10 entries)
    for i in range(10):
        messages = [Message(role=MessageRole.USER, content=f"Message {i}")]
        messages_list.append(messages)
        cache.set(messages, f"Response {i}")

    assert cache.get_stats()["size"] == 10

    # Add one more entry - should evict oldest
    messages_new = [Message(role=MessageRole.USER, content="New message")]
    cache.set(messages_new, "New response")

    # Size should still be 10
    assert cache.get_stats()["size"] == 10

    # Oldest entry should be evicted
    assert cache.get(messages_list[0]) is None

    # Newest entry should be present
    assert cache.get(messages_new) == "New response"


def test_cache_hit_count(cache, sample_messages):
    """Test that hit count is tracked correctly."""
    response = "Test response"
    cache.set(sample_messages, response)

    # Access multiple times
    for _ in range(5):
        cache.get(sample_messages)

    # Stats should show 5 hits
    stats = cache.get_stats()
    assert stats["hits"] == 5
    assert stats["misses"] == 0


def test_cache_clear(cache, sample_messages):
    """Test cache clearing."""
    cache.set(sample_messages, "Response 1")
    cache.set(sample_messages, "Response 2", temperature=0.9)

    assert cache.get_stats()["size"] == 2

    # Clear cache
    cache.clear()

    # Should be empty
    assert cache.get_stats()["size"] == 0
    assert cache.get_stats()["hits"] == 0
    assert cache.get_stats()["misses"] == 0


def test_cache_invalidate(cache, sample_messages):
    """Test specific entry invalidation."""
    cache.set(sample_messages, "Response 1")
    cache.set(sample_messages, "Response 2", temperature=0.9)

    # Invalidate specific entry
    result = cache.invalidate(sample_messages, temperature=0.9)
    assert result is True

    # Entry should be gone
    assert cache.get(sample_messages, temperature=0.9) is None

    # Other entry should still exist
    assert cache.get(sample_messages) == "Response 1"

    # Invalidating non-existent entry
    result = cache.invalidate(sample_messages, temperature=1.0)
    assert result is False


def test_cache_size_bytes(cache, sample_messages):
    """Test cache size estimation."""
    initial_size = cache.get_size_bytes()
    assert initial_size == 0

    # Add entry
    cache.set(sample_messages, "A" * 1000)  # 1000 chars

    size_after = cache.get_size_bytes()
    assert size_after > 1000  # Should be at least 1000 + overhead


def test_cache_cleanup_expired(cache):
    """Test cleanup of expired entries."""
    messages_list = []

    # Add multiple entries
    for i in range(5):
        messages = [Message(role=MessageRole.USER, content=f"Message {i}")]
        messages_list.append(messages)
        cache.set(messages, f"Response {i}")

    assert cache.get_stats()["size"] == 5

    # Wait for expiration
    time.sleep(2.5)

    # Cleanup
    removed = cache.cleanup_expired()
    assert removed == 5
    assert cache.get_stats()["size"] == 0


def test_cache_top_entries(cache):
    """Test getting top accessed entries."""
    messages_list = []

    # Add entries with different access counts
    for i in range(5):
        messages = [Message(role=MessageRole.USER, content=f"Message {i}")]
        messages_list.append(messages)
        cache.set(messages, f"Response {i}")

        # Access each entry i times
        for _ in range(i):
            cache.get(messages)

    # Get top 3 entries
    top = cache.get_top_entries(3)
    assert len(top) == 3

    # Should be sorted by hit count (highest first)
    # Entry 4 should have 4 hits, entry 3 should have 3 hits, etc.
    assert top[0][1].hit_count == 4
    assert top[1][1].hit_count == 3
    assert top[2][1].hit_count == 2


def test_cache_hit_rate(cache, sample_messages):
    """Test hit rate calculation."""
    cache.set(sample_messages, "Response")

    # 5 hits
    for _ in range(5):
        cache.get(sample_messages)

    # 3 misses
    for i in range(3):
        messages = [Message(role=MessageRole.USER, content=f"Different {i}")]
        cache.get(messages)

    stats = cache.get_stats()
    assert stats["hits"] == 5
    assert stats["misses"] == 3
    assert stats["hit_rate"] == 5 / 8  # 62.5%


def test_cache_with_tokens(cache, sample_messages):
    """Test caching with token count."""
    cache.set(sample_messages, "Response", tokens=50)

    # Access the entry to get the CacheEntry
    cache.get(sample_messages)

    # Check entry has token count
    key = cache._generate_key(sample_messages)
    entry = cache._cache[key]
    assert entry.tokens == 50
