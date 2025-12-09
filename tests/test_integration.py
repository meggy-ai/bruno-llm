"""
Integration tests for bruno-llm.

These tests verify end-to-end functionality including:
- Real provider connections (when available)
- Factory pattern integration
- Advanced features integration
- Performance characteristics

Tests marked with @pytest.mark.integration require actual provider access
and are skipped by default. Run with: pytest -m integration
"""

import os
import asyncio
from typing import List

import pytest

from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory
from bruno_llm.base import (
    ResponseCache,
    ContextWindowManager,
    StreamAggregator,
    CachingMiddleware,
    LoggingMiddleware,
)


# Integration test markers
pytestmark = pytest.mark.integration


def has_ollama() -> bool:
    """Check if Ollama is available."""
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def has_openai_key() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def sample_messages() -> List[Message]:
    """Create sample conversation messages."""
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is 2+2?"),
    ]


@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
@pytest.mark.wip
@pytest.mark.asyncio
async def test_ollama_integration_basic(sample_messages):
    """Test basic Ollama integration."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Test connection
    assert await provider.check_connection()
    
    # Test generation
    response = await provider.generate(sample_messages, max_tokens=50)
    assert isinstance(response, str)
    assert len(response) > 0
    
    await provider.close()


@pytest.mark.wip
@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
@pytest.mark.asyncio
async def test_ollama_integration_streaming(sample_messages):
    """Test Ollama streaming integration."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    chunks = []
    async for chunk in provider.stream(sample_messages, max_tokens=50):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0
    
    await provider.close()


@pytest.mark.skipif(not has_openai_key(), reason="OpenAI API key not configured")
@pytest.mark.asyncio
async def test_openai_integration_basic(sample_messages):
    """Test basic OpenAI integration."""
    provider = LLMFactory.create_from_env("openai")
    
    # Test connection
    assert await provider.check_connection()
    
    # Test generation
    response = await provider.generate(sample_messages, max_tokens=50)
    assert isinstance(response, str)
    assert len(response) > 0
    
    await provider.close()


@pytest.mark.skipif(not has_openai_key(), reason="OpenAI API key not configured")
@pytest.mark.asyncio
async def test_openai_integration_streaming(sample_messages):
    """Test OpenAI streaming integration."""
    provider = LLMFactory.create_from_env("openai")
    
    chunks = []
    async for chunk in provider.stream(sample_messages, max_tokens=50):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0
    
    await provider.close()


@pytest.mark.asyncio
async def test_factory_fallback_integration():
    """Test factory fallback mechanism."""
    # Try OpenAI first, fall back to Ollama
    providers = ["openai", "ollama"]
    configs = [
        {"api_key": "invalid-key"},  # Should fail
        {"base_url": "http://localhost:11434", "model": "llama2"},
    ]
    
    # This will try OpenAI (fail) then Ollama
    # If neither works, it raises an error
    try:
        provider = await LLMFactory.create_with_fallback(providers, configs)
        assert provider is not None
        await provider.close()
    except Exception as e:
        # Expected if no providers are available
        pytest.skip(f"No providers available: {e}")


@pytest.mark.wip
@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
@pytest.mark.asyncio
async def test_caching_integration(sample_messages):
    """Test response caching integration."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    cache = ResponseCache(max_size=10, ttl=60)
    
    # First request - cache miss
    response1 = await provider.generate(sample_messages, temperature=0.0)
    cache.set(sample_messages, response1, temperature=0.0)
    
    # Second request - cache hit
    cached_response = cache.get(sample_messages, temperature=0.0)
    assert cached_response == response1
    
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["size"] == 1
    
    await provider.close()


@pytest.mark.wip
@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
@pytest.mark.asyncio
async def test_context_manager_integration(sample_messages):
    """Test context window manager integration."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Create context manager with small limit
    context_manager = ContextWindowManager(
        model="llama2",
        limits=ContextLimits(max_tokens=100, max_output_tokens=20),
    )
    
    # Check if messages fit
    if not context_manager.check_limit(sample_messages):
        sample_messages = context_manager.truncate(sample_messages)
    
    # Generate with truncated messages
    response = await provider.generate(sample_messages, max_tokens=20)
    assert isinstance(response, str)
    
    await provider.close()


@pytest.mark.wip
@pytest.mark.skipif(not has_ollama(), reason="Ollama not available")
@pytest.mark.asyncio
async def test_stream_aggregation_integration(sample_messages):
    """Test stream aggregation integration."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    aggregator = StreamAggregator(strategy="word")
    
    # Stream with word aggregation
    words = []
    stream = provider.stream(sample_messages, max_tokens=30)
    
    async for word in aggregator.aggregate(stream):
        words.append(word)
    
    # Should have received complete words
    assert len(words) > 0
    
    await provider.close()


@pytest.mark.wip
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent requests to provider."""
    if not has_ollama():
        pytest.skip("Ollama not available")
    
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Create multiple concurrent requests
    messages_list = [
        [Message(role=MessageRole.USER, content=f"Count to {i}")]
        for i in range(1, 4)
    ]
    
    # Execute concurrently
    tasks = [
        provider.generate(messages, max_tokens=20)
        for messages in messages_list
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check all succeeded
    successful = [r for r in responses if isinstance(r, str)]
    assert len(successful) > 0
    
    await provider.close()


@pytest.mark.asyncio
async def test_cost_tracking_integration():
    """Test cost tracking integration."""
    if not has_openai_key():
        pytest.skip("OpenAI API key not configured")
    
    provider = LLMFactory.create_from_env("openai")
    
    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = await provider.generate(messages, max_tokens=10)
    
    # Check cost was tracked
    cost_tracker = provider.cost_tracker
    assert cost_tracker.get_total_cost() > 0
    assert cost_tracker.get_request_count() >= 1
    
    await provider.close()


@pytest.mark.asyncio
async def test_model_info_integration():
    """Test getting model information."""
    if not has_ollama():
        pytest.skip("Ollama not available")
    
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    info = provider.get_model_info()
    assert "model" in info
    assert "provider" in info
    assert info["provider"] == "ollama"
    
    await provider.close()


@pytest.mark.asyncio
async def test_list_models_integration():
    """Test listing available models."""
    if not has_ollama():
        pytest.skip("Ollama not available")
    
    provider = LLMFactory.create("ollama")
    
    models = await provider.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    
    await provider.close()


@pytest.mark.wip
@pytest.mark.asyncio
async def test_system_prompt_integration(sample_messages):
    """Test system prompt handling."""
    if not has_ollama():
        pytest.skip("Ollama not available")
    
    provider = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Set system prompt
    provider.set_system_prompt("You are a math tutor.")
    assert provider.get_system_prompt() == "You are a math tutor."
    
    # Generate with system prompt
    user_messages = [Message(role=MessageRole.USER, content="What is 2+2?")]
    response = await provider.generate(user_messages, max_tokens=20)
    assert isinstance(response, str)
    
    await provider.close()


@pytest.mark.asyncio
async def test_error_handling_integration():
    """Test error handling in real scenarios."""
    # Test with invalid model
    provider = LLMFactory.create("ollama", {"model": "nonexistent-model-xyz"})
    
    messages = [Message(role=MessageRole.USER, content="Hello")]
    
    with pytest.raises(Exception):  # Should raise ModelNotFoundError or similar
        await provider.generate(messages)
    
    await provider.close()


@pytest.mark.asyncio
async def test_timeout_integration():
    """Test timeout handling."""
    if not has_ollama():
        pytest.skip("Ollama not available")
    
    provider = LLMFactory.create("ollama", {
        "model": "llama2",
        "timeout": 0.001  # Very short timeout
    })
    
    messages = [Message(role=MessageRole.USER, content="Write a long story")]
    
    # May timeout or succeed depending on system speed
    try:
        response = await provider.generate(messages, max_tokens=100)
        # If it succeeds, that's also okay
        assert isinstance(response, str)
    except Exception as e:
        # Timeout is expected
        assert "timeout" in str(e).lower() or "timed out" in str(e).lower()
    
    await provider.close()


# Import ContextLimits for the test
from bruno_llm.base.context import ContextLimits
