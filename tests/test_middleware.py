"""
Tests for middleware system.
"""

import pytest

from bruno_core.models import Message, MessageRole
from bruno_llm.base.cache import ResponseCache
from bruno_llm.base.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    RetryMiddleware,
    ValidationMiddleware,
)


class TestMiddleware(Middleware):
    """Test middleware for tracking calls."""
    
    def __init__(self):
        self.before_calls = []
        self.after_calls = []
        self.chunk_calls = []
        self.error_calls = []
    
    async def before_request(self, messages, **kwargs):
        self.before_calls.append((messages, kwargs))
        return messages, kwargs
    
    async def after_response(self, messages, response, **kwargs):
        self.after_calls.append((messages, response, kwargs))
        return response
    
    async def on_stream_chunk(self, chunk, **kwargs):
        self.chunk_calls.append((chunk, kwargs))
        return chunk
    
    async def on_error(self, error, messages, **kwargs):
        self.error_calls.append((error, messages, kwargs))


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    return [
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]


@pytest.mark.asyncio
async def test_middleware_before_request(sample_messages):
    """Test before_request hook."""
    middleware = TestMiddleware()
    
    messages, kwargs = await middleware.before_request(
        sample_messages,
        temperature=0.7
    )
    
    assert len(middleware.before_calls) == 1
    assert middleware.before_calls[0][0] == sample_messages
    assert middleware.before_calls[0][1] == {"temperature": 0.7}


@pytest.mark.asyncio
async def test_middleware_after_response(sample_messages):
    """Test after_response hook."""
    middleware = TestMiddleware()
    
    response = await middleware.after_response(
        sample_messages,
        "Test response",
        temperature=0.7
    )
    
    assert len(middleware.after_calls) == 1
    assert middleware.after_calls[0][1] == "Test response"
    assert response == "Test response"


@pytest.mark.asyncio
async def test_middleware_on_stream_chunk():
    """Test on_stream_chunk hook."""
    middleware = TestMiddleware()
    
    chunk = await middleware.on_stream_chunk("Hello", temperature=0.7)
    
    assert len(middleware.chunk_calls) == 1
    assert middleware.chunk_calls[0][0] == "Hello"
    assert chunk == "Hello"


@pytest.mark.asyncio
async def test_middleware_on_error(sample_messages):
    """Test on_error hook."""
    middleware = TestMiddleware()
    error = ValueError("Test error")
    
    await middleware.on_error(error, sample_messages)
    
    assert len(middleware.error_calls) == 1
    assert middleware.error_calls[0][0] == error


@pytest.mark.asyncio
async def test_logging_middleware(sample_messages):
    """Test logging middleware."""
    # Use a simple list to capture log calls
    logs = []
    
    class MockLogger:
        def info(self, msg, **kwargs):
            logs.append(("info", msg, kwargs))
        
        def error(self, msg, **kwargs):
            logs.append(("error", msg, kwargs))
    
    middleware = LoggingMiddleware(logger=MockLogger(), log_messages=True)
    
    # Before request
    await middleware.before_request(sample_messages, temperature=0.7)
    assert len(logs) == 1
    assert logs[0][0] == "info"
    assert "llm_request" in str(logs[0])
    
    # After response
    await middleware.after_response(sample_messages, "Response", temperature=0.7)
    assert len(logs) == 2
    assert "llm_response" in str(logs[1])
    
    # On error
    await middleware.on_error(ValueError("Test"), sample_messages)
    assert len(logs) == 3
    assert logs[2][0] == "error"


@pytest.mark.asyncio
async def test_caching_middleware(sample_messages):
    """Test caching middleware."""
    cache = ResponseCache(max_size=10, ttl=60)
    middleware = CachingMiddleware(cache)
    
    # After response should cache
    await middleware.after_response(sample_messages, "Response", temperature=0.7)
    
    # Check cache
    cached = cache.get(sample_messages, temperature=0.7)
    assert cached == "Response"


@pytest.mark.asyncio
async def test_caching_middleware_streaming():
    """Test caching middleware with streaming."""
    cache = ResponseCache()
    middleware = CachingMiddleware(cache, cache_streaming=True)
    
    # Process stream chunks
    await middleware.on_stream_chunk("Hello")
    await middleware.on_stream_chunk(" world")
    
    # Should collect chunks
    assert middleware._current_stream_chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_validation_middleware_success(sample_messages):
    """Test validation middleware with valid input."""
    middleware = ValidationMiddleware(
        max_message_length=1000,
        allowed_roles=["user", "assistant", "system"],
    )
    
    # Should pass validation
    messages, kwargs = await middleware.before_request(
        sample_messages,
        temperature=0.7
    )
    
    assert messages == sample_messages


@pytest.mark.asyncio
async def test_validation_middleware_length_error():
    """Test validation middleware with message too long."""
    middleware = ValidationMiddleware(max_message_length=10)
    
    messages = [Message(role=MessageRole.USER, content="A" * 100)]
    
    with pytest.raises(ValueError) as exc_info:
        await middleware.before_request(messages)
    
    assert "exceeds max length" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_middleware_role_error():
    """Test validation middleware with invalid role."""
    middleware = ValidationMiddleware(allowed_roles=["user", "assistant"])
    
    messages = [Message(role=MessageRole.SYSTEM, content="System prompt")]
    
    with pytest.raises(ValueError) as exc_info:
        await middleware.before_request(messages)
    
    assert "Invalid message role" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_middleware_required_params():
    """Test validation middleware with required parameters."""
    middleware = ValidationMiddleware(required_params=["temperature", "max_tokens"])
    
    messages = [Message(role=MessageRole.USER, content="Test")]
    
    # Missing required params
    with pytest.raises(ValueError) as exc_info:
        await middleware.before_request(messages)
    
    assert "Required parameter missing" in str(exc_info.value)
    
    # With required params
    messages_out, kwargs_out = await middleware.before_request(
        messages,
        temperature=0.7,
        max_tokens=100
    )
    assert messages_out == messages


@pytest.mark.asyncio
async def test_retry_middleware(sample_messages):
    """Test retry middleware."""
    middleware = RetryMiddleware(max_retries=3, base_delay=0.01)
    
    # Before request should reset counter
    await middleware.before_request(sample_messages)
    assert middleware.retry_count == 0
    
    # On error should increment and delay
    await middleware.on_error(ValueError("Test"), sample_messages)
    assert middleware.retry_count == 1


@pytest.mark.asyncio
async def test_middleware_chain(sample_messages):
    """Test middleware chain execution."""
    middleware1 = TestMiddleware()
    middleware2 = TestMiddleware()
    middleware3 = TestMiddleware()
    
    chain = MiddlewareChain([middleware1, middleware2, middleware3])
    
    # Before request - should call all in order
    messages, kwargs = await chain.before_request(sample_messages, temperature=0.7)
    
    assert len(middleware1.before_calls) == 1
    assert len(middleware2.before_calls) == 1
    assert len(middleware3.before_calls) == 1
    
    # After response - should call all in reverse order
    response = await chain.after_response(sample_messages, "Response")
    
    assert len(middleware1.after_calls) == 1
    assert len(middleware2.after_calls) == 1
    assert len(middleware3.after_calls) == 1


@pytest.mark.asyncio
async def test_middleware_chain_modifies_messages(sample_messages):
    """Test middleware chain can modify messages."""
    
    class ModifyingMiddleware(Middleware):
        def __init__(self, suffix):
            self.suffix = suffix
        
        async def before_request(self, messages, **kwargs):
            # Add suffix to each message
            modified = [
                Message(role=m.role, content=m.content + self.suffix)
                for m in messages
            ]
            return modified, kwargs
        
        async def after_response(self, messages, response, **kwargs):
            return response + self.suffix
    
    chain = MiddlewareChain([
        ModifyingMiddleware(" [A]"),
        ModifyingMiddleware(" [B]"),
    ])
    
    messages, _ = await chain.before_request(sample_messages)
    
    # Both suffixes should be applied
    assert messages[0].content.endswith(" [A] [B]")
    
    # After response
    response = await chain.after_response(sample_messages, "Test")
    assert response == "Test [B] [A]"  # Reversed order


@pytest.mark.asyncio
async def test_middleware_chain_on_error(sample_messages):
    """Test middleware chain error handling."""
    middleware1 = TestMiddleware()
    middleware2 = TestMiddleware()
    
    chain = MiddlewareChain([middleware1, middleware2])
    
    error = ValueError("Test error")
    await chain.on_error(error, sample_messages)
    
    # Both should receive error
    assert len(middleware1.error_calls) == 1
    assert len(middleware2.error_calls) == 1


@pytest.mark.asyncio
async def test_middleware_chain_stream_chunks():
    """Test middleware chain streaming."""
    middleware1 = TestMiddleware()
    middleware2 = TestMiddleware()
    
    chain = MiddlewareChain([middleware1, middleware2])
    
    chunk = await chain.on_stream_chunk("Hello")
    
    assert len(middleware1.chunk_calls) == 1
    assert len(middleware2.chunk_calls) == 1
    assert chunk == "Hello"


@pytest.mark.asyncio
async def test_middleware_chain_empty():
    """Test empty middleware chain."""
    chain = MiddlewareChain([])
    
    messages = [Message(role=MessageRole.USER, content="Test")]
    
    # Should pass through unchanged
    result_messages, result_kwargs = await chain.before_request(messages, temp=0.7)
    assert result_messages == messages
    assert result_kwargs == {"temp": 0.7}
    
    response = await chain.after_response(messages, "Response")
    assert response == "Response"


@pytest.mark.asyncio
async def test_logging_middleware_without_message_content(sample_messages):
    """Test logging middleware doesn't log message content when disabled."""
    logs = []
    
    class MockLogger:
        def info(self, msg, **kwargs):
            logs.append(("info", msg, kwargs))
        
        def error(self, msg, **kwargs):
            logs.append(("error", msg, kwargs))
    
    middleware = LoggingMiddleware(logger=MockLogger(), log_messages=False)
    
    await middleware.before_request(sample_messages, temperature=0.7)
    
    # Should not contain message content
    log_data = logs[0][2]
    assert "messages" not in log_data
    assert log_data["message_count"] == 2
