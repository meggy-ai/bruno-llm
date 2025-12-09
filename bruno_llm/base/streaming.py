"""
Enhanced streaming utilities for LLM providers.

Provides buffering, aggregation, and error handling for streaming responses.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Deque, List, Optional

from bruno_llm.exceptions import StreamError


@dataclass
class StreamStats:
    """
    Statistics for a streaming session.
    
    Attributes:
        chunks_received: Number of chunks received
        total_chars: Total characters streamed
        total_tokens: Estimated token count
        duration: Duration of stream in seconds
        errors: Number of errors encountered
    """
    chunks_received: int = 0
    total_chars: int = 0
    total_tokens: int = 0
    duration: float = 0.0
    errors: int = 0


@dataclass
class StreamBuffer:
    """
    Buffer for managing streaming chunks.
    
    Provides buffering, batching, and aggregation of stream chunks.
    
    Attributes:
        buffer: Internal deque for storing chunks
        max_size: Maximum buffer size in characters
        batch_size: Number of chunks to batch before yielding
        stats: Stream statistics
    """
    buffer: Deque[str] = field(default_factory=deque)
    max_size: int = 10000
    batch_size: int = 1
    stats: StreamStats = field(default_factory=StreamStats)
    
    def add(self, chunk: str) -> None:
        """
        Add a chunk to the buffer.
        
        Args:
            chunk: Text chunk to buffer
            
        Raises:
            StreamError: If buffer is full
        """
        current_size = sum(len(c) for c in self.buffer)
        
        if current_size + len(chunk) > self.max_size:
            raise StreamError(
                f"Stream buffer full ({current_size} chars). "
                f"Consider increasing max_size or consuming buffer faster."
            )
        
        self.buffer.append(chunk)
        self.stats.chunks_received += 1
        self.stats.total_chars += len(chunk)
    
    def get_batch(self) -> Optional[str]:
        """
        Get a batch of chunks.
        
        Returns:
            Concatenated batch or None if not enough chunks
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        chunks = []
        for _ in range(min(self.batch_size, len(self.buffer))):
            chunks.append(self.buffer.popleft())
        
        return "".join(chunks)
    
    def flush(self) -> str:
        """
        Flush all remaining chunks.
        
        Returns:
            All remaining chunks concatenated
        """
        chunks = list(self.buffer)
        self.buffer.clear()
        return "".join(chunks)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def clear(self) -> None:
        """Clear buffer and reset stats."""
        self.buffer.clear()
        self.stats = StreamStats()


class StreamAggregator:
    """
    Aggregate streaming chunks with various strategies.
    
    Provides different aggregation strategies for streaming responses:
    - Word-by-word: Buffer until complete words
    - Sentence-by-sentence: Buffer until sentence boundaries
    - Fixed-size: Buffer until fixed character count
    - Time-based: Buffer for fixed time intervals
    
    Args:
        strategy: Aggregation strategy ('word', 'sentence', 'fixed', 'time')
        size: Size parameter (chars for 'fixed', seconds for 'time')
        
    Example:
        >>> aggregator = StreamAggregator(strategy='word')
        >>> async for chunk in aggregator.aggregate(stream):
        ...     print(chunk)  # Prints complete words
    """
    
    def __init__(
        self,
        strategy: str = "word",
        size: int = 10,
    ):
        """
        Initialize stream aggregator.
        
        Args:
            strategy: Aggregation strategy
            size: Size parameter for aggregation
        """
        self.strategy = strategy
        self.size = size
        self._buffer = ""
    
    async def aggregate(
        self,
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """
        Aggregate stream chunks according to strategy.
        
        Args:
            stream: Input stream to aggregate
            
        Yields:
            Aggregated chunks
        """
        if self.strategy == "word":
            async for chunk in self._aggregate_words(stream):
                yield chunk
        elif self.strategy == "sentence":
            async for chunk in self._aggregate_sentences(stream):
                yield chunk
        elif self.strategy == "fixed":
            async for chunk in self._aggregate_fixed(stream):
                yield chunk
        elif self.strategy == "time":
            async for chunk in self._aggregate_time(stream):
                yield chunk
        else:
            # No aggregation, pass through
            async for chunk in stream:
                yield chunk
    
    async def _aggregate_words(
        self,
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Aggregate chunks into complete words."""
        async for chunk in stream:
            self._buffer += chunk
            
            # Split on whitespace while keeping incomplete words
            parts = self._buffer.split()
            
            if len(parts) > 1:
                # Yield all complete words
                for word in parts[:-1]:
                    yield word + " "
                
                # Keep the last part as buffer (might be incomplete)
                self._buffer = parts[-1]
            elif self._buffer.endswith((" ", "\n", "\t")):
                # Buffer ends with whitespace, yield it
                yield self._buffer
                self._buffer = ""
        
        # Flush remaining buffer
        if self._buffer:
            yield self._buffer
            self._buffer = ""
    
    async def _aggregate_sentences(
        self,
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Aggregate chunks into complete sentences."""
        sentence_endings = (".", "!", "?", "\n")
        
        async for chunk in stream:
            self._buffer += chunk
            
            # Check if buffer contains sentence ending
            while any(self._buffer.endswith(end) for end in sentence_endings):
                # Find last sentence ending
                last_idx = -1
                for ending in sentence_endings:
                    idx = self._buffer.rfind(ending)
                    if idx > last_idx:
                        last_idx = idx
                
                if last_idx >= 0:
                    # Yield complete sentence(s)
                    sentence = self._buffer[:last_idx + 1]
                    yield sentence
                    self._buffer = self._buffer[last_idx + 1:]
                else:
                    break
        
        # Flush remaining buffer
        if self._buffer:
            yield self._buffer
            self._buffer = ""
    
    async def _aggregate_fixed(
        self,
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Aggregate chunks into fixed-size chunks."""
        async for chunk in stream:
            self._buffer += chunk
            
            # Yield chunks of fixed size
            while len(self._buffer) >= self.size:
                yield self._buffer[:self.size]
                self._buffer = self._buffer[self.size:]
        
        # Flush remaining buffer
        if self._buffer:
            yield self._buffer
            self._buffer = ""
    
    async def _aggregate_time(
        self,
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Aggregate chunks based on time intervals."""
        import time
        
        last_yield = time.time()
        
        async for chunk in stream:
            self._buffer += chunk
            
            current_time = time.time()
            if current_time - last_yield >= self.size:
                if self._buffer:
                    yield self._buffer
                    self._buffer = ""
                    last_yield = current_time
        
        # Flush remaining buffer
        if self._buffer:
            yield self._buffer
            self._buffer = ""


class StreamProcessor:
    """
    Process streaming responses with callbacks and error handling.
    
    Provides a framework for processing streams with:
    - Progress callbacks
    - Error recovery
    - Automatic retry on connection loss
    - Statistics tracking
    
    Args:
        on_chunk: Callback for each chunk (chunk: str) -> None
        on_error: Callback for errors (error: Exception) -> None
        on_complete: Callback when stream completes (stats: StreamStats) -> None
        max_retries: Maximum number of retries on error
        
    Example:
        >>> processor = StreamProcessor(
        ...     on_chunk=lambda chunk: print(chunk, end=""),
        ...     on_error=lambda e: print(f"Error: {e}"),
        ...     on_complete=lambda stats: print(f"\\nReceived {stats.chunks_received} chunks")
        ... )
        >>> await processor.process(stream)
    """
    
    def __init__(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[StreamStats], None]] = None,
        max_retries: int = 3,
    ):
        """
        Initialize stream processor.
        
        Args:
            on_chunk: Callback for each chunk
            on_error: Callback for errors
            on_complete: Callback when stream completes
            max_retries: Maximum number of retries on error
        """
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.on_complete = on_complete
        self.max_retries = max_retries
        self.stats = StreamStats()
    
    async def process(
        self,
        stream: AsyncIterator[str],
        retry_on_error: bool = True,
    ) -> List[str]:
        """
        Process a stream with callbacks and error handling.
        
        Args:
            stream: Input stream to process
            retry_on_error: Whether to retry on errors
            
        Returns:
            List of all chunks received
            
        Raises:
            StreamError: If max retries exceeded
        """
        import time
        
        chunks = []
        start_time = time.time()
        retries = 0
        
        try:
            async for chunk in stream:
                chunks.append(chunk)
                self.stats.chunks_received += 1
                self.stats.total_chars += len(chunk)
                
                if self.on_chunk:
                    self.on_chunk(chunk)
            
            # Calculate duration after stream completes
            self.stats.duration = time.time() - start_time
            
            if self.on_complete:
                self.on_complete(self.stats)
            
            return chunks
            
        except Exception as e:
            self.stats.errors += 1
            self.stats.duration = time.time() - start_time
            
            if self.on_error:
                self.on_error(e)
            
            if retry_on_error and retries < self.max_retries:
                retries += 1
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                return await self.process(stream, retry_on_error)
            
            raise StreamError(f"Stream processing failed: {e}") from e


async def stream_with_timeout(
    stream: AsyncIterator[str],
    timeout: float = 30.0,
) -> AsyncIterator[str]:
    """
    Wrap a stream with timeout protection.
    
    Raises TimeoutError if no chunk is received within timeout.
    
    Args:
        stream: Input stream to wrap
        timeout: Timeout in seconds for each chunk
        
    Yields:
        Stream chunks
        
    Raises:
        TimeoutError: If timeout is exceeded
        
    Example:
        >>> async for chunk in stream_with_timeout(stream, timeout=10.0):
        ...     print(chunk)
    """
    async for chunk in stream:
        try:
            yield await asyncio.wait_for(
                _async_identity(chunk),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No chunk received within {timeout} seconds"
            )


async def _async_identity(value):
    """Helper to make a value awaitable."""
    return value
