# Bruno-LLM API Reference

## Overview

Bruno-LLM provides a unified interface for interacting with multiple LLM and embedding providers through a consistent API. This package implements the `LLMInterface` and `EmbeddingInterface` from bruno-core, enabling seamless integration with the broader Bruno AI ecosystem.

## Core Interfaces

### LLMInterface

All LLM providers implement the `bruno_core.interfaces.LLMInterface`:

```python
from bruno_core.interfaces import LLLInterface
from bruno_core.models import Message, MessageRole

class YourProvider(LLMInterface):
    async def generate(self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate a complete response."""

    async def stream(self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> AsyncIterator[str]:
        """Stream response tokens."""

    async def check_connection(self) -> bool:
        """Check if the provider is accessible."""

    async def list_models(self) -> List[str]:
        """List available models."""

    def get_token_count(self, text: str) -> int:
        """Estimate token count for text."""

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt for conversations."""

    def get_system_prompt(self) -> Optional[str]:
        """Get current system prompt."""
```

### EmbeddingInterface

All embedding providers implement the `bruno_core.interfaces.EmbeddingInterface`:

```python
from bruno_core.interfaces import EmbeddingInterface

class YourEmbeddingProvider(EmbeddingInterface):
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""

    async def embed_message(self, message: Message) -> List[float]:
        """Generate embeddings for a message object."""

    async def check_connection(self) -> bool:
        """Check if the provider is accessible."""

    def get_dimension(self) -> int:
        """Get the embedding dimension."""

    def get_model_name(self) -> str:
        """Get the current model name."""

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for processing."""

    def supports_batch(self) -> bool:
        """Check if provider supports batch processing."""

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
```

## LLM Providers

### OpenAI Provider

#### Basic Usage

```python
from bruno_llm.providers.openai import OpenAIProvider
from bruno_core.models import Message, MessageRole

# Initialize provider
provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4",
    organization="org-...",  # Optional
    timeout=30.0
)

# Generate response
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Hello!")
]

response = await provider.generate(
    messages=messages,
    temperature=0.7,
    max_tokens=1000
)
print(response)

# Stream response
async for chunk in provider.stream(messages, temperature=0.7):
    print(chunk, end="", flush=True)
```

#### Configuration

```python
from bruno_llm.providers.openai import OpenAIConfig

config = OpenAIConfig(
    api_key="sk-...",
    model="gpt-4",
    organization="org-...",
    base_url="https://api.openai.com/v1",  # Custom endpoint
    timeout=30.0,
    max_retries=3,
    batch_size=100,
    track_cost=True
)

provider = OpenAIProvider(config=config)
```

#### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4"
export OPENAI_ORG_ID="org-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_TIMEOUT="30.0"
export OPENAI_MAX_RETRIES="3"
```

#### Supported Models

- `gpt-4` - GPT-4 base model
- `gpt-4-turbo` - GPT-4 Turbo model
- `gpt-3.5-turbo` - GPT-3.5 Turbo model
- `gpt-4o` - GPT-4 Omni model
- `gpt-4o-mini` - GPT-4 Omni mini model

### Ollama Provider

#### Basic Usage

```python
from bruno_llm.providers.ollama import OllamaProvider
from bruno_core.models import Message, MessageRole

# Initialize provider (assumes Ollama running on localhost:11434)
provider = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30.0
)

# Generate response
messages = [
    Message(role=MessageRole.USER, content="Explain quantum computing")
]

response = await provider.generate(
    messages=messages,
    temperature=0.8,
    max_tokens=500  # Translated to num_predict for Ollama
)
print(response)

# Stream response
async for chunk in provider.stream(messages, temperature=0.8):
    print(chunk, end="", flush=True)
```

#### Configuration

```python
from bruno_llm.providers.ollama import OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30.0,
    keep_alive="5m",
    num_ctx=4096,
    repeat_penalty=1.1
)

provider = OllamaProvider(config=config)
```

#### Environment Variables

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama2"
export OLLAMA_TIMEOUT="30.0"
export OLLAMA_KEEP_ALIVE="5m"
export OLLAMA_NUM_CTX="4096"
```

#### Supported Models

Popular Ollama models:
- `llama2` - Llama 2 base model
- `llama2:13b` - Llama 2 13B parameter model
- `codellama` - Code Llama for coding tasks
- `mistral` - Mistral 7B model
- `neural-chat` - Intel's Neural Chat model
- `starling-lm` - Starling language model

## Embedding Providers

### OpenAI Embeddings

#### Basic Usage

```python
from bruno_llm.providers.openai import OpenAIEmbeddingProvider

# Initialize provider
embedder = OpenAIEmbeddingProvider(
    api_key="sk-...",
    model="text-embedding-ada-002"
)

# Single text embedding
embedding = await embedder.embed_text("Hello world")
print(f"Embedding dimension: {len(embedding)}")

# Multiple texts
texts = ["Hello", "World", "OpenAI"]
embeddings = await embedder.embed_texts(texts)

# Message embedding
from bruno_core.models import Message, MessageRole
message = Message(role=MessageRole.USER, content="Hello")
msg_embedding = await embedder.embed_message(message)

# Similarity calculation
similarity = embedder.calculate_similarity(embedding, msg_embedding)
print(f"Similarity: {similarity}")
```

#### Configuration

```python
from bruno_llm.providers.openai import OpenAIEmbeddingConfig

config = OpenAIEmbeddingConfig(
    api_key="sk-...",
    model="text-embedding-3-large",
    dimensions=1024,  # For text-embedding-3-* models
    batch_size=100,
    timeout=30.0
)

embedder = OpenAIEmbeddingProvider(config=config)
```

#### Supported Models

- `text-embedding-ada-002` - 1536 dimensions, most cost-effective
- `text-embedding-3-small` - Up to 1536 dimensions, improved performance
- `text-embedding-3-large` - Up to 3072 dimensions, highest performance

### Ollama Embeddings

#### Basic Usage

```python
from bruno_llm.providers.ollama import OllamaEmbeddingProvider

# Initialize provider
embedder = OllamaEmbeddingProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    batch_size=32
)

# Single text embedding
embedding = await embedder.embed_text("Local embedding generation")
print(f"Dimension: {embedder.get_dimension()}")

# Batch processing
texts = ["Local AI", "Privacy-focused", "No API costs"]
embeddings = await embedder.embed_texts(texts)

# Check connection
is_available = await embedder.check_connection()
print(f"Ollama available: {is_available}")
```

#### Configuration

```python
from bruno_llm.providers.ollama import OllamaEmbeddingConfig

config = OllamaEmbeddingConfig(
    base_url="http://localhost:11434",
    model="mxbai-embed-large",
    timeout=60.0,
    batch_size=16
)

embedder = OllamaEmbeddingProvider(config=config)
```

#### Supported Models

- `nomic-embed-text` - 768 dimensions, efficient for most tasks
- `mxbai-embed-large` - 1024 dimensions, higher quality embeddings
- `snowflake-arctic-embed` - 1024 dimensions, specialized for retrieval

## Factory Pattern

### LLM Factory

```python
from bruno_llm.factory import LLMFactory

# List available providers
providers = LLMFactory.list_providers()
print(providers)  # ['openai', 'ollama']

# Create provider directly
llm = LLMFactory.create(
    provider="openai",
    config={"api_key": "sk-...", "model": "gpt-4"}
)

# Create from environment variables
llm = LLMFactory.create_from_env(provider="openai")

# Create with fallback
llm = LLMFactory.create_with_fallback(
    providers=["openai", "ollama"],
    configs=[
        {"api_key": "sk-...", "model": "gpt-4"},
        {"model": "llama2"}
    ]
)

# Check if provider is registered
if LLMFactory.is_registered("openai"):
    llm = LLMFactory.create("openai", config)
```

### Embedding Factory

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# List available embedding providers
providers = EmbeddingFactory.list_providers()
print(providers)  # ['openai', 'ollama']

# Create embedding provider
embedder = EmbeddingFactory.create(
    provider="ollama",
    config={"model": "nomic-embed-text"}
)

# Create from environment
embedder = EmbeddingFactory.create_from_env(provider="openai")

# Get provider info
info = EmbeddingFactory.get_provider_info("openai")
print(info)
```

## Error Handling

### Exception Hierarchy

```python
from bruno_llm.exceptions import (
    LLMError,           # Base exception
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthExceededError,
    StreamError,
    TimeoutError,
    ConfigurationError,
    InvalidResponseError,
    ProviderNotFoundError
)

# Example error handling
try:
    response = await provider.generate(messages)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except ModelNotFoundError:
    print("Model not available")
except ContextLengthExceededError:
    print("Input too long")
except LLMError as e:
    print(f"LLM error: {e}")
```

### Connection Checking

```python
# Check if provider is accessible
try:
    is_connected = await provider.check_connection()
    if not is_connected:
        print("Provider not available")
except Exception as e:
    print(f"Connection check failed: {e}")
```

## Advanced Features

### Cost Tracking

```python
from bruno_llm.providers.openai import OpenAIProvider

# Enable cost tracking
provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4",
    track_cost=True
)

# After making requests
cost_info = provider.get_model_info()
print(f"Total cost: ${cost_info.get('total_cost', 0):.4f}")
print(f"Total tokens: {cost_info.get('total_tokens', 0)}")
```

### Context Management

```python
from bruno_llm.base.context import ContextManager
from bruno_core.models import Message, MessageRole

# Initialize context manager
context_mgr = ContextManager(
    max_tokens=4000,
    model="gpt-4"
)

# Add messages and check limits
messages = [
    Message(role=MessageRole.SYSTEM, content="You are helpful."),
    Message(role=MessageRole.USER, content="Long conversation...")
]

# Check if within limits
if context_mgr.check_limit(messages):
    response = await provider.generate(messages)
else:
    # Truncate if needed
    truncated = context_mgr.truncate_messages(messages)
    response = await provider.generate(truncated)
```

### Streaming with Aggregation

```python
from bruno_llm.base.streaming import StreamAggregator, AggregationStrategy

# Word-based aggregation
aggregator = StreamAggregator(
    strategy=AggregationStrategy.WORD,
    buffer_size=5
)

# Stream and aggregate
async for chunk in provider.stream(messages):
    aggregated_chunks = aggregator.add_chunk(chunk)
    for aggregated in aggregated_chunks:
        print(aggregated, end=" ", flush=True)

# Get final chunks
final_chunks = aggregator.finalize()
for chunk in final_chunks:
    print(chunk, end=" ", flush=True)
```

### Rate Limiting

```python
from bruno_llm.base.rate_limiter import RateLimiter

# Create rate limiter (60 requests per minute)
limiter = RateLimiter(requests_per_minute=60)

# Use with provider
async with limiter:
    response = await provider.generate(messages)
```

### Retry Logic

```python
from bruno_llm.base.retry import retry_async, RetryConfig

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=2.0
)

# Apply retry decorator
@retry_async(retry_config)
async def generate_with_retry():
    return await provider.generate(messages)

response = await generate_with_retry()
```

## Best Practices

### 1. Use Factory Pattern

```python
# Preferred - flexible and configurable
from bruno_llm.factory import LLMFactory

llm = LLMFactory.create_from_env("openai")

# Instead of direct instantiation
from bruno_llm.providers.openai import OpenAIProvider
llm = OpenAIProvider(api_key="...")
```

### 2. Handle Errors Gracefully

```python
from bruno_llm.exceptions import RateLimitError
import asyncio

async def safe_generate(provider, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await provider.generate(messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(e.retry_after or 60)

    raise Exception("Max retries exceeded")
```

### 3. Use Context Managers

```python
# Ensure proper cleanup
async with provider:
    response = await provider.generate(messages)
# Provider is automatically cleaned up
```

### 4. Monitor Usage

```python
# Track token usage and costs
info = provider.get_model_info()
if info.get('total_cost', 0) > budget_limit:
    print("Budget exceeded!")
```

### 5. Environment Configuration

```python
# Use environment variables for configuration
import os

# Set in environment or .env file
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'

# Create providers from environment
llm = LLMFactory.create_from_env('openai')
embedder = EmbeddingFactory.create_from_env('ollama')
```

## Integration with Bruno-Core

Bruno-LLM is designed to work seamlessly with the broader Bruno AI ecosystem:

```python
from bruno_core.base import BaseAssistant
from bruno_core.memory import MemoryInterface
from bruno_llm.factory import LLMFactory
from bruno_llm.embedding_factory import EmbeddingFactory

# Create components
llm = LLMFactory.create_from_env('openai')
embedder = EmbeddingFactory.create_from_env('openai')
memory = SomeMemoryImplementation(embedder)

# Create assistant
assistant = BaseAssistant(
    llm=llm,
    memory=memory,
    name="My Assistant"
)

# Use assistant
await assistant.initialize()
response = await assistant.process_message(
    Message(role=MessageRole.USER, content="Hello!")
)
```

## Performance Considerations

### 1. Batch Processing

```python
# Efficient batch embedding
texts = ["text1", "text2", "text3", ...]
embeddings = await embedder.embed_texts(texts)  # More efficient than individual calls
```

### 2. Connection Pooling

```python
# Providers automatically handle connection pooling
# Reuse provider instances when possible
provider = OpenAIProvider(api_key="...")

# Multiple requests use the same connection pool
for messages in conversation_batch:
    await provider.generate(messages)
```

### 3. Streaming for Long Responses

```python
# Use streaming for real-time feedback
async for chunk in provider.stream(messages):
    # Process chunks as they arrive
    process_chunk(chunk)
```

### 4. Context Length Management

```python
# Monitor and manage context length
from bruno_llm.base.context import ContextManager

context_mgr = ContextManager(max_tokens=4000)
if not context_mgr.check_limit(messages):
    messages = context_mgr.truncate_messages(messages)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```python
   # Verify API keys
   export OPENAI_API_KEY="sk-..."

   # Test connection
   is_connected = await provider.check_connection()
   ```

2. **Rate Limiting**
   ```python
   # Implement exponential backoff
   from bruno_llm.base.retry import retry_async

   @retry_async()
   async def make_request():
       return await provider.generate(messages)
   ```

3. **Ollama Connection Issues**
   ```bash
   # Start Ollama service
   ollama serve

   # Pull required model
   ollama pull llama2
   ```

4. **Memory Issues with Large Embeddings**
   ```python
   # Process in smaller batches
   batch_size = embedder.get_max_batch_size()
   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       batch_embeddings = await embedder.embed_texts(batch)
   ```

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Provider-specific debug info
info = provider.get_model_info()
print(f"Provider info: {info}")

# Check provider status
status = await provider.check_connection()
print(f"Connection status: {status}")
```

## API Reference Summary

### LLM Providers
- `OpenAIProvider` - GPT models via OpenAI API
- `OllamaProvider` - Local models via Ollama

### Embedding Providers
- `OpenAIEmbeddingProvider` - OpenAI embedding models
- `OllamaEmbeddingProvider` - Local embedding models

### Factory Classes
- `LLMFactory` - Create and manage LLM providers
- `EmbeddingFactory` - Create and manage embedding providers

### Base Classes
- `BaseProvider` - Common provider functionality
- `BaseEmbeddingProvider` - Common embedding functionality

### Utilities
- `ContextManager` - Token and context length management
- `RateLimiter` - Request rate limiting
- `CostTracker` - Usage and cost tracking
- `StreamAggregator` - Stream processing and aggregation

### Configuration
- `OpenAIConfig` / `OpenAIEmbeddingConfig`
- `OllamaConfig` / `OllamaEmbeddingConfig`

For the latest API updates and examples, visit the [Bruno-LLM repository](https://github.com/meggy-ai/bruno-llm).
