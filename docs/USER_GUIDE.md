# bruno-llm User Guide

Complete guide to using bruno-llm for LLM provider integration.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Provider Setup](#provider-setup)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation

```bash
pip install bruno-llm
```

This includes:
- bruno-core framework
- Ollama provider support
- All base utilities

### With OpenAI Support

```bash
pip install bruno-llm[openai]
```

Additional packages:
- `openai` - Official OpenAI Python client
- `tiktoken` - Accurate token counting for GPT models

### Development Installation

```bash
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm
pip install -e ".[dev]"
```

Includes testing and development tools.

## Quick Start

### Hello World Example

```python
import asyncio
from bruno_llm import LLMFactory
from bruno_core.models import Message, MessageRole

async def main():
    # Create provider
    llm = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Create message
    messages = [
        Message(role=MessageRole.USER, content="Hello! Who are you?")
    ]
    
    # Generate response
    response = await llm.generate(messages)
    print(response)
    
    # Clean up
    await llm.close()

asyncio.run(main())
```

### Streaming Example

```python
async def streaming_demo():
    llm = LLMFactory.create("ollama", {"model": "llama2"})
    
    messages = [
        Message(role=MessageRole.USER, content="Count from 1 to 10")
    ]
    
    print("Response: ", end="", flush=True)
    async for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)
    print()
    
    await llm.close()

asyncio.run(streaming_demo())
```

## Provider Setup

### Ollama

**Installation:**

Visit [ollama.ai](https://ollama.ai/) and follow installation instructions.

**Starting Ollama:**

```bash
ollama serve
```

**Pulling Models:**

```bash
# General purpose
ollama pull llama2
ollama pull mistral

# Code generation
ollama pull codellama

# Larger models
ollama pull llama2:70b
```

**Usage:**

```python
from bruno_llm import LLMFactory

llm = LLMFactory.create("ollama", {
    "base_url": "http://localhost:11434",
    "model": "llama2",
    "timeout": 60.0
})
```

**Environment Variables:**

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2
```

Then:

```python
llm = LLMFactory.create_from_env("ollama")
```

### OpenAI

**Get API Key:**

1. Sign up at [platform.openai.com](https://platform.openai.com/)
2. Navigate to API Keys
3. Create new key

**Usage:**

```python
from bruno_llm import LLMFactory

llm = LLMFactory.create("openai", {
    "api_key": "sk-...",
    "model": "gpt-4",
    "organization": "org-..."  # Optional
})
```

**Environment Variables:**

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4
export OPENAI_ORG_ID=org-...  # Optional
```

Then:

```python
llm = LLMFactory.create_from_env("openai")
```

**Available Models:**

- `gpt-4` - Most capable
- `gpt-4-turbo-preview` - Latest GPT-4 with larger context
- `gpt-3.5-turbo` - Fast and cost-effective

## Basic Usage

### Creating Providers

**Method 1: Factory Pattern (Recommended)**

```python
from bruno_llm import LLMFactory

llm = LLMFactory.create(
    provider="ollama",
    config={"model": "llama2"}
)
```

**Method 2: Direct Instantiation**

```python
from bruno_llm.providers.ollama import OllamaProvider

llm = OllamaProvider(model="llama2")
```

**Method 3: From Environment**

```python
llm = LLMFactory.create_from_env("openai")
```

### Message Format

bruno-llm uses `Message` objects from bruno-core:

```python
from bruno_core.models import Message, MessageRole

# User message
user_msg = Message(
    role=MessageRole.USER,
    content="What is Python?"
)

# System message (sets context/behavior)
system_msg = Message(
    role=MessageRole.SYSTEM,
    content="You are a helpful programming tutor."
)

# Assistant message (previous AI responses)
assistant_msg = Message(
    role=MessageRole.ASSISTANT,
    content="Python is a programming language..."
)

# Conversation
messages = [system_msg, user_msg]
```

### Generating Responses

**Basic Generation:**

```python
response = await llm.generate(messages)
print(response)  # String response
```

**With Parameters:**

```python
response = await llm.generate(
    messages=messages,
    max_tokens=500,           # Limit response length
    temperature=0.7,          # Creativity (0.0 = deterministic, 2.0 = very creative)
    top_p=0.9,               # Nucleus sampling
    stop=["###", "END"]      # Stop sequences
)
```

**Streaming Responses:**

```python
async for chunk in llm.stream(messages, max_tokens=200):
    print(chunk, end="", flush=True)
```

### Provider Methods

**Check Connection:**

```python
if await llm.check_connection():
    print("Provider is accessible")
else:
    print("Cannot connect to provider")
```

**List Available Models:**

```python
models = await llm.list_models()
for model in models:
    print(f"- {model}")
```

**Get Model Information:**

```python
info = llm.get_model_info()
print(f"Provider: {info['provider']}")
print(f"Model: {info['model']}")
print(f"Context Window: {info.get('max_context_tokens', 'Unknown')}")
```

**Token Counting:**

```python
text = "Hello, world!"
tokens = llm.get_token_count(text)
print(f"Token count: {tokens}")
```

**System Prompts:**

```python
# Set system prompt
llm.set_system_prompt("You are a helpful assistant.")

# Get current system prompt
prompt = llm.get_system_prompt()

# System prompt is automatically added to messages
messages = [Message(role=MessageRole.USER, content="Hello")]
response = await llm.generate(messages)  # System prompt included
```

## Advanced Features

### Response Caching

Reduce API costs and latency by caching responses:

```python
from bruno_llm import LLMFactory
from bruno_llm.base import ResponseCache

llm = LLMFactory.create("ollama", {"model": "llama2"})
cache = ResponseCache(
    max_size=100,  # Maximum number of cached responses
    ttl=300        # Time-to-live in seconds (5 minutes)
)

messages = [Message(role=MessageRole.USER, content="What is 2+2?")]

# First request - cache miss
response = await llm.generate(messages, temperature=0.0)
cache.set(messages, response, temperature=0.0)

# Second request - cache hit
cached_response = cache.get(messages, temperature=0.0)
if cached_response:
    print("From cache!")
    response = cached_response
else:
    response = await llm.generate(messages, temperature=0.0)
    cache.set(messages, response, temperature=0.0)

# Cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

**Cache Management:**

```python
# Clear entire cache
cache.clear()

# Invalidate specific entry
cache.invalidate(messages, temperature=0.0)

# Clean up expired entries
expired_count = cache.cleanup_expired()

# Get top entries by access count
top_entries = cache.get_top_entries(n=10)
```

### Context Window Management

Intelligently truncate conversations that exceed token limits:

```python
from bruno_llm.base import (
    ContextWindowManager,
    ContextLimits,
    TruncationStrategy
)

# Create context manager
context_mgr = ContextWindowManager(
    model="gpt-4",  # Uses predefined limits for known models
    limits=ContextLimits(
        max_tokens=8000,
        max_output_tokens=500  # Reserve tokens for response
    ),
    strategy=TruncationStrategy.SMART  # Preserve important messages
)

# Check if messages fit
if context_mgr.check_limit(messages):
    print("Within limit")
else:
    print("Exceeds limit, truncating...")
    messages = context_mgr.truncate(messages)

# Get statistics
stats = context_mgr.get_stats(messages)
print(f"Input tokens: {stats['input_tokens']}")
print(f"Available output tokens: {stats['available_output_tokens']}")
print(f"Usage: {stats['usage_percent']:.1f}%")
```

**Truncation Strategies:**

- `OLDEST_FIRST`: Remove oldest messages first (keeps recent context)
- `MIDDLE_OUT`: Remove middle messages (keeps beginning and end)
- `SLIDING_WINDOW`: Keep most recent N messages
- `SMART`: Preserve system messages and recent important messages

**Custom Limits:**

```python
# For models without predefined limits
context_mgr = ContextWindowManager(
    model="custom-model",
    limits=ContextLimits(max_tokens=4096, max_output_tokens=256)
)
```

### Stream Aggregation

Control how streaming chunks are batched:

```python
from bruno_llm.base import StreamAggregator

# Word-by-word aggregation
aggregator = StreamAggregator(strategy="word")
async for word in aggregator.aggregate(llm.stream(messages)):
    print(f"[{word.strip()}]", end=" ")

# Sentence-by-sentence
aggregator = StreamAggregator(strategy="sentence")
async for sentence in aggregator.aggregate(llm.stream(messages)):
    print(f"\nSentence: {sentence.strip()}")

# Fixed size chunks
aggregator = StreamAggregator(strategy="fixed", chunk_size=10)
async for chunk in aggregator.aggregate(llm.stream(messages)):
    print(chunk, end="")

# Time-based batching (wait up to N seconds)
aggregator = StreamAggregator(strategy="time", interval=0.5)
async for batch in aggregator.aggregate(llm.stream(messages)):
    print(f"Batch: {batch}")

# Passthrough (no aggregation)
aggregator = StreamAggregator(strategy="passthrough")
async for chunk in aggregator.aggregate(llm.stream(messages)):
    print(chunk, end="")
```

### Cost Tracking

Monitor API usage and costs:

```python
from bruno_llm import LLMFactory

llm = LLMFactory.create_from_env("openai")

# Make some requests
await llm.generate(messages)
await llm.generate(messages)

# Get usage report
report = llm.cost_tracker.get_usage_report()
print(f"Total requests: {report['total_requests']}")
print(f"Total tokens: {report['total_tokens']}")
print(f"Total cost: ${report['total_cost']:.4f}")

# Model breakdown
for model, stats in report['model_breakdown'].items():
    print(f"{model}:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Tokens: {stats['input_tokens']} in + {stats['output_tokens']} out")
    print(f"  Cost: ${stats['cost']:.4f}")

# Export to CSV
llm.cost_tracker.export_to_csv("usage_report.csv")

# Export to JSON
llm.cost_tracker.export_to_json("usage_report.json")

# Time range report
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=7)
end = datetime.now()
weekly_report = llm.cost_tracker.get_time_range_report(start, end)

# Budget checking
status = llm.cost_tracker.check_budget(
    budget_limit=10.0,
    warning_threshold=0.8  # Warn at 80%
)

if status['warning']:
    print(f"⚠️ Warning: {status['percent_used']:.1f}% of budget used")

if not status['within_budget']:
    print(f"❌ Budget exceeded! ${status['total_spent']:.2f} / ${status['budget_limit']:.2f}")
```

### Provider Fallback

Try multiple providers in order:

```python
from bruno_llm import LLMFactory

# Try OpenAI, fallback to Ollama
llm = await LLMFactory.create_with_fallback(
    providers=["openai", "ollama"],
    configs=[
        {"api_key": "sk-...", "model": "gpt-4"},
        {"model": "llama2"}
    ]
)

# The first successfully connected provider is used
info = llm.get_model_info()
print(f"Using provider: {info['provider']}")

response = await llm.generate(messages)
```

### Concurrent Requests

Handle multiple requests in parallel:

```python
import asyncio

async def process_multiple():
    llm = LLMFactory.create("ollama", {"model": "llama2"})
    
    # Create multiple message sets
    message_sets = [
        [Message(role=MessageRole.USER, content=f"Tell me about topic {i}")]
        for i in range(5)
    ]
    
    # Process concurrently
    tasks = [llm.generate(msgs) for msgs in message_sets]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Request {i} failed: {response}")
        else:
            print(f"Request {i}: {response[:50]}...")
    
    await llm.close()

asyncio.run(process_multiple())
```

## Best Practices

### 1. Resource Management

Always close providers:

```python
# Using context manager (recommended)
async def with_context_manager():
    llm = LLMFactory.create("ollama", {"model": "llama2"})
    async with llm:
        response = await llm.generate(messages)
    # Automatically closed

# Manual cleanup
try:
    llm = LLMFactory.create("ollama", {"model": "llama2"})
    response = await llm.generate(messages)
finally:
    await llm.close()
```

### 2. Error Handling

Handle provider-specific errors:

```python
from bruno_llm.exceptions import (
    ModelNotFoundError,
    RateLimitError,
    AuthenticationError,
    ContextLengthExceededError
)

try:
    response = await llm.generate(messages)
except ModelNotFoundError as e:
    print(f"Model not available: {e}")
    # Try alternative model
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Wait and retry
    await asyncio.sleep(60)
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Check API key
except ContextLengthExceededError as e:
    print(f"Context too long: {e}")
    # Truncate messages
    messages = context_mgr.truncate(messages)
    response = await llm.generate(messages)
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Configuration Management

Use environment variables for sensitive data:

```python
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Use environment variables
llm = LLMFactory.create("openai", {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": os.getenv("OPENAI_MODEL", "gpt-4")
})
```

**.env file:**
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Temperature Settings

Choose appropriate temperature based on use case:

```python
# Deterministic (temperature=0.0) - for factual, consistent responses
response = await llm.generate(messages, temperature=0.0)

# Balanced (temperature=0.7) - good default
response = await llm.generate(messages, temperature=0.7)

# Creative (temperature=1.5) - for brainstorming, creative writing
response = await llm.generate(messages, temperature=1.5)
```

### 5. Token Management

Monitor token usage:

```python
# Check message token count before sending
total_tokens = sum(llm.get_token_count(msg.content) for msg in messages)
print(f"Request will use approximately {total_tokens} tokens")

# Limit response length
response = await llm.generate(messages, max_tokens=500)
```

### 6. Caching Strategy

Use caching for repeated queries:

```python
# Cache deterministic responses (temperature=0)
cache = ResponseCache(max_size=1000, ttl=3600)  # 1 hour TTL

# Check cache first
cached = cache.get(messages, temperature=0.0)
if cached:
    response = cached
else:
    response = await llm.generate(messages, temperature=0.0)
    cache.set(messages, response, temperature=0.0)
```

Don't cache:
- Creative/random responses (temperature > 0)
- Time-sensitive information
- User-specific data

## Troubleshooting

### Ollama Issues

**"Cannot connect to Ollama"**

```python
# Check if Ollama is running
import httpx

try:
    response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    print(f"Ollama is running. Models: {response.json()}")
except Exception as e:
    print(f"Ollama not accessible: {e}")
    print("Start Ollama with: ollama serve")
```

**"Model not found"**

```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama2
```

**Slow responses**

- Use smaller models (llama2:7b instead of llama2:70b)
- Reduce max_tokens
- Use GPU if available

### OpenAI Issues

**"Authentication failed"**

```python
# Verify API key
import openai

try:
    openai.api_key = "sk-..."
    models = openai.Model.list()
    print("API key is valid")
except openai.error.AuthenticationError:
    print("Invalid API key")
```

**"Rate limit exceeded"**

```python
from bruno_llm.base import RateLimiter

# Add rate limiting
limiter = RateLimiter(requests_per_minute=50)

async def rate_limited_request():
    async with limiter:
        return await llm.generate(messages)
```

**High costs**

- Use gpt-3.5-turbo instead of gpt-4
- Reduce max_tokens
- Implement caching
- Monitor with cost_tracker

### General Issues

**Import errors**

```bash
# Reinstall package
pip uninstall bruno-llm
pip install bruno-llm

# Or install from source
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm
pip install -e .
```

**Timeout errors**

```python
# Increase timeout
llm = LLMFactory.create("ollama", {
    "model": "llama2",
    "timeout": 120.0  # 2 minutes
})
```

**Memory issues**

- Process in batches
- Clear cache periodically
- Use streaming for large responses

## Next Steps

- Read [examples/](../examples/) for more complete examples
- Check [API Reference](README.md#api-reference) for detailed method documentation
- See [TESTING.md](../TESTING.md) for testing guide
- Join our [community](https://github.com/meggy-ai/bruno-llm/discussions) for support

---

Need help? [Open an issue](https://github.com/meggy-ai/bruno-llm/issues) or email contact@meggy.ai
