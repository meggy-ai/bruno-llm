# OpenAI Provider Guide

The OpenAI provider enables access to GPT models and embedding services through OpenAI's API, providing state-of-the-art language models and high-quality embeddings.

## Overview

- **Latest Models**: Access to GPT-4, GPT-3.5, and latest embedding models
- **High Quality**: Industry-leading model performance
- **Scalable**: Handle high-volume production workloads
- **Rich Features**: Function calling, fine-tuning, advanced parameters

## Quick Setup

### 1. Get API Key

1. Sign up at [platform.openai.com](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Note your Organization ID (optional)

### 2. Install Dependencies

```bash
pip install bruno-llm[openai]
```

### 3. Configure Environment

```bash
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_MODEL=gpt-4
export OPENAI_ORG_ID=org-your-org-id  # Optional
```

## LLM Usage

### Basic Configuration

```python
from bruno_llm.providers.openai import OpenAIProvider

# Create provider
llm = OpenAIProvider(
    api_key="sk-your-key-here",
    model="gpt-4"
)
```

### Factory Pattern

```python
from bruno_llm.factory import LLMFactory

# Direct creation
llm = LLMFactory.create("openai", {
    "api_key": "sk-your-key-here",
    "model": "gpt-3.5-turbo",
    "organization": "org-your-org-id"
})

# From environment
llm = LLMFactory.create_from_env("openai")
```

### Available LLM Models

| Model | Context | Best For | Cost |
|-------|---------|----------|------|
| `gpt-4` | 8K | Complex reasoning | $$$ |
| `gpt-4-turbo-preview` | 128K | Large context | $$ |
| `gpt-3.5-turbo` | 4K | Fast, general use | $ |
| `gpt-3.5-turbo-16k` | 16K | Longer context | $$ |

## Embedding Usage

### Basic Configuration

```python
from bruno_llm.providers.openai import OpenAIEmbeddingProvider

# Create embedding provider
embedder = OpenAIEmbeddingProvider(
    api_key="sk-your-key-here",
    model="text-embedding-3-small"
)
```

### Factory Pattern

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# Direct creation
embedder = EmbeddingFactory.create("openai", {
    "api_key": "sk-your-key-here",
    "model": "text-embedding-3-large"
})

# From environment
embedder = EmbeddingFactory.create_from_env("openai")
```

### Available Embedding Models

| Model | Dimensions | Max Input | Cost/1M tokens | Best For |
|-------|------------|-----------|----------------|----------|
| `text-embedding-3-small` | 1536 | 8191 | $0.02 | Cost-effective |
| `text-embedding-3-large` | 3072 | 8191 | $0.13 | High performance |
| `text-embedding-ada-002` | 1536 | 8191 | $0.10 | Legacy, stable |

## Advanced Configuration

### Custom Parameters

```python
from bruno_llm.providers.openai import OpenAIConfig

config = OpenAIConfig(
    api_key="sk-your-key-here",
    model="gpt-4",
    temperature=0.8,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    timeout=30.0
)

llm = OpenAIProvider(config=config)
```

### Cost Optimization

```python
# Use cheaper models for simple tasks
cheap_llm = OpenAIProvider(
    api_key="sk-your-key-here",
    model="gpt-3.5-turbo"  # Much cheaper than GPT-4
)

# Use efficient embedding model
cheap_embedder = OpenAIEmbeddingProvider(
    api_key="sk-your-key-here",
    model="text-embedding-3-small"  # 5x cheaper than ada-002
)
```

### Function Calling

```python
functions = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        }
    }
}]

response = await llm.generate(
    messages=messages,
    functions=functions,
    function_call="auto"
)
```

## Environment Configuration

### Required Variables

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Optional Variables

```bash
# Organization (for team accounts)
export OPENAI_ORG_ID=org-your-org-id

# Model selection
export OPENAI_MODEL=gpt-4
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Performance tuning
export OPENAI_TIMEOUT=30.0
export OPENAI_MAX_RETRIES=3

# Cost control
export OPENAI_MAX_TOKENS=1000
export OPENAI_TEMPERATURE=0.7
```

## Production Considerations

### Rate Limiting

```python
from bruno_llm.base import RateLimiter

# Respect OpenAI rate limits
limiter = RateLimiter(requests_per_minute=3500)  # Adjust based on your tier

async def safe_generate(messages):
    async with limiter:
        return await llm.generate(messages)
```

### Cost Monitoring

```python
from bruno_llm.base import CostTracker

# Track costs per request
cost_tracker = CostTracker(
    provider_name="openai",
    pricing={
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
    }
)

# Costs are automatically tracked
response = await llm.generate(messages)
daily_cost = cost_tracker.get_daily_cost()
```

### Error Handling

```python
from bruno_llm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError
)

try:
    response = await llm.generate(messages)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit hit, backing off...")
    await asyncio.sleep(60)
except ContextLengthExceededError:
    print("Message too long, truncating...")
```

## Complete Examples

See [OpenAI Provider API Documentation](../../api/OPENAI_PROVIDER.md) for comprehensive examples including:

- Model configurations and features
- Advanced parameters and fine-tuning
- Cost optimization strategies
- Performance monitoring
- Integration patterns
- Best practices

## Troubleshooting

### Common Issues

**Authentication errors:**
```python
# Verify API key
import openai
openai.api_key = "sk-your-key-here"
try:
    models = openai.Model.list()
    print("API key is valid")
except openai.error.AuthenticationError:
    print("Invalid API key")
```

**Rate limit exceeded:**
```python
# Add retry logic with backoff
import asyncio
from bruno_llm.exceptions import RateLimitError

async def retry_on_rate_limit(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise
```

**High costs:**
- Use `gpt-3.5-turbo` instead of `gpt-4` for simple tasks
- Set reasonable `max_tokens` limits
- Implement response caching
- Use `text-embedding-3-small` for embeddings

For detailed troubleshooting, see the [main troubleshooting guide](../../USER_GUIDE.md#troubleshooting).
