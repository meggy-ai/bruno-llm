# OpenAI Provider Documentation

## Overview

The OpenAI provider offers seamless integration with OpenAI's GPT models and embedding services through the official OpenAI Python SDK. It supports all major GPT models and provides advanced features like cost tracking, streaming, and comprehensive error handling.

## Installation Requirements

```bash
pip install bruno-llm[openai]
# or
pip install openai
```

## Quick Start

### Basic LLM Usage

```python
from bruno_llm.providers.openai import OpenAIProvider
from bruno_core.models import Message, MessageRole

# Initialize the provider
provider = OpenAIProvider(
    api_key="sk-your-openai-key",
    model="gpt-4"
)

# Create a conversation
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
    Message(role=MessageRole.USER, content="Explain quantum computing in simple terms.")
]

# Generate response
response = await provider.generate(messages, temperature=0.7, max_tokens=500)
print(response)

# Stream response for real-time output
print("Streaming response:")
async for chunk in provider.stream(messages, temperature=0.7):
    print(chunk, end="", flush=True)
```

### Basic Embedding Usage

```python
from bruno_llm.providers.openai import OpenAIEmbeddingProvider

# Initialize embedding provider
embedder = OpenAIEmbeddingProvider(
    api_key="sk-your-openai-key",
    model="text-embedding-ada-002"
)

# Single text embedding
text = "The quick brown fox jumps over the lazy dog"
embedding = await embedder.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
texts = [
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "AI will transform many industries"
]
embeddings = await embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")

# Calculate similarity
similarity = embedder.calculate_similarity(embeddings[0], embeddings[1])
print(f"Similarity between first two texts: {similarity:.4f}")
```

## Configuration

### LLM Configuration

```python
from bruno_llm.providers.openai import OpenAIConfig, OpenAIProvider

# Detailed configuration
config = OpenAIConfig(
    api_key="sk-your-openai-key",
    model="gpt-4",
    organization="org-your-org-id",  # Optional
    base_url="https://api.openai.com/v1",  # Custom endpoint if needed
    timeout=30.0,
    max_retries=3,
    track_cost=True,
    default_temperature=0.7,
    default_max_tokens=1000
)

provider = OpenAIProvider(config=config)
```

### Embedding Configuration

```python
from bruno_llm.providers.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingProvider

# Embedding configuration
config = OpenAIEmbeddingConfig(
    api_key="sk-your-openai-key",
    model="text-embedding-3-large",
    dimensions=1024,  # For v3 models, can reduce dimensions
    batch_size=100,   # Process up to 100 texts at once
    timeout=30.0,
    organization="org-your-org-id"
)

embedder = OpenAIEmbeddingProvider(config=config)
```

## Environment Variables

Set up your environment for seamless configuration:

```bash
# Required
export OPENAI_API_KEY="sk-your-openai-key"

# Optional LLM settings
export OPENAI_MODEL="gpt-4"
export OPENAI_ORG_ID="org-your-org-id"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_TIMEOUT="30.0"
export OPENAI_MAX_RETRIES="3"

# Optional embedding settings
export OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"
export OPENAI_EMBEDDING_DIMENSIONS="1536"
export OPENAI_EMBEDDING_BATCH_SIZE="100"
```

Create providers from environment:

```python
from bruno_llm.factory import LLMFactory
from bruno_llm.embedding_factory import EmbeddingFactory

# Auto-configure from environment
llm = LLMFactory.create_from_env("openai")
embedder = EmbeddingFactory.create_from_env("openai")
```

## Supported Models

### GPT Models (LLM)

| Model | Description | Context Length | Cost per 1K tokens |
|-------|-------------|----------------|-------------------|
| `gpt-4` | Most capable model | 8,192 tokens | $0.03 / $0.06 |
| `gpt-4-turbo` | Latest GPT-4 with 128K context | 128,000 tokens | $0.01 / $0.03 |
| `gpt-4o` | Optimized for speed and efficiency | 128,000 tokens | $0.005 / $0.015 |
| `gpt-4o-mini` | Fast and cost-effective | 128,000 tokens | $0.00015 / $0.0006 |
| `gpt-3.5-turbo` | Fast and efficient | 16,385 tokens | $0.0015 / $0.002 |

### Embedding Models

| Model | Dimensions | Max Input | Cost per 1M tokens |
|-------|------------|-----------|-------------------|
| `text-embedding-ada-002` | 1,536 | 8,191 tokens | $0.10 |
| `text-embedding-3-small` | 1,536 (default) | 8,191 tokens | $0.02 |
| `text-embedding-3-large` | 3,072 (default) | 8,191 tokens | $0.13 |

Note: For `text-embedding-3-*` models, you can specify custom dimensions (minimum 1).

## Advanced Features

### Cost Tracking

```python
from bruno_llm.providers.openai import OpenAIProvider

# Enable cost tracking
provider = OpenAIProvider(
    api_key="sk-your-key",
    model="gpt-4",
    track_cost=True
)

# After making requests
await provider.generate(messages)

# Get cost information
model_info = provider.get_model_info()
print(f"Total cost: ${model_info['total_cost']:.4f}")
print(f"Total tokens: {model_info['total_tokens']}")
print(f"Request count: {model_info['request_count']}")

# Cost breakdown by request
cost_tracker = provider._cost_tracker
for record in cost_tracker.get_usage_history():
    print(f"Request: {record.input_tokens} + {record.output_tokens} tokens = ${record.total_cost:.4f}")
```

### Streaming with Real-time Processing

```python
import asyncio

async def process_stream_with_buffer():
    messages = [Message(role=MessageRole.USER, content="Write a short story")]

    buffer = ""
    word_count = 0

    async for chunk in provider.stream(messages, temperature=0.8):
        buffer += chunk

        # Process complete words
        if " " in chunk or "." in chunk:
            words = buffer.split()
            if len(words) > 1:
                # Process all complete words except the last (might be incomplete)
                complete_words = words[:-1]
                word_count += len(complete_words)
                print(f"Words processed: {word_count}", end="\r")

                # Keep the last potentially incomplete word
                buffer = words[-1] if words else ""

    # Process any remaining content
    if buffer:
        word_count += len(buffer.split())

    print(f"\nTotal words: {word_count}")
```

### Custom Headers and Parameters

```python
# Custom API parameters
response = await provider.generate(
    messages=messages,
    temperature=0.9,
    max_tokens=2000,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.3,
    stop=["\n\n", "###"]
)

# Function calling (for compatible models)
functions = [
    {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

response = await provider.generate(
    messages=messages,
    functions=functions,
    function_call="auto"
)
```

### Batch Processing for Embeddings

```python
async def process_large_dataset(texts):
    """Process a large dataset of texts efficiently."""
    embedder = OpenAIEmbeddingProvider(
        api_key="sk-your-key",
        batch_size=100  # Process 100 texts at once
    )

    all_embeddings = []

    # Process in batches to respect rate limits
    for i in range(0, len(texts), embedder.batch_size):
        batch = texts[i:i + embedder.batch_size]

        try:
            batch_embeddings = await embedder.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

            print(f"Processed {len(all_embeddings)}/{len(texts)} texts")

            # Rate limiting - small delay between batches
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Continue with next batch or implement retry logic

    return all_embeddings

# Usage
texts = ["text 1", "text 2", ...]  # Your large dataset
embeddings = await process_large_dataset(texts)
```

## Error Handling

### Comprehensive Error Handling

```python
from bruno_llm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthExceededError,
    LLMError
)
import asyncio

async def robust_generate(provider, messages, max_retries=3):
    """Generate response with comprehensive error handling."""

    for attempt in range(max_retries):
        try:
            return await provider.generate(messages)

        except AuthenticationError:
            print("❌ Invalid API key. Please check your OpenAI API key.")
            raise  # Don't retry auth errors

        except RateLimitError as e:
            print(f"⏳ Rate limit hit. Waiting {e.retry_after or 60} seconds...")
            await asyncio.sleep(e.retry_after or 60)

        except ModelNotFoundError:
            print("❌ Model not found. Check if model name is correct.")
            raise  # Don't retry model errors

        except ContextLengthExceededError:
            print("⚠️  Context too long. Try reducing message length or use a model with larger context.")
            # Could implement automatic truncation here
            raise

        except LLLError as e:
            print(f"🔄 LLM error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    raise Exception("Max retries exceeded")

# Usage
try:
    response = await robust_generate(provider, messages)
    print(f"✅ Success: {response}")
except Exception as e:
    print(f"❌ Failed: {e}")
```

### Connection and Model Validation

```python
async def validate_setup(provider):
    """Validate OpenAI provider setup."""

    print("🔍 Validating OpenAI setup...")

    # Check connection
    try:
        is_connected = await provider.check_connection()
        if is_connected:
            print("✅ Connection successful")
        else:
            print("❌ Connection failed")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    # List available models
    try:
        models = await provider.list_models()
        print(f"✅ Available models: {len(models)} found")

        # Check if current model is available
        current_model = provider.get_model_info()["model"]
        if current_model in models:
            print(f"✅ Current model '{current_model}' is available")
        else:
            print(f"⚠️  Current model '{current_model}' not found in available models")

    except Exception as e:
        print(f"❌ Could not list models: {e}")

    # Test generation with a simple request
    try:
        test_messages = [Message(role=MessageRole.USER, content="Say 'test'")]
        response = await provider.generate(test_messages, max_tokens=10)
        print(f"✅ Test generation successful: '{response.strip()}'")
        return True
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        return False

# Validate before using
if await validate_setup(provider):
    # Proceed with actual usage
    pass
```

## Performance Optimization

### Connection Pooling and Reuse

```python
# Reuse provider instances for better performance
class OpenAIService:
    def __init__(self, api_key: str):
        self.provider = OpenAIProvider(api_key=api_key, track_cost=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_response(self, messages, **kwargs):
        return await self.provider.generate(messages, **kwargs)

# Usage with context manager
async with OpenAIService("sk-your-key") as service:
    # Multiple requests reuse the same connection
    for conversation in conversations:
        response = await service.generate_response(conversation)
        process_response(response)
```

### Efficient Token Management

```python
async def smart_token_management(provider, long_conversation):
    """Manage long conversations by tracking tokens."""

    # Get token count for messages
    total_tokens = sum(provider.get_token_count(msg.content) for msg in long_conversation)

    model_info = provider.get_model_info()
    max_tokens = model_info.get("max_context_tokens", 4096)

    if total_tokens > max_tokens * 0.8:  # Use 80% of context as safety margin
        print(f"⚠️  Conversation too long ({total_tokens} tokens). Truncating...")

        # Keep system message and recent messages
        system_messages = [msg for msg in long_conversation if msg.role == MessageRole.SYSTEM]
        user_assistant_messages = [msg for msg in long_conversation if msg.role != MessageRole.SYSTEM]

        # Take last N messages that fit in context
        truncated_messages = system_messages
        current_tokens = sum(provider.get_token_count(msg.content) for msg in system_messages)

        for msg in reversed(user_assistant_messages):
            msg_tokens = provider.get_token_count(msg.content)
            if current_tokens + msg_tokens < max_tokens * 0.8:
                truncated_messages.insert(-len(system_messages) if system_messages else 0, msg)
                current_tokens += msg_tokens
            else:
                break

        long_conversation = truncated_messages

    return await provider.generate(long_conversation)
```

## Integration Examples

### With AsyncIO and Concurrent Requests

```python
import asyncio
import time

async def concurrent_generations():
    """Process multiple requests concurrently."""

    provider = OpenAIProvider(api_key="sk-your-key", model="gpt-3.5-turbo")

    # Create different requests
    requests = [
        [Message(role=MessageRole.USER, content="Explain AI")],
        [Message(role=MessageRole.USER, content="What is Python?")],
        [Message(role=MessageRole.USER, content="How does the internet work?")],
    ]

    start_time = time.time()

    # Process concurrently (respects rate limits automatically)
    responses = await asyncio.gather(*[
        provider.generate(messages, max_tokens=200)
        for messages in requests
    ])

    end_time = time.time()

    print(f"Processed {len(responses)} requests in {end_time - start_time:.2f} seconds")

    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response[:100]}...")

asyncio.run(concurrent_generations())
```

### With FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio

app = FastAPI()

# Global provider instance (reuse connection pool)
provider = None

@app.on_event("startup")
async def startup():
    global provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )

@app.on_event("shutdown")
async def shutdown():
    if provider:
        await provider.__aexit__(None, None, None)

class ChatRequest(BaseModel):
    messages: List[dict]
    temperature: float = 0.7
    max_tokens: int = 1000

class ChatResponse(BaseModel):
    response: str
    usage: dict

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert dict messages to Message objects
        messages = [
            Message(role=MessageRole(msg["role"]), content=msg["content"])
            for msg in request.messages
        ]

        response = await provider.generate(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        usage_info = provider.get_model_info()

        return ChatResponse(
            response=response,
            usage={
                "total_tokens": usage_info.get("total_tokens", 0),
                "total_cost": usage_info.get("total_cost", 0)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
```

## Best Practices

### 1. API Key Security

```python
import os
from dotenv import load_dotenv

# Use environment variables (never hardcode keys)
load_dotenv()  # Load from .env file

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),  # ✅ Good
    # api_key="sk-hardcoded-key",  # ❌ Never do this
)

# For production, use secrets management
# provider = OpenAIProvider(api_key=get_secret("openai-api-key"))
```

### 2. Cost Management

```python
# Set up cost alerts and tracking
class CostAwareProvider:
    def __init__(self, api_key: str, daily_budget: float = 10.0):
        self.provider = OpenAIProvider(api_key=api_key, track_cost=True)
        self.daily_budget = daily_budget

    async def generate_with_budget_check(self, messages, **kwargs):
        # Check current spending
        model_info = self.provider.get_model_info()
        current_cost = model_info.get("total_cost", 0)

        if current_cost >= self.daily_budget:
            raise Exception(f"Daily budget of ${self.daily_budget} exceeded (spent: ${current_cost:.2f})")

        return await self.provider.generate(messages, **kwargs)
```

### 3. Response Quality Optimization

```python
# Template for consistent, high-quality responses
async def generate_structured_response(provider, user_query: str, response_format: str = "helpful"):
    """Generate structured, high-quality responses."""

    format_templates = {
        "helpful": "Provide a helpful, accurate, and detailed response.",
        "concise": "Provide a concise, direct answer in 1-2 sentences.",
        "educational": "Explain the concept step-by-step as if teaching a beginner.",
        "creative": "Provide a creative, engaging, and original response."
    }

    system_prompt = f"""You are a helpful AI assistant. {format_templates.get(response_format, format_templates['helpful'])}

Guidelines:
- Be accurate and factual
- Cite sources when relevant
- Ask for clarification if the question is ambiguous
- Admit when you don't know something"""

    messages = [
        Message(role=MessageRole.SYSTEM, content=system_prompt),
        Message(role=MessageRole.USER, content=user_query)
    ]

    return await provider.generate(
        messages,
        temperature=0.7,  # Balanced creativity and accuracy
        max_tokens=1000
    )

# Usage
response = await generate_structured_response(
    provider,
    "Explain machine learning",
    response_format="educational"
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Rate Limit Errors**
   ```python
   # Implement exponential backoff
   from bruno_llm.base.retry import retry_async, RetryConfig

   retry_config = RetryConfig(max_attempts=5, base_delay=1.0)

   @retry_async(retry_config)
   async def generate_with_retry():
       return await provider.generate(messages)
   ```

2. **Context Length Issues**
   ```python
   # Automatic message truncation
   def truncate_conversation(messages, max_tokens=3000):
       # Keep system messages and recent context
       system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
       other_msgs = [m for m in messages if m.role != MessageRole.SYSTEM]

       # Take last N messages that fit
       total_tokens = 0
       kept_msgs = []

       for msg in reversed(other_msgs):
           msg_tokens = len(msg.content.split()) * 1.3  # Rough estimate
           if total_tokens + msg_tokens < max_tokens:
               kept_msgs.insert(0, msg)
               total_tokens += msg_tokens
           else:
               break

       return system_msgs + kept_msgs
   ```

3. **Network Connectivity Issues**
   ```python
   # Test and validate connection
   async def test_connection():
       try:
           await asyncio.wait_for(provider.check_connection(), timeout=10.0)
           print("✅ Connection OK")
       except asyncio.TimeoutError:
           print("❌ Connection timeout - check internet/proxy settings")
       except Exception as e:
           print(f"❌ Connection error: {e}")
   ```

### Debug Mode

```python
import logging

# Enable debug logging for detailed information
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("bruno_llm").setLevel(logging.DEBUG)

# This will show all HTTP requests and responses
```

For more examples and advanced usage patterns, see the [Bruno-LLM Examples](../examples/) directory.
