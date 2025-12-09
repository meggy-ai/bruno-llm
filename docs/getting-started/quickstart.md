# Quick Start

## Basic Usage

### 1. Create a Provider

```python
from bruno_llm import LLMFactory

# Ollama (local)
llm = LLMFactory.create("ollama", {
    "model": "llama2",
    "base_url": "http://localhost:11434"
})

# OpenAI (cloud)
llm = LLMFactory.create("openai", {
    "api_key": "sk-...",
    "model": "gpt-4"
})
```

### 2. Generate Responses

```python
from bruno_core.models import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="What is Python?")
]

# Generate complete response
response = await llm.generate(messages)
print(response)
```

### 3. Stream Responses

```python
# Stream response tokens
async for chunk in llm.stream(messages):
    print(chunk, end="", flush=True)
```

## Complete Example

```python
import asyncio
from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory

async def main():
    # Create provider
    llm = LLMFactory.create("ollama", {"model": "llama2"})

    # Prepare messages
    messages = [
        Message(role=MessageRole.USER, content="Tell me a joke")
    ]

    # Generate response
    response = await llm.generate(messages)
    print(f"Response: {response}")

    # Get token count
    tokens = llm.get_token_count(response)
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- [User Guide](../user-guide/overview.md) - Complete documentation
- [Ollama Provider](../user-guide/providers/ollama.md) - Local LLM setup
- [OpenAI Provider](../user-guide/providers/openai.md) - Cloud LLM setup
- [Advanced Features](../user-guide/advanced/caching.md) - Caching, streaming, etc.
