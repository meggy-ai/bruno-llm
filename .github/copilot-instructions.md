# Copilot Instructions for bruno-llm

## Project Overview

**bruno-llm** provides LLM provider implementations that integrate with the `bruno-core` framework. This package enables swappable language model backends (Ollama, OpenAI, Claude, etc.) through a unified interface.

**Key Context:**
- Extension of bruno-core: https://github.com/meggy-ai/bruno-core
- Implements `LLMInterface` from bruno-core
- Part of the Meggy-AI ecosystem
- Python 3.9+ with async-first design
- Currently implementing Ollama and OpenAI providers first

## Architecture Principles

### 1. Interface-Driven Design
All providers **must** implement `bruno_core.interfaces.LLMInterface`:

```python
from bruno_core.interfaces import LLMInterface
from bruno_core.models import Message
from typing import AsyncIterator, List, Dict, Any, Optional

class CustomProvider(LLMInterface):
    async def generate(self, messages: List[Message], **kwargs) -> str:
        """Generate complete response"""
        
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream response tokens"""
        
    def get_token_count(self, text: str) -> int:
        """Estimate token count"""
        
    async def check_connection(self) -> bool:
        """Verify provider accessibility"""
        
    async def list_models(self) -> List[str]:
        """List available models"""
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt"""
        
    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt"""
```

### 2. Async-First Everything
- All I/O operations MUST be async
- Use `aiohttp` or `httpx` for HTTP clients
- Use `asyncio` primitives for concurrency
- Never block the event loop

### 3. Provider Structure Pattern
Each provider follows this structure:

```
bruno_llm/providers/{provider_name}/
├── __init__.py          # Public exports
├── provider.py          # LLMInterface implementation
├── client.py            # HTTP/API client (if needed)
├── config.py            # Pydantic configuration model
└── exceptions.py        # Provider-specific errors (optional)
```

## Critical Implementation Details

### Message Format Conversion
bruno-core uses `Message` objects. Convert to provider-specific format:

```python
from bruno_core.models import Message, MessageRole

# OpenAI format
openai_messages = [
    {"role": msg.role.value, "content": msg.content}
    for msg in messages
]

# Claude format (system message separate)
system = None
claude_messages = []
for msg in messages:
    if msg.role == MessageRole.SYSTEM:
        system = msg.content
    else:
        claude_messages.append({
            "role": msg.role.value,
            "content": msg.content
        })
```

### Error Handling Strategy
Use the exception hierarchy in `bruno_llm/exceptions.py`:

```python
from bruno_llm.exceptions import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLengthExceededError,
    StreamError,
)

# In provider code
try:
    response = await self._make_request(messages)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        raise AuthenticationError("Invalid API key") from e
    elif e.response.status_code == 429:
        raise RateLimitError("Rate limit exceeded") from e
    elif e.response.status_code == 404:
        raise ModelNotFoundError(f"Model {self.model} not found") from e
```

### Streaming Implementation
Handle streaming consistently across providers:

```python
async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
    """Stream response tokens."""
    try:
        async with self.client.stream(
            "POST", 
            f"{self.base_url}/chat",
            json=self._format_request(messages, stream=True)
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = self._parse_chunk(line)
                    if chunk:
                        yield chunk
    except Exception as e:
        raise StreamError(f"Streaming failed: {e}") from e
```

### Configuration Management
Use Pydantic v2 models for all configuration:

```python
from pydantic import BaseModel, Field, SecretStr
from typing import Optional

class OllamaConfig(BaseModel):
    """Configuration for Ollama provider."""
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    model: str = Field(
        default="llama2",
        description="Model name to use"
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    
    class Config:
        frozen = True  # Immutable after creation
```

## Testing Requirements

### Test Structure
Each provider MUST have comprehensive tests:

```
tests/
├── providers/
│   ├── test_ollama.py       # Ollama provider tests
│   ├── test_openai.py       # OpenAI provider tests
│   └── conftest.py          # Shared fixtures
├── test_factory.py          # Factory pattern tests
├── test_base.py             # Base utilities tests
└── conftest.py              # Root fixtures
```

### Mock Provider Responses
Use `pytest-asyncio` and mock responses:

```python
import pytest
from unittest.mock import AsyncMock, patch
from bruno_llm.providers.ollama import OllamaProvider

@pytest.mark.asyncio
async def test_generate():
    provider = OllamaProvider()
    
    with patch.object(provider, '_make_request', new_callable=AsyncMock) as mock:
        mock.return_value = {"message": {"content": "test response"}}
        
        messages = [Message(role=MessageRole.USER, content="hello")]
        response = await provider.generate(messages)
        
        assert response == "test response"
        mock.assert_called_once()
```

### Integration Tests (Optional)
Real API tests only if environment configured:

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
async def test_openai_real():
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    # Test with real API
```

## Common Patterns & Conventions

### 1. Base Provider Pattern
Extend `BaseProvider` for common functionality:

```python
from bruno_llm.base import BaseProvider

class NewProvider(BaseProvider):
    """New LLM provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._setup_client()
    
    async def generate(self, messages, **kwargs):
        # Use inherited retry logic
        return await self._with_retry(
            self._generate_impl(messages, **kwargs)
        )
```

### 2. Cost Tracking Integration
Use `CostTracker` for all providers:

```python
from bruno_llm.base import CostTracker

class Provider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.cost_tracker = CostTracker(
            provider_name="openai",
            pricing={
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            }
        )
    
    async def generate(self, messages, **kwargs):
        response = await self._make_request(messages)
        
        # Track costs
        self.cost_tracker.track_request(
            model=self.model,
            input_tokens=self.get_token_count(messages),
            output_tokens=self.get_token_count(response),
        )
        
        return response
```

### 3. Rate Limiting Pattern
Apply rate limiting to all providers:

```python
from bruno_llm.base import RateLimiter

class Provider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit
        )
    
    async def generate(self, messages, **kwargs):
        async with self.rate_limiter:
            return await self._make_request(messages)
```

## Key Files & Their Purpose

- `bruno_llm/__init__.py` - Public API exports, version info
- `bruno_llm/factory.py` - Factory for creating provider instances
- `bruno_llm/base/base_provider.py` - Common provider functionality
- `bruno_llm/base/token_counter.py` - Token counting utilities
- `bruno_llm/base/rate_limiter.py` - Rate limiting implementation
- `bruno_llm/base/retry.py` - Retry logic with exponential backoff
- `bruno_llm/base/cost_tracker.py` - Cost tracking and reporting
- `bruno_llm/exceptions.py` - Exception hierarchy
- `bruno_llm/providers/` - Individual provider implementations

## Development Workflow

### 1. Adding a New Provider
```bash
# 1. Create provider directory
mkdir -p bruno_llm/providers/{provider_name}

# 2. Implement required files
touch bruno_llm/providers/{provider_name}/__init__.py
touch bruno_llm/providers/{provider_name}/provider.py
touch bruno_llm/providers/{provider_name}/config.py

# 3. Write tests
touch tests/providers/test_{provider_name}.py

# 4. Update factory registration
# Edit bruno_llm/factory.py

# 5. Run tests
pytest tests/providers/test_{provider_name}.py -v

# 6. Update documentation
# Add provider guide to docs/
```

### 2. Code Quality Checks
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy bruno_llm

# Run all tests
pytest tests/ -v --cov=bruno_llm

# Check coverage
pytest tests/ --cov=bruno_llm --cov-report=html
```

### 3. Running Specific Tests
```bash
# Test one provider
pytest tests/providers/test_ollama.py -v

# Test with real API (requires env vars)
pytest tests/ -v -m integration

# Test streaming specifically
pytest tests/ -v -k stream
```

## Environment Variables

Providers use these environment variables for configuration:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_ORG_ID=org-...

# General
BRUNO_LLM_LOG_LEVEL=INFO
BRUNO_LLM_TIMEOUT=30.0
```

## Integration with bruno-core

### Usage Example
```python
from bruno_core.base import BaseAssistant
from bruno_core.models import Message, MessageRole
from bruno_llm.providers.ollama import OllamaProvider
from bruno_llm.providers.openai import OpenAIProvider

# Create provider
llm = OllamaProvider(base_url="http://localhost:11434", model="llama2")
# OR
llm = OpenAIProvider(api_key="sk-...", model="gpt-4")

# Use with bruno-core
assistant = BaseAssistant(llm=llm, memory=my_memory)
await assistant.initialize()

message = Message(role=MessageRole.USER, content="Hello!")
response = await assistant.process_message(message)
```

### Factory Pattern Usage
```python
from bruno_llm.factory import LLMFactory

# Create from config dict
llm = LLMFactory.create(
    provider="ollama",
    config={"base_url": "http://localhost:11434", "model": "llama2"}
)

# Create from environment
llm = LLMFactory.create_from_env(provider="openai")

# Create with fallback
llm = LLMFactory.create_with_fallback(
    providers=["openai", "ollama"],
    configs=[openai_config, ollama_config]
)
```

## Common Pitfalls

1. **Don't block the event loop** - Use async/await everywhere for I/O
2. **Handle streaming errors** - Networks fail, implement error recovery
3. **Validate configuration early** - Use Pydantic models, fail fast
4. **Don't assume message order** - Different providers handle system messages differently
5. **Token counting is approximate** - Use provider-specific tokenizers when available
6. **Test with mocks first** - Don't hit real APIs during development
7. **Document provider quirks** - Each LLM has unique behaviors, document them

## Documentation Standards

All public classes and methods MUST have docstrings:

```python
class OllamaProvider(LLMInterface):
    """
    Ollama provider for local LLM inference.
    
    Ollama runs models locally without API keys. Requires Ollama
    to be installed and running on the specified base_url.
    
    Args:
        base_url: Ollama API endpoint (default: http://localhost:11434)
        model: Model name (default: llama2)
        timeout: Request timeout in seconds (default: 30.0)
    
    Examples:
        >>> provider = OllamaProvider(model="llama2")
        >>> response = await provider.generate([
        ...     Message(role=MessageRole.USER, content="Hello")
        ... ])
        
    See Also:
        - https://ollama.ai/
        - bruno-core LLMInterface documentation
    """
```

## Questions & Support

- Check `IMPLEMENTATION_PLAN.md` for task tracking
- See `project-overview.md` for detailed architecture
- Reference bruno-core: https://github.com/meggy-ai/bruno-core
- Look at existing examples in `examples/` directory
