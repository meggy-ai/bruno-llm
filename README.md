# bruno-llm test

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-288%20passed-success)](https://github.com/meggy-ai/bruno-llm)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)](https://github.com/meggy-ai/bruno-llm)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**bruno-llm** provides production-ready LLM and embedding provider implementations for the [bruno-core](https://github.com/meggy-ai/bruno-core) framework. Swap between different language model and embedding providers (Ollama, OpenAI, Anthropic, etc.) through unified interfaces with advanced features like caching, streaming, context management, cost tracking, and semantic search capabilities.

## ✨ Key Features

### Core Capabilities
- **🔌 Unified Interface**: All providers implement `LLMInterface` and `EmbeddingInterface` from bruno-core
- **🧠 Embedding Support**: Text embeddings for semantic search, similarity, and RAG applications
- **⚡ Async-First**: Built on asyncio for non-blocking I/O
- **🏭 Factory Pattern**: Easy provider instantiation and fallback chains for both LLM and embedding providers
- **🔄 Streaming Support**: Real-time token streaming with aggregation
- **📝 Type Safe**: Complete type hints with Pydantic v2 validation

### Advanced Features
- **💾 Response Caching**: LRU cache with TTL to reduce API costs
- **🧠 Context Management**: Intelligent message truncation with multiple strategies
- **💰 Cost Tracking**: Detailed usage analytics with CSV/JSON export
- **🔁 Smart Retry**: Exponential backoff with configurable strategies
- **⏱️ Rate Limiting**: Token bucket algorithm for API compliance
- **🔌 Middleware System**: Extensible request/response pipeline

### Quality & Testing
- **🧪 Well Tested**: 288+ tests with 89% code coverage
- **🛡️ Error Handling**: Comprehensive exception hierarchy
- **📊 Production Ready**: Used in real-world applications
- **📖 Documented**: Extensive documentation and examples

## 🚀 Quick Start

### Installation

```bash
# Basic installation (includes Ollama support)
pip install bruno-llm

# With OpenAI support
pip install bruno-llm[openai]

# For development
pip install bruno-llm[dev]

# Install from source
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm
pip install -e .
```

### Basic Usage

#### Using the Factory Pattern (Recommended)

```python
import asyncio
from bruno_llm import LLMFactory
from bruno_core.models import Message, MessageRole

async def main():
    # Create provider using factory
    llm = LLMFactory.create(
        provider="ollama",
        config={"model": "llama2", "base_url": "http://localhost:11434"}
    )

    # Or from environment variables
    # llm = LLMFactory.create_from_env("openai")

    # Check connection
    if await llm.check_connection():
        print("✅ Connected!")

    # Generate response
    messages = [
        Message(role=MessageRole.USER, content="What is Python?")
    ]
    response = await llm.generate(messages, max_tokens=100)
    print(response)

    # Stream response
    async for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)

    await llm.close()

asyncio.run(main())
```

#### Ollama (Local LLM)

```python
from bruno_llm.providers.ollama import OllamaProvider

# Initialize provider
llm = OllamaProvider(model="llama2")

# Generate response
response = await llm.generate(messages)
```

#### OpenAI (Cloud API)

```python
from bruno_llm.providers.openai import OpenAIProvider

# Initialize provider
llm = OpenAIProvider(api_key="sk-...", model="gpt-4")

# Generate with cost tracking
response = await llm.generate(messages)
cost = llm.cost_tracker.get_total_cost()
print(f"Cost: ${cost:.4f}")
```

### Embedding Usage

#### Quick Start with Embeddings

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# Create embedding provider
embedder = EmbeddingFactory.create("openai", {
    "api_key": "sk-...",
    "model": "text-embedding-3-small"
})

# Generate single embedding
text = "Machine learning transforms data into insights"
embedding = await embedder.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch processing
texts = [
    "Artificial intelligence revolutionizes technology",
    "Machine learning enables pattern recognition",
    "Deep learning uses neural networks"
]
embeddings = await embedder.embed_texts(texts)
```

#### Similarity Search

```python
# Calculate similarity between texts
query = "artificial intelligence"
documents = ["ML algorithms", "Cooking recipes", "AI systems"]

query_emb = await embedder.embed_text(query)
doc_embeddings = await embedder.embed_texts(documents)

# Find most similar
similarities = [
    embedder.calculate_similarity(query_emb, doc_emb)
    for doc_emb in doc_embeddings
]

most_similar_idx = similarities.index(max(similarities))
print(f"Most similar: {documents[most_similar_idx]}")
```

#### Simple RAG (Retrieval-Augmented Generation)

```python
from bruno_llm.factory import LLMFactory

# Combine embeddings + LLM for RAG
llm = LLMFactory.create_from_env("openai")
embedder = EmbeddingFactory.create_from_env("openai")

# Build knowledge base
knowledge = ["Bruno-LLM provides LLM interfaces", "It supports OpenAI and Ollama"]
kb_embeddings = await embedder.embed_texts(knowledge)

# Answer question using relevant knowledge
question = "What does Bruno-LLM provide?"
q_embedding = await embedder.embed_text(question)

# Find most relevant knowledge
similarities = [
    embedder.calculate_similarity(q_embedding, kb_emb)
    for kb_emb in kb_embeddings
]
best_match_idx = similarities.index(max(similarities))
context = knowledge[best_match_idx]

# Generate answer with context
messages = [
    Message(role=MessageRole.SYSTEM, content="Answer based on context."),
    Message(role=MessageRole.USER, content=f"Context: {context}\nQuestion: {question}")
]
answer = await llm.generate(messages)
```

### Advanced Features

#### Response Caching

```python
from bruno_llm import LLMFactory
from bruno_llm.base import ResponseCache

llm = LLMFactory.create("ollama", {"model": "llama2"})
cache = ResponseCache(max_size=100, ttl=300)

# First call - cache miss
response = await llm.generate(messages, temperature=0.0)
cache.set(messages, response, temperature=0.0)

# Second call - cache hit (no API call!)
cached = cache.get(messages, temperature=0.0)
if cached:
    print("Retrieved from cache!")
```

#### Context Management

```python
from bruno_llm.base import ContextWindowManager, ContextLimits, TruncationStrategy

# Create context manager
context_mgr = ContextWindowManager(
    model="gpt-4",
    limits=ContextLimits(max_tokens=8000, max_output_tokens=500),
    strategy=TruncationStrategy.SMART
)

# Check and truncate if needed
if not context_mgr.check_limit(messages):
    messages = context_mgr.truncate(messages)

response = await llm.generate(messages)
```

#### Stream Aggregation

```python
from bruno_llm.base import StreamAggregator

aggregator = StreamAggregator(strategy="word")

# Get word-by-word instead of character-by-character
async for word in aggregator.aggregate(llm.stream(messages)):
    print(word, end=" ", flush=True)
```

#### Provider Fallback

```python
from bruno_llm import LLMFactory

# Try OpenAI first, fallback to Ollama
llm = await LLMFactory.create_with_fallback(
    providers=["openai", "ollama"],
    configs=[
        {"api_key": "sk-...", "model": "gpt-4"},
        {"model": "llama2"}
    ]
)
```

#### Cost Tracking & Export

```python
# Track usage
response = await llm.generate(messages)

# Get report
report = llm.cost_tracker.get_usage_report()
print(f"Total cost: ${report['total_cost']:.4f}")
print(f"Total tokens: {report['total_tokens']}")

# Export to CSV
llm.cost_tracker.export_to_csv("costs.csv")

# Check budget
status = llm.cost_tracker.check_budget(budget_limit=10.0)
if not status["within_budget"]:
    print("⚠️ Budget exceeded!")
```

## 📦 Supported Providers

| Provider | Status | Streaming | Cost Tracking | Token Counting |
|----------|--------|-----------|---------------|----------------|
| **Ollama** | ✅ Ready | ✅ Yes | ✅ Yes (free) | ✅ Approximate |
| **OpenAI** | ✅ Ready | ✅ Yes | ✅ Yes | ✅ tiktoken |
| **Anthropic Claude** | 🚧 Planned | - | - | - |
| **Google Gemini** | 🚧 Planned | - | - | - |

### Provider Setup

#### Ollama
```bash
# Install Ollama
# Visit: https://ollama.ai/

# Start Ollama
ollama serve

# Pull a model
ollama pull llama2
ollama pull codellama
ollama pull mistral
```

#### OpenAI
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Or in Python
llm = OpenAIProvider(api_key="sk-...")
```
```

## 🔧 Configuration

### Environment Variables

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
OPENAI_ORG_ID=org-...
```

### Provider Configuration

```python
from bruno_llm import LLMFactory

# Direct instantiation
from bruno_llm.providers.ollama import OllamaProvider

llm = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30.0
)

# Using factory
llm = LLMFactory.create(
    provider="ollama",
    config={"model": "llama2", "timeout": 60.0}
)

# From environment
llm = LLMFactory.create_from_env("openai")
```

## 🏗️ Architecture

bruno-llm is built with a modular architecture:

```
bruno_llm/
├── base/                    # Core utilities
│   ├── base_provider.py    # Abstract provider base
│   ├── cache.py            # Response caching
│   ├── context.py          # Context window management
│   ├── cost_tracker.py     # Usage and cost tracking
│   ├── middleware.py       # Request/response middleware
│   ├── rate_limiter.py     # Rate limiting
│   ├── retry.py            # Retry logic
│   ├── streaming.py        # Stream utilities
│   └── token_counter.py    # Token counting
├── providers/              # Provider implementations
│   ├── ollama/            # Ollama provider
│   └── openai/            # OpenAI provider
├── exceptions.py           # Exception hierarchy
└── factory.py             # Factory pattern
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=bruno_llm --cov-report=html

# Run specific test file
pytest tests/test_cache.py -v

# Skip WIP tests (default)
pytest tests/ -m "not wip"

# Run integration tests (requires Ollama/OpenAI)
pytest tests/ -m integration
```

**Test Stats**: 203 tests, 193 passing, 91% coverage

See [TESTING.md](TESTING.md) for detailed testing guide.

## 📚 Documentation

- **[Examples](examples/)** - Complete working examples
  - [Basic Usage](examples/basic_usage.py) - Getting started
  - [Advanced Features](examples/advanced_features.py) - Caching, context, cost tracking
- **[Testing Guide](TESTING.md)** - How to run and write tests
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Development roadmap
- **[API Reference](#api-reference)** - Detailed API documentation

## 📖 Examples

### Integration with bruno-core

```python
from bruno_core.base import BaseAssistant
from bruno_llm import LLMFactory

# Create assistant with LLM provider
llm = LLMFactory.create("ollama", {"model": "llama2"})
assistant = BaseAssistant(llm=llm, memory=your_memory)

await assistant.initialize()

# Process messages
response = await assistant.process_message(user_message)
```

### Middleware System

```python
from bruno_llm.base import LoggingMiddleware, CachingMiddleware, MiddlewareChain

# Create middleware chain
chain = MiddlewareChain([
    LoggingMiddleware(),
    CachingMiddleware(cache),
])

# Apply to requests (provider integration)
# Middleware hooks into before_request, after_response, on_stream_chunk, on_error
```

### Custom Provider

```python
from bruno_core.interfaces import LLMInterface
from bruno_llm.base import BaseProvider

class CustomProvider(BaseProvider):
    async def generate(self, messages, **kwargs):
        # Your implementation
        pass

    async def stream(self, messages, **kwargs):
        # Your streaming implementation
        pass

# Register with factory
from bruno_llm import LLMFactory
LLMFactory.register("custom", CustomProvider)
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ruff format .

# Lint code
ruff check .
```

## 🐛 Troubleshooting

### Ollama Connection Issues

```python
# Check if Ollama is running
import httpx
response = httpx.get("http://localhost:11434/api/tags")
print(response.json())

# Common issues:
# - Ollama not running: Start with `ollama serve`
# - Model not installed: Run `ollama pull llama2`
# - Wrong URL: Check base_url configuration
```

### OpenAI Authentication

```python
# Verify API key
import openai
openai.api_key = "sk-..."
models = openai.Model.list()  # Should not raise error

# Common issues:
# - Invalid API key: Check at https://platform.openai.com/api-keys
# - Rate limiting: Use rate_limiter configuration
# - Billing: Ensure account has credits
```

### Import Errors

```bash
# Reinstall package
pip uninstall bruno-llm
pip install -e .

# Or force reinstall dependencies
pip install --force-reinstall bruno-llm[dev]
```

## 📋 API Reference

### Factory

```python
from bruno_llm import LLMFactory

# Create provider
llm = LLMFactory.create(provider: str, config: dict, **kwargs)

# From environment
llm = LLMFactory.create_from_env(provider: str, prefix: str = "")

# With fallback
llm = await LLMFactory.create_with_fallback(
    providers: list[str],
    configs: list[dict]
)

# List available providers
providers = LLMFactory.list_providers()

# Check if registered
is_available = LLMFactory.is_registered("ollama")
```

### LLMInterface Methods

All providers implement these methods:

```python
# Generate complete response
response: str = await llm.generate(
    messages: List[Message],
    max_tokens: int = None,
    temperature: float = 0.7,
    **kwargs
)

# Stream response
async for chunk in llm.stream(
    messages: List[Message],
    max_tokens: int = None,
    **kwargs
):
    print(chunk, end="")

# Get token count
count: int = llm.get_token_count(text: str)

# Check connection
is_connected: bool = await llm.check_connection()

# List models
models: List[str] = await llm.list_models()

# Get model info
info: dict = llm.get_model_info()

# System prompt
llm.set_system_prompt(prompt: str)
prompt: str = llm.get_system_prompt()

# Close resources
await llm.close()
```

### Advanced Utilities

```python
# Response caching
from bruno_llm.base import ResponseCache
cache = ResponseCache(max_size=100, ttl=300)
cache.set(messages, response, **params)
cached = cache.get(messages, **params)
stats = cache.get_stats()

# Context management
from bruno_llm.base import ContextWindowManager, ContextLimits
manager = ContextWindowManager(model="gpt-4", limits=ContextLimits(...))
is_within = manager.check_limit(messages)
truncated = manager.truncate(messages)
stats = manager.get_stats(messages)

# Stream aggregation
from bruno_llm.base import StreamAggregator
aggregator = StreamAggregator(strategy="word")
async for word in aggregator.aggregate(stream):
    print(word)

# Cost tracking
report = llm.cost_tracker.get_usage_report()
llm.cost_tracker.export_to_csv("costs.csv")
status = llm.cost_tracker.check_budget(budget_limit=10.0)
```

## 🔒 Security

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` files (add to `.gitignore`)
- **Rate Limiting**: Built-in rate limiting prevents abuse
- **Input Validation**: All inputs validated with Pydantic
- **Error Handling**: Sensitive data not exposed in errors

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [bruno-core](https://github.com/meggy-ai/bruno-core) framework
- Inspired by [LangChain](https://github.com/hwchase17/langchain) and [LlamaIndex](https://github.com/jerryjliu/llama_index)
- Thanks to the [Ollama](https://ollama.ai/) and [OpenAI](https://openai.com/) teams

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/meggy-ai/bruno-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/meggy-ai/bruno-llm/discussions)
- **Email**: contact@meggy.ai

## 🗺️ Roadmap

- [x] Ollama provider
- [x] OpenAI provider
- [x] Response caching
- [x] Context management
- [x] Cost tracking
- [x] Middleware system
- [ ] Anthropic Claude provider
- [ ] Google Gemini provider
- [ ] Azure OpenAI support
- [ ] Streaming improvements
- [ ] Advanced retry strategies
- [ ] Prompt templates
- [ ] Function calling support

---

**Made with ❤️ by the Meggy AI team**
- **[Examples](./examples/)**

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bruno_llm --cov-report=html

# Run specific provider tests
pytest tests/providers/test_ollama.py -v

# Run integration tests (requires running services)
pytest -m integration
```

### Code Quality

```bash
# Format code
black bruno_llm tests

# Lint
ruff check bruno_llm tests

# Type check
mypy bruno_llm

# Run all checks
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code restructuring

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- **[bruno-core](https://github.com/meggy-ai/bruno-core)** - Foundation framework
- **bruno-memory** - Memory backend implementations (coming soon)
- **bruno-abilities** - Pre-built abilities (coming soon)
- **bruno-pa** - Personal assistant application (coming soon)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/meggy-ai/bruno-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/meggy-ai/bruno-llm/discussions)
- **Documentation**: [https://meggy-ai.github.io/bruno-llm/](https://meggy-ai.github.io/bruno-llm/)

---

Made with ❤️ by the Meggy AI team
