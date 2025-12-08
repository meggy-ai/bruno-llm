# bruno-llm

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

**bruno-llm** provides production-ready LLM provider implementations for the [bruno-core](https://github.com/meggy-ai/bruno-core) framework. Easily swap between different language model providers (Ollama, OpenAI, Claude, etc.) through a unified interface.

## 🎯 Features

- **🔌 Unified Interface**: All providers implement bruno-core's `LLMInterface`
- **⚡ Async-First**: Non-blocking I/O for all operations
- **🔄 Streaming Support**: Real-time response streaming
- **💰 Cost Tracking**: Track API usage and costs per provider
- **🛡️ Error Handling**: Comprehensive exception hierarchy
- **🔁 Retry Logic**: Automatic retry with exponential backoff
- **⏱️ Rate Limiting**: Built-in rate limiting for API calls
- **🧪 Well Tested**: 90%+ code coverage
- **📝 Type Safe**: Full type hints and Pydantic validation

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install bruno-llm

# With OpenAI support
pip install bruno-llm[openai]

# For development
pip install bruno-llm[dev]
```

### Basic Usage

#### Ollama (Local LLM)

```python
import asyncio
from bruno_llm.providers.ollama import OllamaProvider
from bruno_core.models import Message, MessageRole

async def main():
    # Initialize Ollama provider
    llm = OllamaProvider(
        base_url="http://localhost:11434",
        model="llama2"
    )
    
    # Generate response
    messages = [
        Message(role=MessageRole.USER, content="Hello! Tell me a joke.")
    ]
    response = await llm.generate(messages)
    print(response)
    
    # Stream response
    print("\nStreaming response:")
    async for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

#### OpenAI

```python
import asyncio
from bruno_llm.providers.openai import OpenAIProvider
from bruno_core.models import Message, MessageRole

async def main():
    # Initialize OpenAI provider
    llm = OpenAIProvider(
        api_key="sk-...",
        model="gpt-4"
    )
    
    # Generate response
    messages = [
        Message(role=MessageRole.USER, content="Explain quantum computing")
    ]
    response = await llm.generate(messages, temperature=0.7)
    print(response)

asyncio.run(main())
```

### Integration with bruno-core

```python
from bruno_core.base import BaseAssistant
from bruno_llm.providers.ollama import OllamaProvider
from your_memory import YourMemory  # Your memory implementation

# Create LLM provider
llm = OllamaProvider(model="llama2")

# Create assistant
assistant = BaseAssistant(llm=llm, memory=YourMemory())
await assistant.initialize()

# Process messages
message = Message(role=MessageRole.USER, content="Hello!")
response = await assistant.process_message(message)
print(response.text)
```

## 📦 Supported Providers

| Provider | Status | Features |
|----------|--------|----------|
| **Ollama** | ✅ Available | Local inference, streaming, multiple models |
| **OpenAI** | ✅ Available | GPT-3.5/4, streaming, cost tracking, tiktoken |
| **Claude** | 🚧 Planned | Anthropic Claude models |
| **Gemini** | 🚧 Planned | Google Gemini models |

## 🏗️ Architecture

```
bruno-llm/
├── bruno_llm/
│   ├── __init__.py          # Public API
│   ├── __version__.py       # Version info
│   ├── exceptions.py        # Exception hierarchy
│   ├── factory.py           # Provider factory
│   ├── base/                # Base utilities
│   │   ├── base_provider.py # Abstract base provider
│   │   ├── token_counter.py # Token counting
│   │   ├── rate_limiter.py  # Rate limiting
│   │   ├── retry.py         # Retry logic
│   │   └── cost_tracker.py  # Cost tracking
│   └── providers/           # Provider implementations
│       ├── ollama/          # Ollama provider
│       └── openai/          # OpenAI provider
```

## 🔧 Configuration

### Environment Variables

```bash
# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2

# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4
export OPENAI_ORG_ID=org-...

# General
export BRUNO_LLM_LOG_LEVEL=INFO
export BRUNO_LLM_TIMEOUT=30.0
```

### Provider Configuration

```python
from bruno_llm.providers.ollama import OllamaProvider, OllamaConfig

# Using config object
config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30.0
)
llm = OllamaProvider(config=config)

# Using parameters
llm = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2",
    timeout=30.0
)
```

## 📚 Documentation

- **[Full Documentation](https://meggy-ai.github.io/bruno-llm/)**
- **[API Reference](https://meggy-ai.github.io/bruno-llm/api/)**
- **[Provider Guides](https://meggy-ai.github.io/bruno-llm/providers/)**
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
