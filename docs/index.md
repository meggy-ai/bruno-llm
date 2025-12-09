# bruno-llm

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Tests](https://github.com/meggy-ai/bruno-llm/workflows/Tests/badge.svg)](https://github.com/meggy-ai/bruno-llm/actions)
[![Code Quality](https://github.com/meggy-ai/bruno-llm/workflows/Code%20Quality/badge.svg)](https://github.com/meggy-ai/bruno-llm/actions)

Production-ready LLM provider implementations for the [bruno-core](https://github.com/meggy-ai/bruno-core) framework.

## Features

✨ **Multiple Providers**
- Ollama (local models)
- OpenAI (GPT models)
- More coming soon (Claude, Gemini, Azure)

🚀 **Advanced Features**
- Response caching (100% test coverage)
- Context window management (96% coverage)
- Stream aggregation (93% coverage)
- Cost tracking (98% coverage)
- Middleware system (93% coverage)

🛠️ **Production Ready**
- 203 comprehensive tests
- 91% code coverage
- Type hints throughout
- Async-first design

## Quick Start

### Installation

```bash
pip install bruno-llm
```

### Basic Usage

```python
from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory

# Create provider
llm = LLMFactory.create("ollama", {"model": "llama2"})

# Generate response
messages = [Message(role=MessageRole.USER, content="Hello!")]
response = await llm.generate(messages)
print(response)
```

## Documentation Sections

- **[Getting Started](getting-started/installation.md)** - Installation and quick start guide
- **[User Guide](user-guide/overview.md)** - Comprehensive usage documentation
- **[API Reference](api/factory.md)** - Complete API documentation
- **[Development](development/contributing.md)** - Contributing guidelines

## Links

- **GitHub:** [meggy-ai/bruno-llm](https://github.com/meggy-ai/bruno-llm)
- **Issues:** [Report a bug](https://github.com/meggy-ai/bruno-llm/issues)
- **Changelog:** [View releases](about/changelog.md)

## License

MIT License - see [LICENSE](about/license.md) for details.
