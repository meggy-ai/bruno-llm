# GitHub Release - bruno-llm v0.1.0

## Release Information

**Version:** 0.1.0
**Release Date:** December 9, 2025
**Tag:** v0.1.0
**Type:** Initial Release

## Overview

First stable release of **bruno-llm**, providing production-ready LLM provider implementations for the bruno-core framework. This release includes comprehensive support for local and cloud-based language models with advanced features for caching, context management, streaming, and cost tracking.

## Key Features

### Core Providers
- **Ollama Provider** - Local LLM inference with zero-config setup
- **OpenAI Provider** - Cloud-based GPT models with full API support
- **Factory Pattern** - Three flexible instantiation methods

### Advanced Features
- **Response Caching** (100% test coverage) - Redis and in-memory caching with TTL
- **Context Window Management** (96% coverage) - Automatic token counting and truncation
- **Stream Aggregation** (93% coverage) - Timeout handling and chunk collection
- **Cost Tracking** (98% coverage) - Per-model pricing with detailed reports
- **Middleware System** (93% coverage) - Logging, validation, and custom processing

### Base Utilities
- **Token Counter** (83% coverage) - Multi-provider token estimation
- **Rate Limiter** (84% coverage) - Sliding window and token bucket algorithms
- **Retry Logic** (92% coverage) - Exponential backoff with jitter
- **Exception System** (100% coverage) - Comprehensive error hierarchy

## Installation

```bash
# Core installation
pip install bruno-llm

# With OpenAI support
pip install bruno-llm[openai]

# With all optional dependencies
pip install bruno-llm[all]
```

## Quick Start

```python
from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory

# Create Ollama provider (local)
llm = LLMFactory.create("ollama", {"model": "llama2"})

# Or OpenAI provider (cloud)
llm = LLMFactory.create("openai", {
    "api_key": "sk-...",
    "model": "gpt-4"
})

# Generate response
messages = [Message(role=MessageRole.USER, content="Hello!")]
response = await llm.generate(messages)
print(response)

# Stream response
async for chunk in llm.stream(messages):
    print(chunk, end="", flush=True)
```

## Technical Details

### Requirements
- **Python:** 3.9, 3.10, 3.11, 3.12
- **Core Dependencies:**
  - bruno-core >= 0.1.0
  - httpx >= 0.24.0
  - aiohttp >= 3.8.0
  - pydantic >= 2.0.0
  - structlog >= 23.1.0

### Test Coverage
- **Total Tests:** 203 (193 passing, 3 skipped, 7 WIP)
- **Code Coverage:** 91% overall
- **Test Categories:**
  - Provider tests: 42
  - Base utilities: 80
  - Factory pattern: 22
  - Integration tests: 15
  - Other: 44

### Documentation
- **README.md** - 543 lines with comprehensive examples
- **USER_GUIDE.md** - 870+ lines with installation, setup, usage, best practices
- **TESTING.md** - Complete testing guide
- **CONTRIBUTING.md** - Development workflow and guidelines
- **Example Scripts** - basic_usage.py, advanced_features.py

## What's Included

### Distribution Files
- `bruno_llm-0.1.0-py3-none-any.whl` - Wheel package
- `bruno_llm-0.1.0.tar.gz` - Source distribution

### Package Contents
- Full provider implementations (Ollama, OpenAI)
- All base utilities (cache, context, streaming, cost tracking, middleware, rate limiter, retry)
- Factory pattern for easy instantiation
- Type hints (py.typed included)
- Complete documentation

## Known Limitations

1. **bruno-core Dependency** - Requires bruno-core to be installed (not yet on PyPI)
2. **Provider Support** - Currently Ollama and OpenAI only (Claude, Gemini, Azure coming soon)
3. **Function Calling** - Not yet implemented (planned for v0.2.0)

## Upgrade Notes

This is the initial release, so no upgrade paths exist yet. Future releases will include migration guides.

## Coming in v0.2.0

- Claude provider (Anthropic API)
- Google Gemini provider
- Azure OpenAI support
- Function calling / tool use
- Prompt template system
- Batch processing support

## Breaking Changes

None - this is the initial release.

## Contributors

- Meggy AI Team

## Support & Resources

- **Documentation:** [USER_GUIDE.md](./USER_GUIDE.md)
- **Issues:** https://github.com/meggy-ai/bruno-llm/issues
- **Bruno Core:** https://github.com/meggy-ai/bruno-core
- **License:** MIT

## Installation Verification

After installation, verify with:

```python
import bruno_llm
print(bruno_llm.__version__)  # Should print: 0.1.0

from bruno_llm import LLMFactory
print("✓ Installation successful!")
```

## Acknowledgments

Built with support from the bruno-core framework team. Special thanks to all early testers and contributors.

---

**Full Changelog:** [CHANGELOG.md](./CHANGELOG.md)
