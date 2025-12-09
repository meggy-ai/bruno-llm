# Changelog

All notable changes to bruno-llm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Anthropic Claude provider
- Google Gemini provider  
- Azure OpenAI support
- Function calling support
- Prompt templates
- Batch request processing

## [0.1.0] - 2025-12-09

### Added

#### Core Features
- **LLM Provider Interface**: Unified interface implementing bruno-core's `LLMInterface`
- **Ollama Provider**: Complete local LLM support
  - Async HTTP client with streaming
  - Model management and health checks
  - Full bruno-core interface implementation
- **OpenAI Provider**: Full OpenAI API integration
  - GPT-3.5 and GPT-4 support
  - Streaming responses
  - Accurate token counting with tiktoken
  - Per-request cost tracking
- **Factory Pattern**: Easy provider instantiation
  - `LLMFactory.create()` - Create from config dict
  - `LLMFactory.create_from_env()` - Create from environment
  - `LLMFactory.create_with_fallback()` - Provider fallback chain

#### Advanced Features  
- **Response Caching**: LRU cache with TTL (100% coverage)
  - Configurable size and expiration
  - Hit/miss statistics
  - Automatic cleanup
- **Context Window Management**: Intelligent truncation (96% coverage)
  - 4 truncation strategies (OLDEST_FIRST, MIDDLE_OUT, SLIDING_WINDOW, SMART)
  - Model-specific limits (GPT-4, Claude, Llama)
  - Token counting and limit checking
- **Stream Aggregation**: Flexible batching (93% coverage)
  - Word, sentence, fixed-size, time-based
  - StreamProcessor with callbacks
- **Cost Tracking**: Comprehensive monitoring (98% coverage)
  - Per-request and model-specific costs
  - CSV/JSON export
  - Budget checking with warnings
  - Time-range reports
- **Middleware System**: Extensible pipeline (93% coverage)
  - Logging, caching, validation, retry middleware
  - Composable chains
  - Request/response hooks

#### Base Utilities
- **Token Counter**: Accurate counting (83% coverage)
  - SimpleTokenCounter (character-based)
  - TikTokenCounter (OpenAI)
- **Rate Limiter**: Token bucket algorithm (76% coverage)
- **Retry Logic**: Exponential backoff (92% coverage)
- **Exception Hierarchy**: 10 specific exceptions (100% coverage)

#### Testing & Quality
- **203 Tests**: 193 passing, 91% coverage
  - 42 provider tests
  - 80 base utility tests
  - 22 factory tests
  - 15 integration tests
  - 44 other tests
- **Test Organization**:
  - `@pytest.mark.wip` for work-in-progress
  - `@pytest.mark.integration` for end-to-end
  - `@pytest.mark.slow` for performance tests
- **Quality Tools**:
  - Ruff formatting and linting
  - Type hints throughout
  - Pydantic v2 validation

#### Documentation
- **README.md**: Comprehensive overview (543 lines)
  - Quick start and installation
  - Advanced features guide
  - API reference
  - Troubleshooting
- **USER_GUIDE.md**: Complete tutorial (870+ lines)
  - Installation and setup
  - Provider-specific guides
  - Advanced usage patterns
  - Best practices
- **TESTING.md**: Test execution guide
- **CONTRIBUTING.md**: Development guidelines
- **Examples**: 2 working scripts (400+ lines)

### Technical Details
- **Python**: 3.9, 3.10, 3.11, 3.12
- **Core Dependencies**:
  - bruno-core >= 0.1.0
  - httpx >= 0.24.0
  - aiohttp >= 3.8.0
  - pydantic >= 2.0.0
  - structlog >= 23.1.0
- **Optional**: openai >= 1.0.0, tiktoken >= 0.5.0

### Architecture
- Async-first design
- Type-safe with Pydantic
- Modular and extensible
- Production-ready error handling

[Unreleased]: https://github.com/meggy-ai/bruno-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/meggy-ai/bruno-llm/releases/tag/v0.1.0
