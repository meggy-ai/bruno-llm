# Changelog

All notable changes to bruno-llm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Embedding Support (0.2.0 Features)
- **Embedding Provider Interface**: Complete `EmbeddingInterface` implementation from bruno-core
  - Unified interface for text embeddings across providers
  - Similarity calculation utilities
  - Batch processing support
- **Ollama Embedding Provider**: Local embedding generation
  - Support for `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`
  - Privacy-focused local processing
  - No usage costs or rate limits
  - 29 comprehensive tests (91% coverage)
- **OpenAI Embedding Provider**: Cloud embedding service
  - Support for `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
  - High-quality embeddings with batch processing
  - Cost optimization and monitoring
- **Embedding Factory**: Easy provider instantiation
  - `EmbeddingFactory.create()` - Create from config dict
  - `EmbeddingFactory.create_from_env()` - Create from environment
  - `EmbeddingFactory.create_with_fallback()` - Provider fallback chain

#### Bruno-Core Compatibility
- **Interface Compatibility Tests**: 24 tests validating bruno-core integration
  - Method signature verification using inspect module
  - Async generator validation for streaming
  - Parameter compatibility checks
  - Interface inheritance validation
- **Bruno-Core Validation Script**: Comprehensive compatibility verification
  - All providers properly implement LLMInterface and EmbeddingInterface
  - Factory integration confirmed working
  - Full compatibility with bruno-core ecosystem

#### Enhanced Documentation
- **API Reference Documentation**: Complete API coverage
  - Main API reference with all interfaces and providers
  - Provider-specific guides (OpenAI, Ollama)
  - Comprehensive embedding guide with examples
  - Integration patterns and best practices
- **Updated User Guides**: Enhanced with embedding functionality
  - Embedding usage examples and patterns
  - RAG (Retrieval-Augmented Generation) implementations
  - Similarity search and semantic capabilities
  - Updated troubleshooting for embedding issues
- **Provider Documentation**: Detailed provider-specific guides
  - Installation and setup instructions
  - Configuration options and examples
  - Performance optimization tips
  - Integration patterns and use cases

#### Testing Improvements
- **Extended Test Coverage**: 288+ total tests with 89% coverage
  - 29 Ollama embedding provider tests
  - 24 bruno-core compatibility tests
  - Comprehensive error handling and edge case coverage
  - Mock-based testing for reliable CI/CD

### Technical Enhancements
- **Updated Provider Methods**: Enhanced LLM providers with explicit parameter support
  - Added temperature and max_tokens parameters to generate() and stream() methods
  - Improved bruno-core interface compliance
  - Better parameter validation and type safety

### Planned (Future Releases)
- Anthropic Claude provider
- Google Gemini provider
- Azure OpenAI support
- Function calling support
- Prompt templates
- Advanced RAG utilities

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
