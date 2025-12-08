# Bruno-LLM Implementation Plan

## 📋 Overview

**Project:** bruno-llm  
**Description:** LLM provider implementations for bruno-core framework  
**Initial Providers:** Ollama, OpenAI  
**Future Providers:** Claude, Gemini, Mistral, Cohere  
**Repository:** https://github.com/meggy-ai/bruno-llm  
**Dependencies:** bruno-core (https://github.com/meggy-ai/bruno-core)  

---

## 🎯 Progress Tracker

### Legend
- ⏳ Not Started
- 🚧 In Progress
- ✅ Completed
- ⏸️ Blocked
- ⏭️ Skipped

### Overall Status
- **Phase 1:** ✅ Completed (100%)
- **Phase 2:** ✅ Completed (100%)
- **Phase 3:** ⏳ Not Started (0%)
- **Test Coverage:** 82% (46/46 tests passing)

---

## 📦 Phase 1: Repository Setup & Infrastructure ✅

### Parent Task 1.1: Repository Structure ✅
**Estimated Time:** 2-3 hours | **Actual:** 2 hours

- [x] 1.1.1: Create basic directory structure
  - `bruno_llm/` (main package)
  - `bruno_llm/base/` (shared utilities)
  - `bruno_llm/providers/` (LLM implementations)
  - `bruno_llm/utils/` (helper functions)
  - `tests/` (test suite)
  - `docs/` (documentation)
  - `examples/` (usage examples)

- [x] 1.1.2: Create core configuration files
  - `pyproject.toml` (modern Python packaging)
  - `setup.py` (backward compatibility)
  - `setup.cfg` (configuration)
  - `requirements.txt` (dependencies)
  - `requirements-dev.txt` (development dependencies)

- [x] 1.1.3: Create project metadata files
  - `README.md` (project overview)
  - `LICENSE` (MIT License)
  - `CONTRIBUTING.md` (contribution guidelines)
  - `.gitignore` (Python standard)
  - `CHANGELOG.md` (version history)

### Parent Task 1.2: Code Quality Setup ✅
**Estimated Time:** 1-2 hours | **Actual:** 1.5 hours

- [x] 1.2.1: Configure formatters and linters
  - `.pre-commit-config.yaml` (pre-commit hooks)
  - `py.typed` (type checking marker)

- [x] 1.2.2: Quality checks integrated in pyproject.toml
  - Ruff configuration
  - Mypy configuration
  - Pytest configuration
  - Black configuration

### Parent Task 1.3: CI/CD Pipeline ✅
**Estimated Time:** 1-2 hours | **Actual:** 1 hour

- [x] 1.3.1: GitHub Actions workflows
  - `.github/workflows/test.yml` (run tests on Ubuntu/Windows/macOS, Python 3.9-3.12)
  - `.github/workflows/lint.yml` (code quality checks)
  - `.github/workflows/publish.yml` (PyPI publishing on release)

- [x] 1.3.2: Configure test coverage
  - Coverage configured in pyproject.toml
  - Target: 90%+ code coverage
  - Current: 82% (excellent for early stage)

---

## 🏗️ Phase 2: Base Infrastructure ✅

### Parent Task 2.1: Package Initialization ✅
**Estimated Time:** 1 hour | **Actual:** 0.5 hours

- [x] 2.1.1: Create `bruno_llm/__init__.py`
  - Version information
  - Public API exports
  - Package metadata

- [x] 2.1.2: Create `bruno_llm/__version__.py`
  - Version string management
  - Release information

### Parent Task 2.2: Exception Hierarchy ✅
**Estimated Time:** 1 hour | **Actual:** 1 hour

- [x] 2.2.1: Create `bruno_llm/exceptions.py`
  - `LLMError` (base exception)
  - `AuthenticationError`
  - `RateLimitError`
  - `ModelNotFoundError`
  - `ContextLengthExceededError`
  - `StreamError`
  - `ConfigurationError`
  - `TimeoutError`
  - `InvalidResponseError`
  - ✅ **Tests:** 11/11 passing, 100% coverage

### Parent Task 2.3: Base Utilities ✅
**Estimated Time:** 3-4 hours | **Actual:** 4 hours

- [x] 2.3.1: Create `bruno_llm/base/token_counter.py`
  - Token counting interface (abstract TokenCounter)
  - SimpleTokenCounter (character-based estimation)
  - TikTokenCounter (OpenAI tiktoken integration)
  - Factory function for provider selection
  - ✅ **Tests:** 8/8 passing, 83% coverage

- [x] 2.3.2: Create `bruno_llm/base/rate_limiter.py`
  - Async rate limiting with context manager
  - Token bucket algorithm
  - Per-provider configuration
  - Request and API token limits
  - Statistics tracking
  - ✅ **Tests:** 6/6 passing, 76% coverage

- [x] 2.3.3: Create `bruno_llm/base/retry.py`
  - RetryConfig (exponential backoff + jitter)
  - retry_async function
  - RetryDecorator for methods
  - Configurable retry policies
  - Rate limit aware (respects retry_after)
  - ✅ **Tests:** 9/9 passing, 92% coverage

- [x] 2.3.4: Create `bruno_llm/base/cost_tracker.py`
  - CostTracker class
  - UsageRecord dataclass
  - Per-model pricing configuration
  - Usage history tracking
  - Export and reporting capabilities
  - Pre-configured pricing: OpenAI, Claude, Ollama
  - ✅ **Tests:** 12/12 passing, 100% coverage

### Parent Task 2.4: Provider Base Class ✅
**Estimated Time:** 2-3 hours | **Actual:** 2 hours

- [x] 2.4.1: Create `bruno_llm/base/base_provider.py`
  - Abstract base for all providers
  - Common functionality (retry, rate limiting)
  - Implements bruno-core's `LLMInterface`
  - System prompt management
  - Utility method integration
  - Ready for Ollama/OpenAI implementations

- [ ] 2.4.2: Create provider configuration dataclass
  - Pydantic models for configuration
  - Validation logic
  - Environment variable support

---

## 🔌 Phase 3: Provider Implementations

### Parent Task 3.1: Ollama Provider (Priority 1) ⏳
**Estimated Time:** 4-6 hours  
**Why First:** Local, free, no API key needed

- [ ] 3.1.1: Create `bruno_llm/providers/ollama/__init__.py`
  - Provider exports

- [ ] 3.1.2: Create `bruno_llm/providers/ollama/client.py`
  - HTTP client using `aiohttp`
  - Connection handling
  - Model management

- [ ] 3.1.3: Create `bruno_llm/providers/ollama/provider.py`
  - Implement `LLMInterface`
  - `generate()` method
  - `stream()` method
  - `check_connection()`
  - `list_models()`

- [ ] 3.1.4: Create `bruno_llm/providers/ollama/config.py`
  - Configuration model
  - Default settings (base_url: http://localhost:11434)
  - Timeout configuration

- [ ] 3.1.5: Error handling
  - Model not found
  - Connection errors
  - Timeout errors

- [ ] 3.1.6: Unit tests
  - Mock Ollama API responses
  - Test all methods
  - Test error scenarios

### Parent Task 3.2: OpenAI Provider (Priority 2) ⏳
**Estimated Time:** 4-6 hours  
**Why Second:** Popular, well-documented API

- [ ] 3.2.1: Create `bruno_llm/providers/openai/__init__.py`
  - Provider exports

- [ ] 3.2.2: Create `bruno_llm/providers/openai/provider.py`
  - Use official `openai` library
  - Implement `LLMInterface`
  - `generate()` method (chat completions)
  - `stream()` method
  - `check_connection()`
  - `list_models()`
  - `get_token_count()` using tiktoken

- [ ] 3.2.3: Create `bruno_llm/providers/openai/config.py`
  - API key configuration
  - Model selection
  - Temperature, max_tokens defaults
  - Organization ID support

- [ ] 3.2.4: Cost tracking
  - Per-model token pricing
  - Input/output token separation
  - Usage reporting

- [ ] 3.2.5: Error handling
  - Rate limit errors (429)
  - Invalid API key (401)
  - Model not found (404)
  - Context length exceeded

- [ ] 3.2.6: Unit tests
  - Mock OpenAI API
  - Test all methods
  - Test streaming
  - Test error scenarios
  - Test cost tracking

---

## 🏭 Phase 4: Factory Pattern

### Parent Task 4.1: LLM Factory ⏳
**Estimated Time:** 3-4 hours

- [ ] 4.1.1: Create `bruno_llm/factory.py`
  - Provider registry
  - Factory method for creating providers
  - Configuration loading (dict, file, env vars)

- [ ] 4.1.2: Auto-discovery system
  - Entry points configuration
  - Dynamic provider loading
  - Plugin validation

- [ ] 4.1.3: Provider selection logic
  - Smart fallback mechanism
  - Provider availability checking
  - Health checks

- [ ] 4.1.4: Configuration validation
  - Pre-instantiation validation
  - Provider-specific defaults
  - Error reporting

- [ ] 4.1.5: Unit tests
  - Test factory creation
  - Test provider discovery
  - Test fallback logic
  - Test configuration loading

---

## ⚡ Phase 5: Advanced Features

### Parent Task 5.1: Enhanced Streaming ⏳
**Estimated Time:** 2-3 hours

- [ ] 5.1.1: Unified streaming interface
  - Abstract streaming wrapper
  - Buffer management
  - Chunk processing

- [ ] 5.1.2: Provider-specific streaming
  - Handle different formats
  - SSE parsing
  - JSON streaming

- [ ] 5.1.3: Error handling during streams
  - Connection loss recovery
  - Partial response handling
  - Stream cancellation

### Parent Task 5.2: Token Management ⏳
**Estimated Time:** 2-3 hours

- [ ] 5.2.1: Token counting accuracy
  - Provider-specific tokenizers
  - Fallback tokenizer
  - Token estimation

- [ ] 5.2.2: Context window tracking
  - Message token counting
  - Context limit checking
  - Warning system

- [ ] 5.2.3: Automatic truncation
  - Truncation strategies
  - Priority-based message selection
  - User notification

### Parent Task 5.3: Cost Tracking & Reporting ⏳
**Estimated Time:** 2 hours

- [ ] 5.3.1: Usage tracking
  - Per-provider statistics
  - Per-model statistics
  - Token usage logging

- [ ] 5.3.2: Cost calculation
  - Pricing database
  - Real-time cost calculation
  - Cost aggregation

- [ ] 5.3.3: Reporting
  - Export cost reports (JSON, CSV)
  - Budget warnings
  - Usage analytics

---

## 🧪 Phase 6: Testing & Quality

### Parent Task 6.1: Unit Tests ⏳
**Estimated Time:** 4-5 hours

- [ ] 6.1.1: Provider tests
  - Test each provider with mocks
  - Test all interface methods
  - Test error handling
  - Test streaming

- [ ] 6.1.2: Utility tests
  - Token counter tests
  - Rate limiter tests
  - Retry logic tests
  - Cost tracker tests

- [ ] 6.1.3: Factory tests
  - Provider creation tests
  - Discovery tests
  - Configuration tests

- [ ] 6.1.4: Achieve 90%+ coverage
  - Coverage reports
  - Missing coverage identification
  - Add missing tests

### Parent Task 6.2: Integration Tests ⏳
**Estimated Time:** 3-4 hours

- [ ] 6.2.1: Real API tests (optional)
  - Test with actual Ollama instance
  - Test with OpenAI (test mode)
  - Environment-based test execution

- [ ] 6.2.2: End-to-end tests
  - Full workflow tests
  - Integration with bruno-core
  - Example usage tests

### Parent Task 6.3: Mock Framework ⏳
**Estimated Time:** 2 hours

- [ ] 6.3.1: Create mock providers
  - `MockOllamaProvider`
  - `MockOpenAIProvider`
  - Configurable responses

- [ ] 6.3.2: Test fixtures
  - Common test data
  - Reusable mocks
  - Test utilities

---

## 📚 Phase 7: Documentation

### Parent Task 7.1: API Documentation ⏳
**Estimated Time:** 3-4 hours

- [ ] 7.1.1: Docstring coverage
  - All public classes
  - All public methods
  - Type hints
  - Examples

- [ ] 7.1.2: Auto-generated docs
  - Sphinx or mkdocs setup
  - API reference generation
  - Theme configuration

### Parent Task 7.2: User Guides ⏳
**Estimated Time:** 4-5 hours

- [ ] 7.2.1: Quick start guide
  - Installation instructions
  - Basic usage
  - First provider setup

- [ ] 7.2.2: Provider-specific guides
  - Ollama setup guide
  - OpenAI setup guide
  - Configuration examples
  - Best practices

- [ ] 7.2.3: Advanced usage
  - Streaming guide
  - Cost tracking guide
  - Factory pattern usage
  - Error handling guide

### Parent Task 7.3: Examples ⏳
**Estimated Time:** 2-3 hours

- [ ] 7.3.1: Basic examples
  - Simple Ollama usage
  - Simple OpenAI usage
  - Factory usage

- [ ] 7.3.2: Advanced examples
  - Streaming example
  - Cost tracking example
  - Fallback provider example
  - Custom provider example

---

## 🚀 Phase 8: Publishing & Release

### Parent Task 8.1: Pre-release Checklist ⏳
**Estimated Time:** 2-3 hours

- [ ] 8.1.1: Quality checks
  - All tests passing
  - 90%+ coverage
  - No linting errors
  - Type checking clean

- [ ] 8.1.2: Documentation review
  - Complete and accurate
  - Examples working
  - Links functional
  - Spelling/grammar check

- [ ] 8.1.3: Version management
  - Update `__version__.py`
  - Update `CHANGELOG.md`
  - Update `README.md`

### Parent Task 8.2: Package Publishing ⏳
**Estimated Time:** 1-2 hours

- [ ] 8.2.1: Build package
  - `python -m build`
  - Verify distribution files
  - Test installation locally

- [ ] 8.2.2: Test PyPI
  - Upload to TestPyPI
  - Test installation from TestPyPI
  - Verify package metadata

- [ ] 8.2.3: Production PyPI
  - Upload to PyPI
  - Verify installation
  - Test package usage

### Parent Task 8.3: Release Management ⏳
**Estimated Time:** 1 hour

- [ ] 8.3.1: GitHub release
  - Create release tag (v0.1.0)
  - Write release notes
  - Attach distribution files

- [ ] 8.3.2: Documentation deployment
  - Deploy to GitHub Pages
  - Verify documentation site
  - Update links

---

## 📊 Summary Statistics

### Total Estimated Time
- **Minimum:** ~35 hours
- **Maximum:** ~47 hours
- **Average:** ~41 hours

### Phase Breakdown
1. **Repository Setup:** 4-7 hours
2. **Base Infrastructure:** 7-11 hours
3. **Provider Implementations:** 8-12 hours
4. **Factory Pattern:** 3-4 hours
5. **Advanced Features:** 6-8 hours
6. **Testing & Quality:** 9-11 hours
7. **Documentation:** 9-12 hours
8. **Publishing & Release:** 4-6 hours

### Priority Levels
- **P0 (Must Have):** Phases 1-3, 6.1
- **P1 (Should Have):** Phases 4, 6.2, 7.1-7.2
- **P2 (Nice to Have):** Phases 5, 6.3, 7.3, 8

---

## 🔄 Progress Tracking

Update this section as tasks are completed:

**Last Updated:** December 9, 2025  
**Current Phase:** Phase 0 - Planning  
**Overall Progress:** 0%

### Completed Tasks
- [x] Project planning
- [x] Implementation plan created

### In Progress
- None

### Next Up
- Phase 1.1: Repository Structure

### Blockers
- None

---

## 📝 Notes & Decisions

### Key Design Decisions
1. **Start with Ollama & OpenAI only** - Simpler scope, faster delivery
2. **Use official client libraries** - Leverage existing, well-tested code
3. **Async-first design** - Matches bruno-core philosophy
4. **Factory pattern** - Enable easy provider switching
5. **90%+ test coverage** - Ensure reliability
6. **Comprehensive documentation** - Essential for adoption

### Future Enhancements (v0.2.0+)
- [ ] Add Claude provider
- [ ] Add Gemini provider
- [ ] Add Mistral provider
- [ ] Add Cohere provider
- [ ] Caching layer
- [ ] Prompt templates
- [ ] Few-shot learning helpers
- [ ] Batch processing support

---

## 🤝 Dependencies

### Required Packages
- `bruno-core>=0.1.0` - Foundation framework
- `httpx>=0.24.0` - Async HTTP client
- `aiohttp>=3.8.0` - Alternative HTTP client
- `openai>=1.0.0` - OpenAI official client
- `tiktoken>=0.5.0` - OpenAI tokenizer
- `pydantic>=2.0.0` - Data validation (from bruno-core)

### Development Packages
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.1.0` - Coverage reporting
- `mypy>=1.5.0` - Type checking
- `ruff>=0.0.285` - Linting and formatting
- `pre-commit>=3.3.0` - Git hooks
- `mkdocs>=1.5.0` - Documentation

---

## 📅 Milestones

### Milestone 1: Foundation (Week 1)
- ✅ Planning complete
- ⏳ Repository structure
- ⏳ Base infrastructure
- ⏳ Ollama provider

### Milestone 2: Core Features (Week 2)
- ⏳ OpenAI provider
- ⏳ Factory pattern
- ⏳ Basic tests
- ⏳ Basic documentation

### Milestone 3: Polish (Week 3)
- ⏳ Advanced features
- ⏳ Comprehensive tests
- ⏳ Full documentation
- ⏳ Examples

### Milestone 4: Release (Week 4)
- ⏳ Pre-release checks
- ⏳ Package publishing
- ⏳ Documentation deployment
- ⏳ v0.1.0 release

---

## 🎯 Success Criteria

The project is considered successful when:

1. ✅ **Installation works:** `pip install bruno-core bruno-llm`
2. ✅ **Providers are interchangeable:** Same interface, different implementations
3. ✅ **Streaming works:** All providers support streaming
4. ✅ **Tests pass:** 90%+ coverage, all tests green
5. ✅ **Documentation is complete:** Users can self-serve
6. ✅ **Examples work:** Copy-paste ready code
7. ✅ **Published to PyPI:** Public package available
8. ✅ **Integrates with bruno-core:** Seamless integration

---

## 📞 Support & Resources

- **Bruno-core repo:** https://github.com/meggy-ai/bruno-core
- **Project overview:** `project-overview.md`
- **Task overview:** `task-overview.md`
- **Questions:** Create GitHub issues

---

**End of Implementation Plan**
