**bruno-llm**.

---

## BRUNO-LLM DEVELOPMENT PLAN

### **Goal**
The source code for bruno-core is at : https://github.com/meggy-ai/bruno-core?tab=readme-ov-file

Implement concrete LLM provider clients that follow `bruno-core`'s `LLMInterface`, enabling seamless swapping between different language models.

---

### **PHASE 1: Repository Setup** (Day 1)

#### 1.1 Repository Structure
- Create `bruno-llm` repository under `meggy-ai` organization
- Mirror bruno-core's structure (same quality standards)
- Setup: pyproject.toml, setup.py, README, LICENSE, CONTRIBUTING
- Configure: black, mypy, ruff, pre-commit hooks
- Add: bruno-core as dependency (`bruno-core>=0.1.0`)

#### 1.2 Documentation Setup
- Initialize mkdocs with same theme as bruno-core
- Create: docs/index.md, docs/quickstart.md, docs/providers/

#### 1.3 CI/CD Pipeline
- Copy GitHub Actions from bruno-core
- Add: Provider-specific integration tests (with API mocking)
- Setup: PyPI publishing workflow

---

### **PHASE 2: Base Infrastructure** (Day 1-2)

#### 2.1 Package Structure
```
bruno-llm/
├── bruno_llm/
│   ├── __init__.py
│   ├── base/              # Shared utilities for providers
│   ├── providers/         # Individual LLM implementations
│   ├── factory.py         # LLMFactory for creating providers
│   ├── utils/             # Token counting, rate limiting
│   └── exceptions.py      # LLM-specific exceptions
```

#### 2.2 Shared Utilities (Base Layer)
- **Token Counter**: Count tokens for different models (tiktoken, custom)
- **Rate Limiter**: Async rate limiting for API calls
- **Retry Logic**: Exponential backoff with jitter
- **Response Parser**: Parse streaming/non-streaming responses uniformly
- **Cost Tracker**: Track API usage costs per provider

#### 2.3 Exception Hierarchy
- `LLMError` (base)
- `AuthenticationError`
- `RateLimitError`
- `ModelNotFoundError`
- `ContextLengthExceededError`
- `StreamError`

---

### **PHASE 3: Provider Implementations** (Day 3-5)

Implement in order of priority:

#### 3.1 **Ollama Provider** (Day 3 - Priority 1)
**Why first:** Local, free, no API key needed, easiest to test

**Implementation checklist:**
- Implement `LLMInterface` from bruno-core
- Support: generate() and stream() methods
- Configuration: model name, base URL, timeout
- Handle: Model not found, connection errors
- Test: Against local Ollama instance
- Document: Setup guide for Ollama

#### 3.2 **Claude Provider** (Day 4 - Priority 2)
**Why second:** Your preferred API, great for production

**Implementation checklist:**
- Implement `LLMInterface`
- Support: Messages API, streaming
- Configuration: API key, model, max tokens, temperature
- Handle: Rate limits, token counting (Claude-specific)
- Cost tracking: Input/output tokens separately
- Test: With mocked Anthropic API
- Document: API key setup, model selection

#### 3.3 **OpenAI Provider** (Day 5 - Priority 3)
**Why third:** Popular choice, good for comparison

**Implementation checklist:**
- Implement `LLMInterface`
- Support: Chat Completions API, streaming
- Configuration: API key, model, parameters
- Handle: OpenAI-specific errors
- Cost tracking: Per-model pricing
- Test: With mocked OpenAI API
- Document: GPT-4, GPT-3.5 setup

#### 3.4 **Gemini Provider** (Optional - Day 6)
- Google's Gemini API
- Free tier support
- Implementation similar to above

---

### **PHASE 4: Factory Pattern** (Day 6)

#### 4.1 LLM Factory
- **Purpose:** Create LLM instances from config
- **Registry system:** Auto-discover providers via entry points
- **Configuration loading:** From dict, file, environment variables
- **Provider selection:** Smart selection based on availability

#### 4.2 Factory Features
- Validate configuration before instantiation
- Support for fallback providers (if primary fails)
- Connection testing/health checks
- Provider-specific defaults

---

### **PHASE 5: Advanced Features** (Day 7-8)

#### 5.1 Streaming Support
- Unified streaming interface across all providers
- Handle provider-specific streaming formats
- Buffer management for partial tokens
- Error handling during streams

#### 5.2 Token Management
- Accurate token counting per provider
- Context window tracking
- Automatic truncation strategies
- Warning when approaching limits

#### 5.3 Cost Tracking
- Track API usage per provider
- Calculate costs based on token usage
- Export cost reports
- Budget warnings

#### 5.4 Rate Limiting
- Per-provider rate limits
- Async semaphores for concurrent requests
- Queue management
- Backoff strategies

---

### **PHASE 6: Plugin System Integration** (Day 8)

#### 6.1 Entry Points Configuration
Register providers in `pyproject.toml`:
```toml
[project.entry-points."bruno.llm_providers"]
ollama = "bruno_llm.providers.ollama:OllamaProvider"
claude = "bruno_llm.providers.claude:ClaudeProvider"
openai = "bruno_llm.providers.openai:OpenAIProvider"
```

#### 6.2 Plugin Discovery
- Validate each provider implements `LLMInterface`
- Lazy loading (don't import until needed)
- Provider capability detection

---

### **PHASE 7: Testing** (Day 9)

#### 7.1 Unit Tests
- Test each provider with mocked APIs
- Test factory pattern
- Test utilities (token counting, rate limiting)
- Test error handling

#### 7.2 Integration Tests
- Test against real APIs (with test accounts)
- Test streaming behavior
- Test concurrent requests
- Test rate limiting

#### 7.3 Mock Framework
- Create reusable mocks for each provider
- Make it easy for users to test their code

---

### **PHASE 8: Documentation** (Day 10)

#### 8.1 User Documentation
- Quick start guide
- Provider-specific setup guides
- Configuration examples
- Best practices for each provider

#### 8.2 API Documentation
- Auto-generate from docstrings
- Usage examples for each provider
- Migration guides (switching providers)

#### 8.3 Examples
- Basic usage example (all providers)
- Streaming example
- Cost tracking example
- Fallback provider example
- Custom provider example

---

### **PHASE 9: Publishing** (Day 10)

#### 9.1 Pre-release Checklist
- All tests pass
- Documentation complete
- CHANGELOG.md updated
- Version bump to 0.1.0

#### 9.2 PyPI Publishing
- Build distributions
- Publish to Test PyPI first
- Verify installation
- Publish to production PyPI

#### 9.3 GitHub Release
- Create release notes
- Tag version
- Update README badges

---

## **DELIVERABLES**

### Core Deliverables
✅ 3-4 production-ready LLM providers (Ollama, Claude, OpenAI, optionally Gemini)
✅ Factory pattern for easy provider creation
✅ Comprehensive test suite (unit + integration)
✅ Full documentation with examples
✅ Published to PyPI as `bruno-llm`

### Quality Standards
- 90%+ test coverage
- Type-checked with mypy (strict mode)
- Formatted with black
- Async-first (all I/O operations)
- Proper error handling
- Structured logging

### Integration Points
- Seamlessly works with bruno-core
- Uses bruno-core's models and interfaces
- Follows bruno-core's patterns and conventions
- Compatible with bruno-core's plugin system

---

## **SUCCESS CRITERIA**

By end of development, users should be able to:

1. **Install:** `pip install bruno-core bruno-llm`

2. **Use any provider with same interface:**
```python
from bruno_llm import OllamaProvider, ClaudeProvider
# Swap providers without changing code
llm = OllamaProvider(model="llama3.2")  # or ClaudeProvider()
```

3. **Stream responses:** All providers support streaming

4. **Track costs:** Automatic cost calculation per provider

5. **Handle errors:** Graceful degradation and retry logic

6. **Test easily:** Mock providers for unit testing

7. **Extend:** Add custom providers following the pattern

---

## **TIMELINE ESTIMATE**

- **Day 1-2:** Setup + Base Infrastructure
- **Day 3:** Ollama Provider
- **Day 4:** Claude Provider  
- **Day 5:** OpenAI Provider
- **Day 6:** Factory + Optional Gemini
- **Day 7-8:** Advanced Features
- **Day 9:** Testing
- **Day 10:** Documentation + Publishing

**Total: 10 days** (assuming ~6 hours/day of focused work)

---

## **DEPENDENCIES**

Required packages:
- `bruno-core>=0.1.0` (your foundation)
- `httpx` (async HTTP client)
- `anthropic` (for Claude)
- `openai` (for OpenAI)
- `tiktoken` (for token counting)
- `pydantic>=2.0` (already in bruno-core)
- `aiohttp` (for Ollama)

---

## **POST-LAUNCH**

After v0.1.0 release:
- Gather user feedback
- Add more providers (Mistral, Cohere, etc.)
- Improve token counting accuracy
- Add caching layer
- Add prompt templates
- Add few-shot learning helpers

---

**Ready to start Day 1?** Let me know and I'll give you the detailed repository setup commands and initial file structures!