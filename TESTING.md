# Testing Guide

## Running Tests

### Default Test Run
```bash
pytest tests/
```
This runs all tests except those marked as WIP (work-in-progress).

**Result**: 193 passed, 3 skipped, 7 deselected (WIP)

### Run All Tests (Including WIP)
```bash
pytest tests/ -m ""
```
Or explicitly include WIP tests:
```bash
pytest tests/ -m "wip or not wip"
```

### Run Only WIP Tests
```bash
pytest tests/ -m wip
```
These are integration tests that require real Ollama setup with llama2 model.

### Run Without Integration Tests
```bash
pytest tests/ -m "not integration"
```

### Run Without WIP and Integration
```bash
pytest tests/ -m "not wip and not integration"
```

## Test Markers

### `@pytest.mark.wip`
Work-in-progress tests that require external setup:
- Real Ollama server with llama2 model installed
- May fail if dependencies not configured
- **Skipped by default** to keep CI/CD green

**Setup for WIP tests**:
```bash
# Install and start Ollama
ollama serve

# Pull llama2 model
ollama pull llama2

# Run WIP tests
pytest tests/ -m wip -v
```

### `@pytest.mark.integration`
Integration tests that test end-to-end functionality:
- May require real API connections
- Test actual provider behavior
- **Included by default** unless provider unavailable

### `@pytest.mark.slow`
Tests that take longer to execute:
- Performance benchmarks
- Large dataset processing
- Can be skipped with `-m "not slow"`

## Coverage

### Generate Coverage Report
```bash
pytest tests/ --cov=bruno_llm --cov-report=html
```
Then open `htmlcov/index.html` in your browser.

### Coverage by Module
```bash
pytest tests/ --cov=bruno_llm --cov-report=term-missing
```

### Current Coverage: 91%

High coverage modules (>90%):
- ✅ `cache.py`: 100%
- ✅ `cost_tracker.py`: 100%
- ✅ `exceptions.py`: 100%
- ✅ `context.py`: 96%
- ✅ `factory.py`: 94%
- ✅ `middleware.py`: 93%
- ✅ `streaming.py`: 93%
- ✅ `retry.py`: 92%

## Common Test Commands

### Quick Validation
```bash
# Fast test run (no coverage, skip slow tests)
pytest tests/ -m "not slow" --no-cov
```

### Full Test Suite
```bash
# All tests with coverage
pytest tests/ --cov=bruno_llm --cov-report=html
```

### Test Specific Module
```bash
pytest tests/test_cache.py -v
pytest tests/providers/test_ollama.py -v
```

### Test Specific Function
```bash
pytest tests/test_cache.py::test_cache_set_and_get -v
```

### Debug Failed Test
```bash
pytest tests/test_integration.py::test_ollama_integration_basic -vvs
```

## Continuous Integration

Our CI/CD pipeline runs:
```bash
pytest tests/ -m "not wip"
```

This ensures:
- ✅ All unit tests pass
- ✅ Mock-based integration tests pass
- ✅ 91% code coverage maintained
- ⏭️ WIP tests skipped (require external setup)

## WIP Test Status

Currently marked as WIP (require Ollama with llama2):
1. `test_ollama_integration_basic` - Basic Ollama generation
2. `test_ollama_integration_streaming` - Streaming responses
3. `test_caching_integration` - Cache with real provider
4. `test_context_manager_integration` - Context truncation
5. `test_stream_aggregation_integration` - Word aggregation
6. `test_concurrent_requests` - Parallel requests
7. `test_system_prompt_integration` - System prompt handling

To enable these tests:
```bash
# 1. Install Ollama
# 2. Start Ollama service
ollama serve

# 3. Pull model
ollama pull llama2

# 4. Run WIP tests
pytest tests/ -m wip -v
```

## Troubleshooting

### Tests Hang
- Check if Ollama service is running
- Verify network connectivity
- Increase timeout in test configuration

### Import Errors
```bash
# Reinstall in development mode
pip install -e .
```

### Coverage Not Generated
```bash
# Install coverage plugin
pip install pytest-cov
```

### OpenAI Tests Skipped
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Run OpenAI tests
pytest tests/ -k openai -v
```

## Writing New Tests

### Mark WIP Tests
```python
@pytest.mark.wip
@pytest.mark.asyncio
async def test_new_feature():
    """Test requiring external setup."""
    pass
```

### Mark Integration Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_feature():
    """End-to-end test."""
    pass
```

### Mark Slow Tests
```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance():
    """Performance benchmark."""
    pass
```

## Test Organization

```
tests/
├── providers/              # Provider-specific tests
│   ├── test_ollama.py     # Ollama provider (20 tests)
│   └── test_openai.py     # OpenAI provider (22 tests)
├── test_cache.py          # Cache functionality (14 tests)
├── test_context.py        # Context management (18 tests)
├── test_cost_tracker.py   # Cost tracking (12+14 tests)
├── test_exceptions.py     # Exception handling (11 tests)
├── test_factory.py        # Factory pattern (22 tests)
├── test_integration.py    # Integration tests (15 tests, 7 WIP)
├── test_middleware.py     # Middleware system (20 tests)
├── test_rate_limiter.py   # Rate limiting (6 tests)
├── test_retry.py          # Retry logic (9 tests)
├── test_streaming.py      # Streaming utilities (16 tests)
└── test_token_counter.py  # Token counting (8 tests)
```

Total: 203 tests (193 passing by default, 7 WIP, 3 conditional)
