# Testing

See [TESTING.md](https://github.com/meggy-ai/bruno-llm/blob/main/TESTING.md) for complete testing guide.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=bruno_llm --cov-report=html

# Run specific test file
pytest tests/test_factory.py

# Run integration tests
pytest -m integration
```

## Test Organization

- `tests/providers/` - Provider-specific tests
- `tests/test_*.py` - Feature tests
- 203 total tests, 91% coverage
