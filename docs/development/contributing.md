# Contributing

See [CONTRIBUTING.md](https://github.com/meggy-ai/bruno-llm/blob/main/CONTRIBUTING.md) in the repository for detailed contribution guidelines.

## Quick Start

1. Fork and clone the repository
2. Install dependencies: `pip install -e ".[all]"`
3. **Install pre-commit hooks:** `pre-commit install`
4. Make changes and run tests: `pytest tests/`
5. Submit a pull request

## Pre-commit Hooks

**Critical:** Always install pre-commit hooks to catch issues before CI:

```bash
pip install pre-commit
pre-commit install
```

See [Pre-commit Setup](pre-commit.md) for complete guide.
