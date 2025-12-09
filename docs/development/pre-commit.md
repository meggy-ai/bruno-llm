# Pre-commit Hooks

See [PRE_COMMIT_SETUP.md](https://github.com/meggy-ai/bruno-llm/blob/main/docs/PRE_COMMIT_SETUP.md) for complete documentation.

## Quick Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks (run once per repository)
pre-commit install
```

## What It Does

On every commit, automatically:

- ✅ Formats code with ruff
- ✅ Runs linting checks
- ✅ Performs type checking
- ✅ Validates YAML/JSON/TOML
- ✅ Removes trailing whitespace
- ✅ Checks for large files and secrets

## Benefits

- Catches issues before CI
- Auto-fixes formatting and linting
- Faster feedback (seconds vs minutes)
- Prevents CI failures
- Saves time and CI minutes
