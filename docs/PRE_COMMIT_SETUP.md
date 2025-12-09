# Pre-commit Hooks Setup

Pre-commit hooks automatically check your code **before each commit** to catch issues that would fail CI.

## Why Pre-commit Hooks?

Without pre-commit hooks:
- ❌ Linting errors discovered in CI (after push)
- ❌ Formatting issues found during code review
- ❌ Type errors caught late in the process
- ❌ Wasted CI minutes on preventable failures

With pre-commit hooks:
- ✅ Errors caught **before commit**
- ✅ Code automatically formatted
- ✅ Fast feedback loop (seconds, not minutes)
- ✅ CI always passes

## Installation

### One-time Setup

```bash
# 1. Install pre-commit (if not already installed)
pip install pre-commit

# 2. Install the git hooks
pre-commit install

# 3. (Optional) Run on all files to check current state
pre-commit run --all-files
```

### What Gets Checked

On every `git commit`, these hooks run automatically:

1. **Basic Checks**
   - Remove trailing whitespace
   - Fix end-of-file issues
   - Validate YAML/JSON/TOML syntax
   - Check for large files
   - Detect merge conflicts
   - Detect private keys

2. **Code Quality**
   - **Ruff format** - Auto-format code
   - **Ruff lint** - Check code quality (auto-fix when possible)
   - **MyPy** - Type checking (relaxed settings)

## Usage

### Normal Workflow

```bash
# Make changes
vim bruno_llm/some_file.py

# Stage changes
git add bruno_llm/some_file.py

# Commit (hooks run automatically)
git commit -m "Your message"
```

If hooks fail:
- Code is automatically fixed (formatting, some lint issues)
- You'll see what failed
- Stage the auto-fixes and commit again:

```bash
git add -u
git commit -m "Your message"
```

### Bypass Hooks (Emergency Only)

```bash
# Skip hooks (not recommended!)
git commit --no-verify -m "Emergency fix"
```

### Run Manually

```bash
# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

### Update Hooks

```bash
# Update to latest versions
pre-commit autoupdate

# Re-install after changes
pre-commit install
```

## Configuration

Hooks are configured in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]           # Auto-fix issues
      - id: ruff-format          # Format code

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]  # Match CI settings
        exclude: ^tests/                   # Skip test files
```

## Troubleshooting

### Hooks Not Running

```bash
# Verify installation
pre-commit --version

# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Hooks Too Slow

```bash
# Skip specific hooks temporarily
SKIP=mypy git commit -m "Message"

# Or disable specific hooks in .pre-commit-config.yaml
```

### False Positives

If a hook incorrectly flags an issue:

1. **Fix the issue** (preferred)
2. **Add exception** in config file
3. **Disable hook** in `.pre-commit-config.yaml` (last resort)

## CI Alignment

Pre-commit hooks are configured to match CI workflows:

| Check | Pre-commit | CI (.github/workflows) |
|-------|-----------|----------------------|
| Formatting | ✅ ruff format | ✅ ruff format --check |
| Linting | ✅ ruff check | ✅ ruff check |
| Type checking | ✅ mypy (relaxed) | ✅ mypy (relaxed) |
| Tests | ❌ (too slow) | ✅ pytest |

## Best Practices

1. **Always install hooks** when cloning the repo
2. **Don't bypass hooks** unless emergency
3. **Update regularly** with `pre-commit autoupdate`
4. **Fix issues** rather than suppressing them
5. **Run manually** on all files after config changes

## Why Some Checks Are Skipped

- **Tests** - Too slow for pre-commit (run locally with `pytest`)
- **Coverage** - Needs full test run
- **Integration tests** - Require external services

These run in CI instead, which is fine since pre-commit catches 90% of issues.

## Summary

```bash
# Setup once
pip install pre-commit
pre-commit install

# Then forget about it - it just works!
git commit -m "Feature: Add new provider"
# Hooks run automatically ✨
```

Pre-commit hooks are your first line of defense against CI failures. Install them and save yourself time! 🚀
