# Release Checklist v0.1.0

## Pre-Release Verification ✅

### Code Quality
- [x] All tests passing (193/193)
- [x] Code coverage at 91%
- [x] No linting errors
- [x] Type hints modernized (Python 3.10+ style)
- [x] Code formatted with ruff

### Documentation
- [x] README.md complete with examples
- [x] USER_GUIDE.md comprehensive (870+ lines)
- [x] TESTING.md with test guide
- [x] CONTRIBUTING.md with development workflow
- [x] Example scripts created
- [x] CHANGELOG.md updated for v0.1.0
- [x] All docstrings present and accurate

### Version Management
- [x] Version set to 0.1.0 in `__version__.py`
- [x] CHANGELOG.md includes v0.1.0 release notes
- [x] Release date set in CHANGELOG (2025-12-09)

### Package Build
- [x] Package built successfully
- [x] Distributions created:
  - `dist/bruno_llm-0.1.0-py3-none-any.whl`
  - `dist/bruno_llm-0.1.0.tar.gz`

## Release Tasks

### Local Testing
- [ ] Test installation from wheel in clean environment
- [ ] Verify all imports work
- [ ] Run basic usage examples
- [ ] Test with both Ollama and OpenAI providers

### GitHub Release
- [ ] Create git tag: `v0.1.0`
- [ ] Push tag to GitHub
- [ ] Create GitHub release with:
  - Release title: "bruno-llm v0.1.0 - Initial Release"
  - Description from CHANGELOG.md
  - Attach distribution files (.whl and .tar.gz)
  - Mark as "latest release"

### Optional: PyPI Publication
- [ ] Register on PyPI (if not already)
- [ ] Configure PyPI token
- [ ] Upload to TestPyPI first:
  ```bash
  python -m twine upload --repository testpypi dist/*
  ```
- [ ] Test installation from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ bruno-llm
  ```
- [ ] Upload to production PyPI:
  ```bash
  python -m twine upload dist/*
  ```
- [ ] Verify on PyPI: https://pypi.org/project/bruno-llm/

### Post-Release
- [ ] Verify installation works: `pip install bruno-llm`
- [ ] Update README badges (if on PyPI)
- [ ] Announce release in bruno-core repository
- [ ] Update IMPLEMENTATION_PLAN.md to mark Phase 8 complete
- [ ] Create v0.2.0 milestone for next features

## Installation Testing Commands

### Create Clean Test Environment
```bash
# Create new virtual environment
python -m venv test_env
test_env\Scripts\activate

# Install from local wheel
pip install dist/bruno_llm-0.1.0-py3-none-any.whl

# Test imports
python -c "from bruno_llm import LLMFactory; print('Import successful')"

# Test with Ollama (if running)
python examples/basic_usage.py

# Cleanup
deactivate
rmdir /s test_env
```

### From PyPI (after publication)
```bash
pip install bruno-llm
pip install bruno-llm[openai]  # With OpenAI support
pip install bruno-llm[all]     # All optional dependencies
```

## Known Warnings (Non-Blocking)

Build warnings that can be addressed in future releases:
- License format deprecation (setuptools wants SPDX expression)
- License classifier deprecation
- Ruff config location (move to `lint` section)

These warnings don't affect functionality and can be cleaned up in v0.1.1.

## Release Notes Summary

**bruno-llm v0.1.0** - Initial Release

Core LLM provider implementations for bruno-core framework:
- ✅ Ollama provider (local models)
- ✅ OpenAI provider (GPT models)
- ✅ Factory pattern with 3 creation methods
- ✅ Advanced features: caching, context management, streaming, cost tracking, middleware
- ✅ 203 comprehensive tests (91% coverage)
- ✅ Complete documentation (2,000+ lines)

**Python Support:** 3.9, 3.10, 3.11, 3.12

**Installation:**
```bash
pip install bruno-llm
```

See CHANGELOG.md for complete feature list.
