# Contributing to bruno-llm

Thank you for your interest in contributing to bruno-llm! This document provides comprehensive guidelines and instructions for contributing.

## 🤝 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please be respectful and constructive in all interactions.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- bruno-core installed (automatically installed with bruno-llm)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/bruno-llm.git
   cd bruno-llm
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify Setup**
   ```bash
   pytest tests/
   ```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write code following our style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed
- Run tests frequently: `pytest tests/`

### 3. Run Quality Checks

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Run tests with coverage
pytest tests/ --cov=bruno_llm --cov-report=html
```

### 4. Commit Changes

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue #123"
git commit -m "docs: update API reference"
```

**Commit types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference to related issues (`Fixes #123`)
- Screenshots/examples if relevant

# Run specific test file
pytest tests/test_exceptions.py

# Run integration tests
pytest -m integration
```

### Code Quality

```bash
# Format code with black
black bruno_llm tests

# Lint with ruff
ruff check bruno_llm tests

# Type check with mypy
mypy bruno_llm

# Run all checks
pre-commit run --all-files
```

### Adding a New Provider

1. Create provider directory:
```bash
mkdir -p bruno_llm/providers/{provider_name}
```

2. Implement required files:
- `__init__.py` - Provider exports
- `provider.py` - LLMInterface implementation
- `config.py` - Pydantic configuration model

3. Write tests:
```bash
touch tests/providers/test_{provider_name}.py
```

4. Update documentation
5. Run tests and quality checks
6. Submit pull request

## Pull Request Process

1. **Branch Naming**: Use descriptive branch names
   - `feature/add-claude-provider`
   - `fix/streaming-error-handling`
   - `docs/update-readme`

2. **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat: add Claude provider`
   - `fix: handle streaming errors`
   - `docs: update README with examples`
   - `test: add unit tests for Ollama`

3. **Code Requirements**:
   - All tests must pass
   - Code coverage must be maintained (90%+)
   - Type hints required for all functions
   - Docstrings for all public APIs
   - No linting errors

4. **Pull Request Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Follows code style guidelines
- [ ] No breaking changes (or documented)
```

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use async/await for I/O operations

### Docstring Format
```python
def example_function(param1: str, param2: int) -> str:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        'test42'
    """
    pass
```

## Testing Guidelines

### Test Structure
- Use pytest fixtures for common setup
- Mock external API calls
- Test both success and error cases
- Use `pytest-asyncio` for async tests

### Test Example
```python
import pytest
from unittest.mock import AsyncMock, patch
from bruno_llm.providers.ollama import OllamaProvider

@pytest.mark.asyncio
async def test_generate():
    """Test basic generation."""
    provider = OllamaProvider()
    
    with patch.object(provider, '_make_request', new_callable=AsyncMock) as mock:
        mock.return_value = {"message": {"content": "response"}}
        
        result = await provider.generate([Message(role="user", content="hi")])
        
        assert result == "response"
        mock.assert_called_once()
```

## Documentation

- Update docstrings for any API changes
- Add examples for new features
- Update README.md if needed
- Keep CHANGELOG.md current

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and PRs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
