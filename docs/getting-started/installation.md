# Installation

## Requirements

- Python 3.9 or higher
- pip

## Install from PyPI

```bash
# Core installation
pip install bruno-llm

# With OpenAI support
pip install bruno-llm[openai]

# With all optional dependencies
pip install bruno-llm[all]
```

## Install from Source

```bash
# Clone repository
git clone https://github.com/meggy-ai/bruno-llm.git
cd bruno-llm

# Install bruno-core dependency
pip install git+https://github.com/meggy-ai/bruno-core.git@main

# Install in development mode
pip install -e ".[all]"
```

## Verify Installation

```python
import bruno_llm
print(bruno_llm.__version__)
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [User Guide](../user-guide/overview.md)
