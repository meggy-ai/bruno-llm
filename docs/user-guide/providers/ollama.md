# Ollama Provider Guide

The Ollama provider enables local LLM and embedding model execution, providing privacy-focused AI capabilities without external dependencies or usage costs.

## Overview

- **Privacy First**: All processing happens locally
- **No Usage Costs**: Free to run after initial setup
- **No Rate Limits**: Process as much as your hardware allows
- **Offline Capable**: Works without internet connection
- **Multiple Models**: Support for various LLM and embedding models

## Quick Setup

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Pull Models

```bash
# LLM models
ollama pull llama2
ollama pull mistral

# Embedding models
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

## LLM Usage

### Basic Configuration

```python
from bruno_llm.providers.ollama import OllamaProvider

# Create provider
llm = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2"
)
```

### Factory Pattern

```python
from bruno_llm.factory import LLMFactory

# Direct creation
llm = LLMFactory.create("ollama", {
    "base_url": "http://localhost:11434",
    "model": "mistral"
})

# From environment
llm = LLMFactory.create_from_env("ollama")
```

### Environment Variables

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2
export OLLAMA_TIMEOUT=60.0
```

## Embedding Usage

### Basic Configuration

```python
from bruno_llm.providers.ollama import OllamaEmbeddingProvider

# Create embedding provider
embedder = OllamaEmbeddingProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)
```

### Factory Pattern

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# Direct creation
embedder = EmbeddingFactory.create("ollama", {
    "base_url": "http://localhost:11434",
    "model": "mxbai-embed-large"
})

# From environment
embedder = EmbeddingFactory.create_from_env("ollama")
```

### Environment Variables

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export EMBEDDING_BATCH_SIZE=32
```

## Advanced Configuration

### Custom Parameters

```python
from bruno_llm.providers.ollama import OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama2:13b",
    timeout=120.0,
    keep_alive="10m",
    num_ctx=4096,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)

llm = OllamaProvider(config=config)
```

### Performance Tuning

```python
# For speed (lower quality)
fast_config = OllamaConfig(
    model="llama2",
    num_ctx=2048,
    num_predict=256,
    temperature=0.3
)

# For quality (slower)
quality_config = OllamaConfig(
    model="llama2:13b",
    num_ctx=4096,
    temperature=0.7,
    top_p=0.95
)
```

## Popular Models

### LLM Models

| Model | Size | Best For |
|-------|------|----------|
| `llama2` | 7B | General purpose |
| `llama2:13b` | 13B | Better reasoning |
| `mistral` | 7B | Fast, efficient |
| `codellama` | 7B | Code generation |
| `neural-chat` | 7B | Conversations |

### Embedding Models

| Model | Dimensions | Best For |
|-------|------------|----------|
| `nomic-embed-text` | 768 | General purpose |
| `mxbai-embed-large` | 1024 | High accuracy |
| `snowflake-arctic-embed` | 1024 | Retrieval tasks |
| `all-minilm` | 384 | Compact, fast |

## Complete Examples

See [Ollama Provider API Documentation](../../api/OLLAMA_PROVIDER.md) for comprehensive examples including:

- Installation and setup
- Model management
- Performance optimization
- Error handling
- Integration patterns
- Troubleshooting

## Troubleshooting

### Common Issues

**Ollama not running:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start service
ollama serve
```

**Model not found:**
```bash
# List models
ollama list

# Pull model
ollama pull llama2
ollama pull nomic-embed-text
```

**Slow performance:**
- Use smaller models
- Reduce context window
- Enable GPU if available

For detailed troubleshooting, see the [main troubleshooting guide](../../USER_GUIDE.md#troubleshooting).
