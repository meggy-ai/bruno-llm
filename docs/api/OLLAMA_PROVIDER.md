# Ollama Provider Documentation

## Overview

The Ollama provider enables local LLM and embedding model execution without external API dependencies. It connects to a locally running Ollama instance, providing privacy-focused AI capabilities with no usage costs or rate limits.

## Installation Requirements

First, install Ollama on your system:

### System Installation

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download and install from [https://ollama.ai/download](https://ollama.ai/download)

### Start Ollama Service

```bash
# Start Ollama service
ollama serve

# In another terminal, pull your first model
ollama pull llama2
ollama pull nomic-embed-text  # For embeddings
```

### Python Package

```bash
pip install bruno-llm  # Ollama support included by default
```

## Quick Start

### Basic LLM Usage

```python
from bruno_llm.providers.ollama import OllamaProvider
from bruno_core.models import Message, MessageRole

# Initialize the provider (assumes Ollama running on localhost:11434)
provider = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2"  # or any model you've pulled
)

# Create a conversation
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Explain the benefits of local AI models.")
]

# Generate response
response = await provider.generate(messages, temperature=0.7, max_tokens=500)
print(response)

# Stream response for real-time output
print("Streaming response:")
async for chunk in provider.stream(messages, temperature=0.7):
    print(chunk, end="", flush=True)
```

### Basic Embedding Usage

```python
from bruno_llm.providers.ollama import OllamaEmbeddingProvider

# Initialize embedding provider
embedder = OllamaEmbeddingProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text"  # Make sure this model is pulled
)

# Single text embedding
text = "Local AI models provide privacy and control"
embedding = await embedder.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
texts = [
    "Ollama runs models locally",
    "No internet required for inference",
    "Complete data privacy"
]
embeddings = await embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")

# Calculate similarity
similarity = embedder.calculate_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

## Configuration

### LLM Configuration

```python
from bruno_llm.providers.ollama import OllamaConfig, OllamaProvider

# Detailed configuration
config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama2:13b",  # Specify model variant
    timeout=60.0,        # Longer timeout for larger models
    keep_alive="10m",    # Keep model loaded for 10 minutes
    num_ctx=4096,        # Context window size
    num_predict=512,     # Max tokens to generate
    repeat_penalty=1.1,  # Reduce repetition
    temperature=0.8,     # Default temperature
    top_k=40,           # Top-k sampling
    top_p=0.9           # Top-p sampling
)

provider = OllamaProvider(config=config)
```

### Embedding Configuration

```python
from bruno_llm.providers.ollama import OllamaEmbeddingConfig, OllamaEmbeddingProvider

# Embedding configuration
config = OllamaEmbeddingConfig(
    base_url="http://localhost:11434",
    model="mxbai-embed-large",
    timeout=30.0,
    batch_size=16  # Process fewer texts at once for larger models
)

embedder = OllamaEmbeddingProvider(config=config)
```

## Environment Variables

Set up your environment for automatic configuration:

```bash
# Ollama service configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_HOST="0.0.0.0:11434"  # For Ollama service itself

# Bruno-LLM Ollama provider settings
export OLLAMA_MODEL="llama2"
export OLLAMA_TIMEOUT="60.0"
export OLLAMA_KEEP_ALIVE="5m"
export OLLAMA_NUM_CTX="4096"

# Embedding settings
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export EMBEDDING_TIMEOUT="30.0"
export EMBEDDING_BATCH_SIZE="32"
```

Create providers from environment:

```python
from bruno_llm.factory import LLMFactory
from bruno_llm.embedding_factory import EmbeddingFactory

# Auto-configure from environment
llm = LLMFactory.create_from_env("ollama")
embedder = EmbeddingFactory.create_from_env("ollama")
```

## Supported Models

### Popular LLM Models

| Model | Size | Description | Pull Command |
|-------|------|-------------|--------------|
| `llama2` | 7B | Meta's Llama 2 base model | `ollama pull llama2` |
| `llama2:13b` | 13B | Larger Llama 2 variant | `ollama pull llama2:13b` |
| `llama2:70b` | 70B | Largest Llama 2 model | `ollama pull llama2:70b` |
| `codellama` | 7B | Code-specialized Llama | `ollama pull codellama` |
| `codellama:13b` | 13B | Larger code model | `ollama pull codellama:13b` |
| `mistral` | 7B | Mistral AI's efficient model | `ollama pull mistral` |
| `mixtral` | 8x7B | Mixture of experts model | `ollama pull mixtral` |
| `neural-chat` | 7B | Intel's conversational model | `ollama pull neural-chat` |
| `starling-lm` | 7B | UC Berkeley's model | `ollama pull starling-lm` |
| `dolphin-mixtral` | 8x7B | Uncensored Mixtral variant | `ollama pull dolphin-mixtral` |

### Embedding Models

| Model | Dimensions | Description | Pull Command |
|-------|------------|-------------|--------------|
| `nomic-embed-text` | 768 | Nomic AI's embedding model | `ollama pull nomic-embed-text` |
| `mxbai-embed-large` | 1024 | MixedBread AI's large embeddings | `ollama pull mxbai-embed-large` |
| `snowflake-arctic-embed` | 1024 | Snowflake's Arctic embeddings | `ollama pull snowflake-arctic-embed` |
| `all-minilm` | 384 | Sentence Transformers compact model | `ollama pull all-minilm` |

### Model Management

```python
# List available models
models = await provider.list_models()
print("Available models:", models)

# Check if a specific model is available
if "llama2:13b" in models:
    # Switch to a different model
    provider = OllamaProvider(model="llama2:13b")
else:
    print("Model not found. Pull it first:")
    print("ollama pull llama2:13b")
```

## Advanced Features

### Custom Model Parameters

```python
# Advanced generation parameters
response = await provider.generate(
    messages=messages,
    temperature=0.8,        # Creativity (0.0-2.0)
    max_tokens=1000,        # Maps to num_predict
    top_k=40,              # Top-k sampling
    top_p=0.9,             # Top-p sampling
    repeat_penalty=1.1,     # Reduce repetition
    seed=42,               # Reproducible outputs
    stop=["\n\n", "###"]   # Stop sequences
)
```

### Streaming with Custom Processing

```python
import asyncio

async def stream_with_processing():
    """Stream with real-time processing and statistics."""

    messages = [Message(role=MessageRole.USER, content="Write a creative story")]

    word_count = 0
    char_count = 0
    start_time = time.time()

    print("Streaming response:")
    print("-" * 50)

    async for chunk in provider.stream(messages, temperature=0.9):
        print(chunk, end="", flush=True)

        # Update statistics
        word_count += len(chunk.split())
        char_count += len(chunk)

        # Show progress every 50 characters
        if char_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"\n[Stats: {word_count} words, {char_count} chars, {elapsed:.1f}s]", end="")

    total_time = time.time() - start_time
    print(f"\n\n📊 Final stats:")
    print(f"   Words: {word_count}")
    print(f"   Characters: {char_count}")
    print(f"   Time: {total_time:.2f}s")
    print(f"   Speed: {word_count/total_time:.1f} words/sec")
```

### Model Performance Optimization

```python
async def optimize_model_performance():
    """Configure Ollama for optimal performance."""

    # For faster response (lower quality)
    fast_provider = OllamaProvider(
        model="llama2",
        num_ctx=2048,      # Smaller context window
        num_predict=256,   # Fewer tokens
        top_k=20,         # More focused sampling
        temperature=0.3    # Less randomness
    )

    # For higher quality (slower)
    quality_provider = OllamaProvider(
        model="llama2:13b",  # Larger model
        num_ctx=4096,        # Full context
        num_predict=1000,    # More tokens
        temperature=0.7,     # Balanced creativity
        top_p=0.95          # Diverse sampling
    )

    # Choose based on use case
    provider = fast_provider if need_speed else quality_provider

    return await provider.generate(messages)
```

### Batch Processing for Embeddings

```python
async def process_document_embeddings(documents: List[str]):
    """Process large document collections efficiently."""

    embedder = OllamaEmbeddingProvider(
        model="nomic-embed-text",
        batch_size=16  # Conservative batch size
    )

    all_embeddings = []

    print(f"Processing {len(documents)} documents...")

    for i in range(0, len(documents), embedder.batch_size):
        batch = documents[i:i + embedder.batch_size]

        try:
            # Process batch
            batch_embeddings = await embedder.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

            # Progress update
            progress = len(all_embeddings) / len(documents) * 100
            print(f"Progress: {progress:.1f}% ({len(all_embeddings)}/{len(documents)})")

            # Small delay to prevent overwhelming Ollama
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Add empty embeddings for failed batch to maintain alignment
            all_embeddings.extend([None] * len(batch))

    return all_embeddings
```

## Error Handling

### Connection and Model Management

```python
from bruno_llm.exceptions import (
    ModelNotFoundError,
    LLMTimeoutError,
    LLMError
)

async def robust_ollama_usage():
    """Handle common Ollama issues gracefully."""

    provider = OllamaProvider(model="llama2")

    # Check Ollama service connectivity
    try:
        is_connected = await provider.check_connection()
        if not is_connected:
            print("❌ Ollama service not running. Start with: ollama serve")
            return None
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Ensure Ollama is installed and running:")
        print("  1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. Start: ollama serve")
        return None

    # Check model availability
    try:
        models = await provider.list_models()
        if "llama2" not in models:
            print("❌ llama2 model not found. Pull it first:")
            print("  ollama pull llama2")
            return None
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")

    # Generate with error handling
    try:
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        response = await provider.generate(messages, timeout=30.0)
        print(f"✅ Response: {response}")
        return response

    except ModelNotFoundError:
        print("❌ Model not loaded. Try: ollama pull llama2")
    except LLMTimeoutError:
        print("⏰ Request timed out. Large models may need more time.")
    except LLMError as e:
        print(f"❌ Generation error: {e}")

    return None
```

### Memory and Resource Management

```python
async def monitor_ollama_resources():
    """Monitor and manage Ollama resource usage."""

    import psutil
    import subprocess

    # Check system resources
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print(f"⚠️  High memory usage: {memory.percent:.1f}%")
        print("Consider using a smaller model or increasing RAM")

    # Check if Ollama is running
    try:
        result = subprocess.run(["pgrep", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("❌ Ollama process not found")
            return False
    except FileNotFoundError:
        print("ℹ️  Cannot check process status (pgrep not available)")

    # Check GPU availability (if using CUDA)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA GPU available")
        else:
            print("ℹ️  No CUDA GPU found, using CPU")
    except FileNotFoundError:
        print("ℹ️  nvidia-smi not found, likely using CPU")

    return True
```

## Performance Optimization

### Model Selection Strategy

```python
def select_optimal_model(task_type: str, performance_priority: str = "balanced"):
    """Select the best model for a specific task."""

    model_recommendations = {
        "chat": {
            "speed": "llama2",
            "balanced": "mistral",
            "quality": "llama2:13b"
        },
        "code": {
            "speed": "codellama",
            "balanced": "codellama:13b",
            "quality": "codellama:34b"
        },
        "creative": {
            "speed": "neural-chat",
            "balanced": "starling-lm",
            "quality": "mixtral"
        },
        "analysis": {
            "speed": "mistral",
            "balanced": "llama2:13b",
            "quality": "mixtral"
        }
    }

    return model_recommendations.get(task_type, {}).get(performance_priority, "llama2")

# Usage
model = select_optimal_model("code", "quality")
provider = OllamaProvider(model=model)
```

### Connection Pooling and Reuse

```python
class OllamaManager:
    """Manage multiple Ollama providers efficiently."""

    def __init__(self):
        self.providers = {}

    def get_provider(self, model: str, **config):
        """Get or create a provider for the specified model."""
        if model not in self.providers:
            self.providers[model] = OllamaProvider(model=model, **config)
        return self.providers[model]

    async def cleanup(self):
        """Clean up all providers."""
        for provider in self.providers.values():
            await provider.__aexit__(None, None, None)
        self.providers.clear()

# Global manager instance
ollama_manager = OllamaManager()

# Usage
async def multi_model_processing():
    # Different models for different tasks
    chat_provider = ollama_manager.get_provider("mistral")
    code_provider = ollama_manager.get_provider("codellama")

    # Use providers without recreating connections
    chat_response = await chat_provider.generate(chat_messages)
    code_response = await code_provider.generate(code_messages)

    return chat_response, code_response
```

### Concurrent Processing

```python
async def concurrent_ollama_requests():
    """Process multiple requests concurrently with Ollama."""

    # Note: Ollama typically processes requests sequentially
    # But you can prepare multiple providers or use async queuing

    provider = OllamaProvider(model="llama2")

    requests = [
        [Message(role=MessageRole.USER, content="Explain AI")],
        [Message(role=MessageRole.USER, content="What is Python?")],
        [Message(role=MessageRole.USER, content="How do computers work?")]
    ]

    # Process with controlled concurrency
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests

    async def process_request(messages):
        async with semaphore:
            return await provider.generate(messages, max_tokens=200)

    # Process all requests
    responses = await asyncio.gather(*[
        process_request(messages) for messages in requests
    ])

    return responses
```

## Integration Examples

### Local RAG (Retrieval-Augmented Generation)

```python
from typing import List, Tuple
import numpy as np

class LocalRAGSystem:
    """Privacy-focused RAG using only local Ollama models."""

    def __init__(self):
        self.llm = OllamaProvider(model="mistral")
        self.embedder = OllamaEmbeddingProvider(model="nomic-embed-text")
        self.knowledge_base = []  # (text, embedding) pairs

    async def add_document(self, text: str):
        """Add a document to the knowledge base."""
        embedding = await self.embedder.embed_text(text)
        self.knowledge_base.append((text, embedding))

    async def search_similar(self, query: str, top_k: int = 3) -> List[str]:
        """Find most similar documents to the query."""
        query_embedding = await self.embedder.embed_text(query)

        similarities = []
        for text, doc_embedding in self.knowledge_base:
            similarity = self.embedder.calculate_similarity(query_embedding, doc_embedding)
            similarities.append((text, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in similarities[:top_k]]

    async def answer_question(self, question: str) -> str:
        """Answer a question using RAG."""
        # Find relevant documents
        relevant_docs = await self.search_similar(question)

        # Create context
        context = "\n\n".join(relevant_docs)

        # Generate answer
        messages = [
            Message(role=MessageRole.SYSTEM, content=
                "Answer the question based only on the provided context. "
                "If the context doesn't contain the information, say so."),
            Message(role=MessageRole.USER, content=f"Context:\n{context}\n\nQuestion: {question}")
        ]

        return await self.llm.generate(messages)

# Usage
rag = LocalRAGSystem()

# Add documents
await rag.add_document("Ollama is a tool for running large language models locally.")
await rag.add_document("Local AI models provide privacy and data security.")

# Ask questions
answer = await rag.answer_question("What are the benefits of local AI?")
print(answer)
```

### Document Processing Pipeline

```python
async def process_documents_locally(file_paths: List[str]):
    """Process documents entirely with local models."""

    llm = OllamaProvider(model="llama2")
    embedder = OllamaEmbeddingProvider(model="nomic-embed-text")

    results = []

    for file_path in file_paths:
        print(f"📄 Processing {file_path}...")

        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Generate summary
        summary_messages = [
            Message(role=MessageRole.SYSTEM, content=
                "Summarize the following document in 2-3 sentences."),
            Message(role=MessageRole.USER, content=content[:4000])  # Truncate if needed
        ]

        summary = await llm.generate(summary_messages, max_tokens=200)

        # Generate embedding for semantic search
        embedding = await embedder.embed_text(summary)

        # Extract key topics
        topics_messages = [
            Message(role=MessageRole.SYSTEM, content=
                "Extract 3-5 key topics from the document as a comma-separated list."),
            Message(role=MessageRole.USER, content=content[:4000])
        ]

        topics = await llm.generate(topics_messages, max_tokens=100)

        results.append({
            'file': file_path,
            'summary': summary,
            'topics': topics.split(', '),
            'embedding': embedding
        })

        print(f"✅ Completed {file_path}")

    return results

# Usage
documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
processed = await process_documents_locally(documents)

for doc in processed:
    print(f"\n📄 {doc['file']}")
    print(f"Summary: {doc['summary']}")
    print(f"Topics: {', '.join(doc['topics'])}")
```

## Best Practices

### 1. Model Management

```python
# Check model availability before use
async def ensure_model_available(model_name: str):
    """Ensure a model is available, pull if necessary."""

    provider = OllamaProvider()

    try:
        models = await provider.list_models()
        if model_name not in models:
            print(f"Model {model_name} not found. Please pull it:")
            print(f"ollama pull {model_name}")
            return False
        return True
    except Exception as e:
        print(f"Could not check models: {e}")
        return False

# Use before creating providers
if await ensure_model_available("llama2"):
    provider = OllamaProvider(model="llama2")
```

### 2. Resource Management

```python
# Monitor system resources
import psutil

def check_system_resources():
    """Check if system can handle the model."""
    memory = psutil.virtual_memory()

    if memory.available < 4 * 1024**3:  # Less than 4GB available
        print("⚠️  Low memory. Consider using a smaller model.")
        return "small"
    elif memory.available < 16 * 1024**3:  # Less than 16GB
        return "medium"
    else:
        return "large"

# Choose model based on resources
resource_level = check_system_resources()
models = {
    "small": "llama2",
    "medium": "llama2:13b",
    "large": "llama2:70b"
}

provider = OllamaProvider(model=models[resource_level])
```

### 3. Error Recovery

```python
async def resilient_generation(provider, messages, max_retries=3):
    """Generate with automatic retry and fallback."""

    for attempt in range(max_retries):
        try:
            return await provider.generate(messages)

        except LLMTimeoutError:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Fallback to smaller model
                print("Falling back to smaller model...")
                fallback_provider = OllamaProvider(model="llama2")
                return await fallback_provider.generate(messages)

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)
```

## Troubleshooting

### Common Issues

1. **Ollama Service Not Running**
   ```bash
   # Check if Ollama is running
   ps aux | grep ollama

   # Start Ollama
   ollama serve

   # Or run in background
   nohup ollama serve > ollama.log 2>&1 &
   ```

2. **Model Not Found**
   ```bash
   # List available models
   ollama list

   # Pull a model
   ollama pull llama2

   # Remove unused models to save space
   ollama rm unused-model
   ```

3. **Memory Issues**
   ```python
   # Monitor memory usage
   async def check_memory_usage():
       import psutil
       memory = psutil.virtual_memory()
       print(f"Memory usage: {memory.percent}%")
       print(f"Available: {memory.available / 1024**3:.1f} GB")
   ```

4. **Slow Performance**
   ```python
   # Optimize for speed
   fast_provider = OllamaProvider(
       model="llama2",      # Use smaller model
       num_ctx=2048,        # Reduce context
       num_predict=256,     # Limit output
       temperature=0.3      # Reduce randomness
   )
   ```

### Performance Monitoring

```python
import time
import asyncio

async def benchmark_ollama():
    """Benchmark Ollama performance."""

    provider = OllamaProvider(model="llama2")

    # Test message
    messages = [Message(role=MessageRole.USER, content="Explain AI in one paragraph.")]

    # Benchmark generation
    start_time = time.time()
    response = await provider.generate(messages, max_tokens=100)
    generation_time = time.time() - start_time

    # Calculate metrics
    tokens = len(response.split())
    tokens_per_second = tokens / generation_time

    print(f"📊 Performance Metrics:")
    print(f"   Response time: {generation_time:.2f}s")
    print(f"   Tokens generated: {tokens}")
    print(f"   Speed: {tokens_per_second:.1f} tokens/sec")
    print(f"   Response: {response[:100]}...")

    return {
        'time': generation_time,
        'tokens': tokens,
        'speed': tokens_per_second
    }

# Run benchmark
results = await benchmark_ollama()
```

For more examples and deployment scenarios, see the [Bruno-LLM Examples](../examples/) directory.
