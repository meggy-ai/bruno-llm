# Embedding Guide

## Overview

Bruno-LLM provides powerful embedding capabilities through various providers. Embeddings convert text into numerical vectors that capture semantic meaning, enabling similarity search, clustering, classification, and retrieval-augmented generation (RAG) systems.

## Supported Embedding Providers

### OpenAI Embeddings
- **Models**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Features**: High-quality embeddings, batch processing, cost optimization
- **Use Cases**: Production applications, high-accuracy requirements

### Ollama Embeddings
- **Models**: `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`
- **Features**: Local processing, privacy-focused, no usage costs
- **Use Cases**: Privacy-sensitive applications, offline processing

## Quick Start

### Basic Usage

```python
from bruno_llm.embedding_factory import EmbeddingFactory
from bruno_llm.providers.openai import OpenAIEmbeddingProvider
from bruno_llm.providers.ollama import OllamaEmbeddingProvider

# OpenAI embeddings (cloud-based)
openai_embedder = OpenAIEmbeddingProvider(
    api_key="your-api-key",
    model="text-embedding-3-small"
)

# Ollama embeddings (local)
ollama_embedder = OllamaEmbeddingProvider(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)

# Single text embedding
text = "Machine learning transforms how we process information"
embedding = await openai_embedder.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch processing
texts = [
    "Artificial intelligence is reshaping technology",
    "Machine learning enables pattern recognition",
    "Deep learning uses neural networks"
]
embeddings = await ollama_embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Factory Pattern

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# Create from configuration
embedder = EmbeddingFactory.create(
    provider="openai",
    config={
        "api_key": "your-api-key",
        "model": "text-embedding-3-small"
    }
)

# Create from environment variables
embedder = EmbeddingFactory.create_from_env("ollama")

# Auto-selection with fallback
embedder = EmbeddingFactory.create_with_fallback(
    providers=["openai", "ollama"],
    configs=[openai_config, ollama_config]
)
```

## Embedding Models Comparison

### OpenAI Models

| Model | Dimensions | Max Input | Price/1M tokens | Use Case |
|-------|------------|-----------|-----------------|----------|
| `text-embedding-ada-002` | 1536 | 8191 | $0.10 | General purpose |
| `text-embedding-3-small` | 1536 | 8191 | $0.02 | Cost-effective |
| `text-embedding-3-large` | 3072 | 8191 | $0.13 | High performance |

### Ollama Models

| Model | Dimensions | Context Length | Description |
|-------|------------|----------------|-------------|
| `nomic-embed-text` | 768 | 2048 | Fast, general purpose |
| `mxbai-embed-large` | 1024 | 512 | High quality embeddings |
| `snowflake-arctic-embed` | 1024 | 512 | Specialized for retrieval |
| `all-minilm` | 384 | 256 | Compact, efficient |

## Advanced Usage

### Similarity Search

```python
import numpy as np
from typing import List, Tuple

class SemanticSearch:
    """Semantic search using embeddings."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []
        self.embeddings = []

    async def add_documents(self, texts: List[str]):
        """Add documents to the search index."""
        # Generate embeddings for all documents
        embeddings = await self.embedder.embed_texts(texts)

        self.documents.extend(texts)
        self.embeddings.extend(embeddings)

        print(f"Added {len(texts)} documents to index")

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        if not self.embeddings:
            return []

        # Get query embedding
        query_embedding = await self.embedder.embed_text(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedder.calculate_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
embedder = EmbeddingFactory.create_from_env("openai")
search = SemanticSearch(embedder)

# Build index
documents = [
    "Python is a versatile programming language",
    "Machine learning algorithms learn from data",
    "Neural networks are inspired by the brain",
    "Data science combines statistics and programming",
    "APIs enable software communication"
]
await search.add_documents(documents)

# Search
results = await search.search("programming languages", top_k=3)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

### Document Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

async def cluster_documents(embedder, texts: List[str], n_clusters: int = 3):
    """Cluster documents based on their embeddings."""

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = await embedder.embed_texts(texts)

    # Convert to numpy array
    embedding_matrix = np.array(embeddings)

    # Perform clustering
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding_matrix)

    # Group documents by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(texts[i])

    # Print results
    for cluster_id, cluster_docs in clusters.items():
        print(f"\n📁 Cluster {cluster_id}:")
        for doc in cluster_docs:
            print(f"  • {doc}")

    return clusters

# Usage
embedder = EmbeddingFactory.create_from_env("ollama")

documents = [
    "Python programming tutorial",
    "JavaScript web development",
    "Machine learning with scikit-learn",
    "Deep learning neural networks",
    "React frontend framework",
    "Vue.js component system",
    "Supervised learning algorithms",
    "Unsupervised learning techniques"
]

clusters = await cluster_documents(embedder, documents, n_clusters=3)
```

### RAG (Retrieval-Augmented Generation)

```python
from bruno_llm.factory import LLMFactory
from bruno_llm.embedding_factory import EmbeddingFactory
from bruno_core.models import Message, MessageRole

class RAGSystem:
    """Complete RAG system with embeddings and LLM."""

    def __init__(self, llm_provider: str = "openai", embedding_provider: str = "openai"):
        self.llm = LLMFactory.create_from_env(llm_provider)
        self.embedder = EmbeddingFactory.create_from_env(embedding_provider)
        self.knowledge_base = []  # (text, embedding) pairs

    async def add_knowledge(self, documents: List[str]):
        """Add documents to the knowledge base."""
        print(f"Processing {len(documents)} documents...")

        # Generate embeddings in batches
        embeddings = await self.embedder.embed_texts(documents)

        # Store documents and embeddings
        for doc, embedding in zip(documents, embeddings):
            self.knowledge_base.append((doc, embedding))

        print(f"Knowledge base now contains {len(self.knowledge_base)} documents")

    async def retrieve_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant documents for a query."""
        if not self.knowledge_base:
            return []

        # Get query embedding
        query_embedding = await self.embedder.embed_text(query)

        # Calculate similarities
        similarities = []
        for doc, doc_embedding in self.knowledge_base:
            similarity = self.embedder.calculate_similarity(query_embedding, doc_embedding)
            similarities.append((doc, similarity))

        # Sort and return top documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:top_k]]

    async def answer_question(self, question: str) -> str:
        """Answer a question using RAG."""
        # Retrieve relevant context
        context_docs = await self.retrieve_relevant(question, top_k=3)

        if not context_docs:
            return "I don't have relevant information to answer this question."

        # Build context
        context = "\n\n".join(context_docs)

        # Generate answer
        messages = [
            Message(role=MessageRole.SYSTEM, content=
                "Answer the question based on the provided context. "
                "If the context doesn't contain enough information, say so clearly."),
            Message(role=MessageRole.USER, content=
                f"Context:\n{context}\n\nQuestion: {question}")
        ]

        response = await self.llm.generate(messages, temperature=0.1)
        return response

# Usage
rag = RAGSystem(llm_provider="openai", embedding_provider="openai")

# Build knowledge base
knowledge = [
    "Bruno-LLM is a Python package for LLM provider integration.",
    "It supports OpenAI, Ollama, and other providers through a unified interface.",
    "The factory pattern makes it easy to switch between providers.",
    "Embedding providers enable semantic search and RAG applications.",
    "All providers implement async interfaces for high performance."
]

await rag.add_knowledge(knowledge)

# Ask questions
answer = await rag.answer_question("What is Bruno-LLM and what does it support?")
print(f"Answer: {answer}")
```

## Performance Optimization

### Batch Processing

```python
async def efficient_batch_processing(embedder, texts: List[str], batch_size: int = 32):
    """Process large text collections efficiently."""

    total_texts = len(texts)
    all_embeddings = []

    print(f"Processing {total_texts} texts in batches of {batch_size}...")

    for i in range(0, total_texts, batch_size):
        batch = texts[i:i + batch_size]

        try:
            # Process batch
            batch_embeddings = await embedder.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

            # Progress update
            processed = len(all_embeddings)
            progress = processed / total_texts * 100
            print(f"Progress: {progress:.1f}% ({processed}/{total_texts})")

            # Rate limiting for API providers
            if hasattr(embedder, 'api_key'):  # Likely an API provider
                await asyncio.sleep(0.1)  # Small delay

        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # Add empty embeddings to maintain alignment
            all_embeddings.extend([None] * len(batch))

    print(f"✅ Completed processing {len(all_embeddings)} embeddings")
    return all_embeddings
```

### Caching for Performance

```python
import hashlib
import json
import os
from pathlib import Path

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding from cache."""
        cache_key = self._get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def store_embedding(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        cache_key = self._get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump(embedding, f)

    def clear_cache(self):
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

class CachedEmbedder:
    """Embedder with caching support."""

    def __init__(self, embedder, cache_dir: str = "embedding_cache"):
        self.embedder = embedder
        self.cache = EmbeddingCache(cache_dir)
        self.model = getattr(embedder, 'model', 'unknown')

    async def embed_text(self, text: str) -> List[float]:
        """Get embedding with caching."""
        # Check cache first
        cached = self.cache.get_embedding(text, self.model)
        if cached is not None:
            return cached

        # Generate new embedding
        embedding = await self.embedder.embed_text(text)

        # Store in cache
        self.cache.store_embedding(text, self.model, embedding)

        return embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get_embedding(text, self.model)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            print(f"Generating {len(uncached_texts)} new embeddings...")
            new_embeddings = await self.embedder.embed_texts(uncached_texts)

            # Store new embeddings and fill placeholders
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self.cache.store_embedding(texts[idx], self.model, embedding)
                embeddings[idx] = embedding

        return embeddings

# Usage
base_embedder = EmbeddingFactory.create_from_env("openai")
cached_embedder = CachedEmbedder(base_embedder)

# First call generates embeddings
embeddings1 = await cached_embedder.embed_texts(["Hello world", "AI is amazing"])

# Second call uses cache
embeddings2 = await cached_embedder.embed_texts(["Hello world", "New text"])  # Only "New text" computed
```

### Memory-Efficient Processing

```python
async def memory_efficient_embeddings(embedder, texts: List[str], max_memory_mb: int = 1000):
    """Process embeddings while managing memory usage."""

    import sys

    def estimate_memory_usage(num_embeddings: int, embedding_dim: int) -> int:
        """Estimate memory usage in MB."""
        # Each float is 4 bytes
        bytes_per_embedding = embedding_dim * 4
        total_bytes = num_embeddings * bytes_per_embedding
        return total_bytes / (1024 * 1024)  # Convert to MB

    # Get embedding dimension from a sample
    sample_embedding = await embedder.embed_text("sample")
    embedding_dim = len(sample_embedding)

    # Calculate optimal batch size
    max_embeddings_in_memory = int(max_memory_mb / (embedding_dim * 4 / 1024 / 1024))
    batch_size = min(len(texts), max_embeddings_in_memory)

    print(f"Processing with batch size: {batch_size}")
    print(f"Estimated memory per batch: {estimate_memory_usage(batch_size, embedding_dim):.1f} MB")

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Process batch
        batch_embeddings = await embedder.embed_texts(batch)

        # Store results (could write to disk for very large datasets)
        all_embeddings.extend(batch_embeddings)

        # Optional: Force garbage collection
        import gc
        gc.collect()

        print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    return all_embeddings
```

## Error Handling

### Robust Embedding Generation

```python
from bruno_llm.exceptions import LLMError, LLMTimeoutError
import asyncio

async def robust_embed_texts(embedder, texts: List[str], max_retries: int = 3):
    """Generate embeddings with error handling and retries."""

    results = []

    for i, text in enumerate(texts):
        success = False

        for attempt in range(max_retries):
            try:
                embedding = await embedder.embed_text(text)
                results.append(embedding)
                success = True
                break

            except LLMTimeoutError:
                print(f"Timeout for text {i}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except LLMError as e:
                print(f"Error for text {i}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    # Add placeholder for failed embedding
                    results.append(None)
                    success = True  # Don't retry further
                    break

        if not success:
            results.append(None)  # Final fallback

    # Count successful embeddings
    successful = sum(1 for r in results if r is not None)
    print(f"Successfully generated {successful}/{len(texts)} embeddings")

    return results
```

## Integration Patterns

### Vector Database Integration

```python
# Example with a simple in-memory vector store
import numpy as np
from typing import Dict, Any

class SimpleVectorStore:
    """Simple vector database for embeddings."""

    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.ids = []
        self._id_counter = 0

    def add_vectors(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add vectors with metadata."""
        for embedding, meta in zip(embeddings, metadata):
            self.vectors.append(np.array(embedding))
            self.metadata.append(meta)
            self.ids.append(self._id_counter)
            self._id_counter += 1

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        query_array = np.array(query_vector)
        similarities = []

        for i, vector in enumerate(self.vectors):
            # Cosine similarity
            similarity = np.dot(query_array, vector) / (np.linalg.norm(query_array) * np.linalg.norm(vector))
            similarities.append((self.ids[i], similarity, self.metadata[i]))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage with embeddings
async def build_vector_database():
    """Build and query a vector database."""

    embedder = EmbeddingFactory.create_from_env("openai")
    vector_store = SimpleVectorStore()

    # Sample documents
    documents = [
        {"text": "Python is a programming language", "category": "programming"},
        {"text": "Machine learning uses algorithms", "category": "AI"},
        {"text": "Databases store structured data", "category": "data"},
        {"text": "APIs connect different systems", "category": "programming"},
        {"text": "Neural networks learn patterns", "category": "AI"}
    ]

    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = await embedder.embed_texts(texts)

    # Add to vector store
    vector_store.add_vectors(embeddings, documents)

    # Query the database
    query = "What are programming concepts?"
    query_embedding = await embedder.embed_text(query)

    results = vector_store.search(query_embedding, top_k=3)

    print(f"Query: {query}")
    print("Results:")
    for doc_id, similarity, metadata in results:
        print(f"  {similarity:.3f}: {metadata['text']} (category: {metadata['category']})")

await build_vector_database()
```

## Best Practices

### 1. Provider Selection

```python
def choose_embedding_provider(use_case: str, budget: str, privacy: str):
    """Choose the best embedding provider for your needs."""

    if privacy == "high" or budget == "free":
        return "ollama"  # Local, private, free
    elif budget == "low":
        return "openai", "text-embedding-3-small"  # Cheap OpenAI option
    elif use_case == "high_accuracy":
        return "openai", "text-embedding-3-large"  # Best performance
    else:
        return "openai", "text-embedding-ada-002"  # Balanced choice

provider_name, model = choose_embedding_provider(
    use_case="general",
    budget="medium",
    privacy="medium"
)
```

### 2. Text Preprocessing

```python
def preprocess_text_for_embeddings(text: str) -> str:
    """Prepare text for optimal embedding generation."""

    # Remove excessive whitespace
    text = " ".join(text.split())

    # Truncate if too long (model-dependent)
    max_length = 8000  # Adjust based on model
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary

    # Optional: Remove special characters, normalize case, etc.
    # Be careful not to lose important semantic information

    return text.strip()
```

### 3. Monitoring and Metrics

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingMetrics:
    """Track embedding generation metrics."""
    total_texts: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    total_time: float = 0.0
    total_tokens: int = 0

    @property
    def success_rate(self) -> float:
        return self.successful_embeddings / self.total_texts if self.total_texts > 0 else 0.0

    @property
    def average_time_per_embedding(self) -> float:
        return self.total_time / self.successful_embeddings if self.successful_embeddings > 0 else 0.0

class MetricsTracker:
    """Track embedding metrics."""

    def __init__(self):
        self.metrics = EmbeddingMetrics()

    async def track_embedding(self, embedder, text: str) -> Optional[List[float]]:
        """Generate embedding and track metrics."""
        self.metrics.total_texts += 1
        start_time = time.time()

        try:
            embedding = await embedder.embed_text(text)

            self.metrics.successful_embeddings += 1
            self.metrics.total_time += time.time() - start_time
            self.metrics.total_tokens += len(text.split())  # Rough estimate

            return embedding

        except Exception as e:
            self.metrics.failed_embeddings += 1
            print(f"Embedding failed: {e}")
            return None

    def print_summary(self):
        """Print metrics summary."""
        print(f"📊 Embedding Metrics Summary:")
        print(f"   Total texts: {self.metrics.total_texts}")
        print(f"   Success rate: {self.metrics.success_rate:.1%}")
        print(f"   Average time: {self.metrics.average_time_per_embedding:.3f}s per embedding")
        print(f"   Total time: {self.metrics.total_time:.2f}s")

# Usage
tracker = MetricsTracker()
embedder = EmbeddingFactory.create_from_env("openai")

texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
for text in texts:
    embedding = await tracker.track_embedding(embedder, text)

tracker.print_summary()
```

For production deployments and advanced integration patterns, see the [Integration Examples](../examples/) and [Deployment Guide](../deployment/).
