# Quick Start

## Basic Usage

### 1. Create a Provider

```python
from bruno_llm import LLMFactory

# Ollama (local)
llm = LLMFactory.create("ollama", {
    "model": "llama2",
    "base_url": "http://localhost:11434"
})

# OpenAI (cloud)
llm = LLMFactory.create("openai", {
    "api_key": "sk-...",
    "model": "gpt-4"
})
```

### 2. Generate Responses

```python
from bruno_core.models import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="What is Python?")
]

# Generate complete response
response = await llm.generate(messages)
print(response)
```

### 3. Stream Responses

```python
# Stream response tokens
async for chunk in llm.stream(messages):
    print(chunk, end="", flush=True)
```

### 4. Generate Embeddings

```python
from bruno_llm.embedding_factory import EmbeddingFactory

# Create embedding provider
embedder = EmbeddingFactory.create("openai", {
    "api_key": "sk-...",
    "model": "text-embedding-3-small"
})

# Generate embedding
text = "Machine learning transforms data into insights"
embedding = await embedder.embed_text(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
texts = ["AI is powerful", "Python is versatile", "Data drives decisions"]
embeddings = await embedder.embed_texts(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### 5. Similarity Search

```python
# Calculate similarity between texts
query = "artificial intelligence"
doc1 = "Machine learning algorithms"
doc2 = "Cooking recipes"

query_emb = await embedder.embed_text(query)
doc1_emb = await embedder.embed_text(doc1)
doc2_emb = await embedder.embed_text(doc2)

similarity1 = embedder.calculate_similarity(query_emb, doc1_emb)
similarity2 = embedder.calculate_similarity(query_emb, doc2_emb)

print(f"Query-Doc1 similarity: {similarity1:.3f}")  # Higher
print(f"Query-Doc2 similarity: {similarity2:.3f}")  # Lower
```

## Complete Examples

### LLM Example

```python
import asyncio
from bruno_core.models import Message, MessageRole
from bruno_llm import LLMFactory

async def main():
    # Create provider
    llm = LLMFactory.create("ollama", {"model": "llama2"})

    # Prepare messages
    messages = [
        Message(role=MessageRole.USER, content="Tell me a joke")
    ]

    # Generate response
    response = await llm.generate(messages)
    print(f"Response: {response}")

    # Get token count
    tokens = llm.get_token_count(response)
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Embedding Example

```python
import asyncio
from bruno_llm.embedding_factory import EmbeddingFactory

async def embedding_demo():
    # Create embedding provider (local with Ollama)
    embedder = EmbeddingFactory.create("ollama", {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434"
    })

    # Sample documents
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Databases store information",
        "APIs connect systems"
    ]

    # Generate embeddings
    embeddings = await embedder.embed_texts(documents)

    # Search for similar content
    query = "programming languages"
    query_embedding = await embedder.embed_text(query)

    # Find most similar document
    similarities = []
    for i, doc_embedding in enumerate(embeddings):
        similarity = embedder.calculate_similarity(query_embedding, doc_embedding)
        similarities.append((documents[i], similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("Most similar documents:")
    for doc, score in similarities[:3]:
        print(f"  {score:.3f}: {doc}")

if __name__ == "__main__":
    asyncio.run(embedding_demo())
```

### Combined RAG Example

```python
import asyncio
from bruno_llm.factory import LLMFactory
from bruno_llm.embedding_factory import EmbeddingFactory
from bruno_core.models import Message, MessageRole

async def simple_rag_demo():
    # Create providers
    llm = LLMFactory.create_from_env("openai")
    embedder = EmbeddingFactory.create_from_env("openai")

    # Knowledge base
    knowledge = [
        "Bruno-LLM is a Python library for LLM integration",
        "It supports multiple providers like OpenAI and Ollama",
        "The factory pattern makes switching providers easy",
        "Embeddings enable semantic search capabilities"
    ]

    # Generate embeddings for knowledge
    knowledge_embeddings = await embedder.embed_texts(knowledge)

    # User question
    question = "What is Bruno-LLM?"
    question_embedding = await embedder.embed_text(question)

    # Find relevant knowledge
    similarities = []
    for i, kb_embedding in enumerate(knowledge_embeddings):
        similarity = embedder.calculate_similarity(question_embedding, kb_embedding)
        similarities.append((knowledge[i], similarity))

    # Get top 2 most relevant
    similarities.sort(key=lambda x: x[1], reverse=True)
    context = "\n".join([doc for doc, _ in similarities[:2]])

    # Generate answer with context
    messages = [
        Message(role=MessageRole.SYSTEM, content=
            "Answer the question based on the provided context."),
        Message(role=MessageRole.USER, content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    answer = await llm.generate(messages)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(simple_rag_demo())
```

## Next Steps

- [User Guide](../user-guide/overview.md) - Complete documentation
- [Ollama Provider](../user-guide/providers/ollama.md) - Local LLM setup
- [OpenAI Provider](../user-guide/providers/openai.md) - Cloud LLM setup
- [Advanced Features](../user-guide/advanced/caching.md) - Caching, streaming, etc.
