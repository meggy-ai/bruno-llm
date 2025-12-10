"""
bruno-llm: LLM and Embedding provider implementations for bruno-core.

This package provides production-ready LLM and Embedding provider implementations
that integrate seamlessly with the bruno-core framework. All providers implement
the respective interface contracts from bruno-core.

Available LLM Providers:
    - OllamaProvider: Local LLM inference via Ollama
    - OpenAIProvider: OpenAI GPT models

Available Embedding Providers:
    - OpenAIEmbeddingProvider: OpenAI text embeddings (planned)
    - HuggingFaceEmbeddingProvider: HuggingFace sentence transformers (planned)
    - OllamaEmbeddingProvider: Local embeddings via Ollama (planned)

Example Usage:
    >>> # LLM Provider
    >>> from bruno_llm.providers.ollama import OllamaProvider
    >>> llm = OllamaProvider(model="llama2")
    >>> response = await llm.generate([Message(role="user", content="Hello")])

    >>> # Embedding Provider (when available)
    >>> from bruno_llm.embedding_factory import EmbeddingFactory
    >>> embedder = EmbeddingFactory.create("openai", {"api_key": "sk-..."})
    >>> embedding = await embedder.embed_text("Hello world")
"""

from bruno_llm.__version__ import (
    __author__,
    __description__,
    __email__,
    __license__,
    __version__,
)

# Public API will be populated as providers are implemented
from bruno_llm.factory import LLMFactory

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "LLMFactory",
]
