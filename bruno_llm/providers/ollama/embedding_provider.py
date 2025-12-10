"""
Ollama Embedding Provider implementation.

This module provides an embedding provider for Ollama's local embedding models
using the existing Ollama infrastructure and following external library first policy.
"""

import json
import logging

import httpx

from bruno_llm.base.embedding_interface import BaseEmbeddingProvider
from bruno_llm.exceptions import (
    InvalidResponseError,
    LLMError,
    ModelNotFoundError,
)
from bruno_llm.exceptions import (
    TimeoutError as LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama embedding provider for local embedding model inference.

    This provider supports Ollama's local embedding models including:
    - nomic-embed-text
    - mxbai-embed-large
    - snowflake-arctic-embed
    - all-minilm:l6-v2

    Uses the existing Ollama API infrastructure for local embedding generation,
    following the external library first policy by leveraging Ollama's proven
    embedding capabilities.

    Args:
        base_url: Ollama API endpoint (default: http://localhost:11434)
        model: Embedding model name (default: "nomic-embed-text")
        timeout: Request timeout in seconds (default: 30.0)
        batch_size: Maximum batch size for embedding requests (default: 32)

    Examples:
        >>> provider = OllamaEmbeddingProvider(
        ...     model="nomic-embed-text"
        ... )
        >>> embedding = await provider.embed_text("Hello world")
        >>> len(embedding)  # Model-dependent dimension

        >>> # Use with different model
        >>> provider = OllamaEmbeddingProvider(
        ...     base_url="http://192.168.1.100:11434",
        ...     model="mxbai-embed-large"
        ... )
    """

    # Known embedding models and their dimensions
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed": 1024,
        "all-minilm:l6-v2": 384,
        "all-minilm": 384,  # Alias
        "nomic-embed": 768,  # Alias
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(model=model, timeout=timeout, **kwargs)

        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

        # Estimate dimensions from known models
        self._dimensions = self.MODEL_DIMENSIONS.get(model, 768)  # Default to 768

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text using Ollama API.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            LLMError: If embedding generation fails
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimensions

        try:
            response = await self._make_request({"input": text})

            if "embedding" not in response:
                raise InvalidResponseError("No embedding in Ollama response")

            embedding = response["embedding"]

            # Validate and update dimensions
            if len(embedding) != self._dimensions:
                logger.info(
                    f"Updating model dimensions from {self._dimensions} to {len(embedding)}"
                )
                self._dimensions = len(embedding)

            # Validate embedding using base class
            self.validate_embedding(embedding)

            return embedding

        except Exception as e:
            if isinstance(e, (LLMError, InvalidResponseError)):
                raise
            raise LLMError(f"Ollama embedding failed: {e}", provider="ollama") from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            LLMError: If any embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts and track indices
        non_empty_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                text_indices.append(i)

        if not non_empty_texts:
            # All texts are empty, return zero vectors
            return [[0.0] * self._dimensions] * len(texts)

        # Process in batches (Ollama handles batching internally, but we'll do smaller chunks)
        all_embeddings = []
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch = non_empty_texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        # Reconstruct full results with zero vectors for empty texts
        results = []
        embedding_idx = 0

        for _i, text in enumerate(texts):
            if text.strip():
                results.append(all_embeddings[embedding_idx])
                embedding_idx += 1
            else:
                results.append([0.0] * self._dimensions)

        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Ollama API."""
        try:
            # Ollama embedding API can handle multiple inputs
            response = await self._make_request({"input": texts})

            if "embedding" in response:
                # Single embedding returned (older Ollama versions or single input)
                embeddings = [response["embedding"]]
            elif "embeddings" in response:
                # Multiple embeddings returned
                embeddings = response["embeddings"]
            else:
                raise InvalidResponseError("No embeddings in Ollama response")

            # Validate number of embeddings matches input
            if len(embeddings) != len(texts):
                raise InvalidResponseError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )

            # Validate all embeddings and update dimensions if needed
            for embedding in embeddings:
                if len(embedding) != self._dimensions:
                    logger.info(
                        f"Updating model dimensions from {self._dimensions} to {len(embedding)}"
                    )
                    self._dimensions = len(embedding)
                self.validate_embedding(embedding)

            return embeddings

        except Exception as e:
            if isinstance(e, (LLMError, InvalidResponseError)):
                raise
            raise LLMError(f"Ollama batch embedding failed: {e}", provider="ollama") from e

    async def _make_request(self, payload: dict) -> dict:
        """Make HTTP request to Ollama API."""
        request_data = {
            "model": self.model,
            **payload,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json=request_data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama request timeout after {self.timeout}s", provider="ollama"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Check if it's a model not found error
                try:
                    error_data = e.response.json()
                    if "model" in error_data.get("error", "").lower():
                        raise ModelNotFoundError(
                            f"Ollama model '{self.model}' not found. "
                            f"Please pull the model: ollama pull {self.model}",
                            provider="ollama",
                        ) from e
                except (json.JSONDecodeError, ValueError):
                    pass
                raise ModelNotFoundError(
                    f"Ollama model '{self.model}' not found or endpoint not available",
                    provider="ollama",
                ) from e
            elif e.response.status_code == 500:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", str(e))
                    raise LLMError(f"Ollama server error: {error_msg}", provider="ollama") from e
                except (json.JSONDecodeError, ValueError):
                    raise LLMError(f"Ollama server error: {e}", provider="ollama") from e
            else:
                raise LLMError(f"Ollama HTTP error: {e}", provider="ollama") from e
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to connect to Ollama at {self.base_url}: {e}", provider="ollama"
            ) from e

    def get_dimension(self) -> int:
        """
        Get the embedding dimension for this provider.

        Returns:
            Number of dimensions in embeddings
        """
        return self._dimensions

    async def check_connection(self) -> bool:
        """
        Check if the Ollama API is accessible and model is available.

        Returns:
            True if API is accessible and model available, False otherwise
        """
        try:
            # Test with a simple embedding request
            await self.embed_text("test")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @classmethod
    def from_env(cls, **kwargs) -> "OllamaEmbeddingProvider":
        """
        Create provider from environment variables.

        Expected environment variables:
        - OLLAMA_BASE_URL (optional, default: http://localhost:11434)
        - OLLAMA_EMBEDDING_MODEL (optional, default: nomic-embed-text)
        - EMBEDDING_TIMEOUT (optional, default: 30.0)
        - EMBEDDING_BATCH_SIZE (optional, default: 32)

        Returns:
            Configured OllamaEmbeddingProvider instance
        """
        import os

        config = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        }

        # Handle timeout
        if timeout_str := os.getenv("EMBEDDING_TIMEOUT"):
            try:
                config["timeout"] = float(timeout_str)
            except ValueError:
                logger.warning(f"Invalid EMBEDDING_TIMEOUT: {timeout_str}")

        # Handle batch size
        if batch_size_str := os.getenv("EMBEDDING_BATCH_SIZE"):
            try:
                config["batch_size"] = int(batch_size_str)
            except ValueError:
                logger.warning(f"Invalid EMBEDDING_BATCH_SIZE: {batch_size_str}")

        # Override with any provided kwargs
        config.update(kwargs)

        return cls(**config)
