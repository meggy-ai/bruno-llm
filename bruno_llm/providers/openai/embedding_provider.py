"""
OpenAI Embedding Provider implementation.

This module provides an embedding provider for OpenAI's text embedding models
using the official OpenAI Python library (external library first policy).
"""

import logging
from typing import Optional, Union

import httpx
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from bruno_llm.base.embedding_interface import BaseEmbeddingProvider
from bruno_llm.exceptions import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using the official OpenAI Python library.

    This provider supports OpenAI's text embedding models including:
    - text-embedding-ada-002 (1536 dimensions)
    - text-embedding-3-small (1536 dimensions, configurable)
    - text-embedding-3-large (3072 dimensions, configurable)

    Uses the official OpenAI library for reliable API communication and follows
    the external library first policy.

    Args:
        api_key: OpenAI API key
        model: Model name (default: "text-embedding-ada-002")
        organization: OpenAI organization ID (optional)
        dimensions: Embedding dimensions for v3 models (optional)
        batch_size: Maximum batch size for embedding requests (default: 100)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retries (default: 3)

    Examples:
        >>> provider = OpenAIEmbeddingProvider(
        ...     api_key="sk-...",
        ...     model="text-embedding-ada-002"
        ... )
        >>> embedding = await provider.embed_text("Hello world")
        >>> len(embedding)  # 1536

        >>> # Use newer v3 models with custom dimensions
        >>> provider = OpenAIEmbeddingProvider(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small",
        ...     dimensions=512
        ... )
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,  # Default, configurable
        "text-embedding-3-large": 3072,  # Default, configurable
    }

    # Model batch size limits
    MODEL_BATCH_LIMITS = {
        "text-embedding-ada-002": 2048,
        "text-embedding-3-small": 2048,
        "text-embedding-3-large": 2048,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        organization: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(model=model, timeout=timeout, **kwargs)

        if not api_key:
            raise ConfigurationError("OpenAI API key is required")

        self.api_key = api_key
        self.organization = organization
        self.batch_size = min(batch_size, self.MODEL_BATCH_LIMITS.get(model, 100))
        self.max_retries = max_retries

        # Handle dimensions for v3 models
        if dimensions is not None:
            if model in ["text-embedding-3-small", "text-embedding-3-large"]:
                self._dimensions = dimensions
            else:
                logger.warning(f"Dimensions parameter not supported for model {model}, ignoring")
                self._dimensions = self.MODEL_DIMENSIONS.get(model, 1536)
        else:
            self._dimensions = self.MODEL_DIMENSIONS.get(model, 1536)

        # Initialize OpenAI client (using external library)
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text using OpenAI API.

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
            # Use official OpenAI library (external library)
            kwargs = {"input": text, "model": self.model}

            # Add dimensions for v3 models
            if self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
                kwargs["dimensions"] = self._dimensions

            response: CreateEmbeddingResponse = await self.client.embeddings.create(**kwargs)

            if not response.data:
                raise InvalidResponseError("Empty response from OpenAI API")

            embedding = response.data[0].embedding

            # Validate embedding using numpy (from base class)
            self.validate_embedding(embedding)

            return embedding

        except Exception as e:
            await self._handle_api_error(e)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using batch API.

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

        # Process in batches
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
        """Embed a batch of texts using OpenAI API."""
        try:
            kwargs = {"input": texts, "model": self.model}

            # Add dimensions for v3 models
            if self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
                kwargs["dimensions"] = self._dimensions

            response: CreateEmbeddingResponse = await self.client.embeddings.create(**kwargs)

            if len(response.data) != len(texts):
                raise InvalidResponseError(
                    f"Expected {len(texts)} embeddings, got {len(response.data)}"
                )

            # Sort by index to maintain order
            embeddings: list[Union[list[float], None]] = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding

            # Validate all embeddings
            for embedding in embeddings:
                if embedding is None:
                    raise InvalidResponseError("Missing embedding in response")
                self.validate_embedding(embedding)

            return embeddings

        except Exception as e:
            await self._handle_api_error(e)

    def get_dimension(self) -> int:
        """
        Get the embedding dimension for this provider.

        Returns:
            Number of dimensions in embeddings
        """
        return self._dimensions

    async def check_connection(self) -> bool:
        """
        Check if the OpenAI API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Test with a simple embedding request
            await self.embed_text("test")
            return True
        except Exception:
            return False

    async def _handle_api_error(self, error: Exception) -> None:
        """Handle and convert OpenAI API errors to bruno-llm exceptions."""
        # Import here to avoid dependency if OpenAI not installed
        try:
            from openai import (
                APIError as OpenAIAPIError,
            )
            from openai import (
                AuthenticationError as OpenAIAuthError,
            )
            from openai import (
                NotFoundError as OpenAINotFound,
            )
            from openai import (
                RateLimitError as OpenAIRateLimit,
            )
        except ImportError:
            raise LLMError(f"OpenAI API error: {error}", provider="openai") from error

        if isinstance(error, OpenAIAuthError):
            raise AuthenticationError(
                "Invalid OpenAI API key or insufficient permissions",
                provider="openai",
                original_error=error,
            ) from error
        elif isinstance(error, OpenAIRateLimit):
            retry_after = getattr(error.response, "headers", {}).get("retry-after")
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {error}",
                provider="openai",
                original_error=error,
                retry_after=int(retry_after) if retry_after else None,
            ) from error
        elif isinstance(error, OpenAINotFound):
            raise ModelNotFoundError(
                f"OpenAI model '{self.model}' not found: {error}",
                provider="openai",
                original_error=error,
            ) from error
        elif isinstance(error, OpenAIAPIError):
            raise InvalidResponseError(
                f"OpenAI API error: {error}", provider="openai", original_error=error
            ) from error
        elif isinstance(error, httpx.TimeoutException):
            raise LLMError(
                f"Request timeout after {self.timeout}s", provider="openai", original_error=error
            ) from error
        else:
            raise LLMError(
                f"Unexpected error: {error}", provider="openai", original_error=error
            ) from error

    @classmethod
    def from_env(cls, **kwargs) -> "OpenAIEmbeddingProvider":
        """
        Create provider from environment variables.

        Expected environment variables:
        - OPENAI_API_KEY (required)
        - OPENAI_ORG_ID (optional)
        - OPENAI_EMBEDDING_MODEL (optional, default: text-embedding-ada-002)
        - OPENAI_EMBEDDING_DIMENSIONS (optional, for v3 models)
        - EMBEDDING_TIMEOUT (optional, default: 30.0)
        - EMBEDDING_BATCH_SIZE (optional, default: 100)

        Returns:
            Configured OpenAIEmbeddingProvider instance

        Raises:
            ConfigurationError: If required environment variables missing
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable is required")

        config = {
            "api_key": api_key,
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            "organization": os.getenv("OPENAI_ORG_ID"),
        }

        # Handle dimensions
        if dimensions_str := os.getenv("OPENAI_EMBEDDING_DIMENSIONS"):
            try:
                config["dimensions"] = int(dimensions_str)
            except ValueError:
                logger.warning(f"Invalid OPENAI_EMBEDDING_DIMENSIONS: {dimensions_str}")

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
