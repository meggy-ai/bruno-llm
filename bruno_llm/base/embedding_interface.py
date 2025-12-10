"""
Base embedding provider implementation using external libraries.

This module provides a base class for embedding providers that implements
the bruno-core EmbeddingInterface using numpy for efficient vector operations.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from bruno_core.interfaces import EmbeddingInterface
from bruno_core.models import Message


class BaseEmbeddingProvider(EmbeddingInterface, ABC):
    """
    Base class for embedding providers using numpy for vector operations.

    This base class implements the EmbeddingInterface from bruno-core and provides:
    - Efficient vector operations using numpy
    - Input validation and error handling
    - Similarity calculation using numpy's optimized functions
    - Batch processing utilities

    All embedding providers should inherit from this class and implement the
    abstract methods for their specific API or model.

    Args:
        model: Model name or identifier
        timeout: Request timeout in seconds

    Examples:
        >>> class MyEmbeddingProvider(BaseEmbeddingProvider):
        ...     async def embed_text(self, text: str) -> List[float]:
        ...         # Implementation specific to your provider
        ...         return await self._call_api(text)
        ...
        ...     def get_dimension(self) -> int:
        ...         return 768  # Your model's dimension
    """

    def __init__(
        self,
        model: str,
        timeout: float = 30.0,
        **kwargs,
    ):
        self.model = model
        self.timeout = timeout
        self.config = kwargs
        self._dimension_cache: Optional[int] = None

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            LLMError: If embedding generation fails
        """
        pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Default implementation calls embed_text for each text.
        Override for providers with native batch support.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            LLMError: If any embedding generation fails
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    async def embed_message(self, message: Message) -> list[float]:
        """
        Generate embedding for a message.

        Args:
            message: Message object to embed

        Returns:
            Embedding vector for the message content
        """
        return await self.embed_text(message.content)

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the embedding dimension for this provider.

        Returns:
            Number of dimensions in embeddings
        """
        pass

    def get_model_name(self) -> str:
        """
        Get the model name used by this provider.

        Returns:
            Model name string
        """
        return self.model

    def calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings using numpy.

        Uses numpy's optimized dot product and norm calculations for efficiency.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (-1.0 to 1.0)

        Raises:
            ValueError: If vectors have different dimensions
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)

        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} != {vec2.shape}")

        # Handle edge cases
        if vec1.size == 0 or vec2.size == 0:
            return 0.0

        # Calculate cosine similarity using numpy
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Handle zero vectors
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Ensure result is in valid range due to floating point precision
        return float(np.clip(similarity, -1.0, 1.0))

    @abstractmethod
    async def check_connection(self) -> bool:
        """
        Check if the provider is accessible.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    # Utility methods for common operations

    def validate_embedding(
        self, embedding: list[float], expected_dimension: Optional[int] = None
    ) -> None:
        """
        Validate an embedding vector using numpy.

        Args:
            embedding: Embedding to validate
            expected_dimension: Expected dimension (uses get_dimension() if None)

        Raises:
            ValueError: If embedding is invalid
        """
        if not isinstance(embedding, list):
            raise ValueError(f"Embedding must be a list, got {type(embedding)}")

        if len(embedding) == 0:
            raise ValueError("Embedding cannot be empty")

        # Convert to numpy for validation
        vec = np.array(embedding, dtype=np.float32)

        # Check for invalid values using numpy
        if np.any(np.isnan(vec)):
            raise ValueError("Embedding contains NaN values")

        if np.any(np.isinf(vec)):
            raise ValueError("Embedding contains infinite values")

        # Check dimension
        expected_dim = expected_dimension or self.get_dimension()
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, " f"got {len(embedding)}"
            )

    def batch_cosine_similarity(
        self, embeddings1: list[list[float]], embeddings2: list[list[float]]
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarities between two sets of embeddings.

        Uses numpy for efficient batch computation.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix as numpy array (len(embeddings1) x len(embeddings2))
        """
        # Convert to numpy arrays
        matrix1 = np.array(embeddings1, dtype=np.float32)
        matrix2 = np.array(embeddings2, dtype=np.float32)

        # Normalize vectors
        norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)

        # Handle zero vectors
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)

        normalized1 = matrix1 / norm1
        normalized2 = matrix2 / norm2

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(normalized1, normalized2.T)

        return similarity_matrix

    def find_most_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to a query using numpy.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results (None = all)

        Returns:
            List of (index, similarity) tuples sorted by similarity (descending)
        """
        if not candidate_embeddings:
            return []

        # Calculate similarities using numpy
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        candidates_matrix = np.array(candidate_embeddings, dtype=np.float32)

        # Use batch similarity calculation
        similarities = self.batch_cosine_similarity(query_vec.tolist(), candidates_matrix.tolist())[
            0
        ]  # Get first (and only) row

        # Create index-similarity pairs
        indexed_similarities = list(enumerate(similarities.tolist()))

        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        if top_k is not None:
            indexed_similarities = indexed_similarities[:top_k]

        return indexed_similarities

    def average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """
        Calculate average embedding using numpy.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Average embedding vector

        Raises:
            ValueError: If embeddings list is empty
        """
        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings")

        # Use numpy for efficient averaging
        matrix = np.array(embeddings, dtype=np.float32)
        average = np.mean(matrix, axis=0)

        return average.tolist()

    def weighted_average_embeddings(
        self,
        embeddings: list[list[float]],
        weights: list[float],
    ) -> list[float]:
        """
        Calculate weighted average embedding using numpy.

        Args:
            embeddings: List of embedding vectors
            weights: List of weights (must sum to 1.0)

        Returns:
            Weighted average embedding

        Raises:
            ValueError: If inputs are invalid
        """
        if len(embeddings) != len(weights):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of weights ({len(weights)})"
            )

        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings")

        # Validate weights sum using numpy
        weights_array = np.array(weights, dtype=np.float32)
        if not np.isclose(weights_array.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {weights_array.sum()}")

        # Calculate weighted average using numpy
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        weighted_avg = np.average(embeddings_matrix, axis=0, weights=weights_array)

        return weighted_avg.tolist()
