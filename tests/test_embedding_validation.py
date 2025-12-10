"""
Basic validation test for embedding interface implementation.

This test verifies that our EmbeddingInterface implementation:
1. Properly implements the bruno-core interface
2. Uses numpy for efficient vector operations
3. Factory pattern works correctly
"""


import numpy as np
import pytest

from bruno_core.interfaces import EmbeddingInterface
from bruno_core.models import Message, MessageRole
from bruno_llm.base.embedding_interface import BaseEmbeddingProvider
from bruno_llm.embedding_factory import EmbeddingFactory
from bruno_llm.exceptions import ProviderNotFoundError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, model: str = "mock-model", dimension: int = 768, **kwargs):
        super().__init__(model, **kwargs)
        self._dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        """Return a mock embedding based on text length."""
        # Create a deterministic embedding based on text
        embedding = np.random.RandomState(len(text)).normal(0, 1, self._dimension)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return mock dimension."""
        return self._dimension

    async def check_connection(self) -> bool:
        """Mock connection check."""
        return True


class TestBaseEmbeddingProvider:
    """Test BaseEmbeddingProvider functionality."""

    def test_implements_interface(self):
        """Test that BaseEmbeddingProvider implements EmbeddingInterface."""
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingInterface)

    @pytest.mark.asyncio
    async def test_embed_text(self):
        """Test single text embedding."""
        provider = MockEmbeddingProvider(dimension=384)

        embedding = await provider.embed_text("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_texts(self):
        """Test batch text embedding."""
        provider = MockEmbeddingProvider(dimension=256)

        texts = ["Hello", "world", "test"]
        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 256 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_message(self):
        """Test message embedding."""
        provider = MockEmbeddingProvider(dimension=128)

        message = Message(role=MessageRole.USER, content="Test message")
        embedding = await provider.embed_message(message)

        assert len(embedding) == 128

    def test_calculate_similarity(self):
        """Test similarity calculation using numpy."""
        provider = MockEmbeddingProvider()

        # Test identical vectors
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = provider.calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10

        # Test orthogonal vectors
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = provider.calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10

    def test_batch_cosine_similarity(self):
        """Test batch similarity calculation."""
        provider = MockEmbeddingProvider()

        embeddings1 = [[1.0, 0.0], [0.0, 1.0]]
        embeddings2 = [[1.0, 0.0], [0.0, 1.0]]

        similarity_matrix = provider.batch_cosine_similarity(embeddings1, embeddings2)

        assert similarity_matrix.shape == (2, 2)
        # Should be identity matrix
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(similarity_matrix, expected, atol=1e-10)

    def test_find_most_similar(self):
        """Test finding most similar embeddings."""
        provider = MockEmbeddingProvider()

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]

        results = provider.find_most_similar(query, candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # Most similar is index 0
        assert abs(results[0][1] - 1.0) < 1e-10

    def test_average_embeddings(self):
        """Test embedding averaging using numpy."""
        provider = MockEmbeddingProvider()

        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        average = provider.average_embeddings(embeddings)

        expected = [2.0, 3.0]  # (1+3)/2, (2+4)/2
        assert all(abs(a - b) < 1e-10 for a, b in zip(average, expected))

    def test_validate_embedding(self):
        """Test embedding validation."""
        provider = MockEmbeddingProvider(dimension=3)

        # Valid embedding
        provider.validate_embedding([1.0, 2.0, 3.0])

        # Invalid dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            provider.validate_embedding([1.0, 2.0])

        # NaN values
        with pytest.raises(ValueError, match="NaN values"):
            provider.validate_embedding([1.0, float("nan"), 3.0])


class TestEmbeddingFactory:
    """Test EmbeddingFactory functionality."""

    def setup_method(self):
        """Clear factory before each test."""
        EmbeddingFactory._providers.clear()

    def test_register_provider(self):
        """Test provider registration."""
        EmbeddingFactory.register("test", MockEmbeddingProvider)

        assert "test" in EmbeddingFactory.list_providers()
        assert EmbeddingFactory.is_registered("test")

    def test_create_provider(self):
        """Test provider creation."""
        EmbeddingFactory.register("test", MockEmbeddingProvider)

        provider = EmbeddingFactory.create("test", {"model": "test-model"})

        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.model == "test-model"

    def test_create_nonexistent_provider(self):
        """Test creating non-existent provider."""
        with pytest.raises(ProviderNotFoundError):
            EmbeddingFactory.create("nonexistent")

    def test_get_provider_info(self):
        """Test getting provider information."""
        EmbeddingFactory.register("test", MockEmbeddingProvider)

        info = EmbeddingFactory.get_provider_info("test")

        assert info["name"] == "test"
        assert info["class_name"] == "MockEmbeddingProvider"

    def test_list_providers(self):
        """Test listing providers."""
        EmbeddingFactory.register("test1", MockEmbeddingProvider)
        EmbeddingFactory.register("test2", MockEmbeddingProvider)

        providers = EmbeddingFactory.list_providers()

        assert set(providers) == {"test1", "test2"}


if __name__ == "__main__":
    # Run a quick validation
    import asyncio

    async def main():
        print("🧪 Running basic validation tests...")

        # Test 1: Interface implementation
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingInterface)
        print("✅ Interface implementation: PASS")

        # Test 2: Embedding generation
        embedding = await provider.embed_text("Hello world")
        assert len(embedding) == provider.get_dimension()
        print("✅ Embedding generation: PASS")

        # Test 3: Similarity calculation (using numpy)
        similarity = provider.calculate_similarity(embedding, embedding)
        assert abs(similarity - 1.0) < 1e-10
        print("✅ Similarity calculation: PASS")

        # Test 4: Factory registration
        EmbeddingFactory.register("test", MockEmbeddingProvider)
        factory_provider = EmbeddingFactory.create("test")
        assert isinstance(factory_provider, EmbeddingInterface)
        print("✅ Factory pattern: PASS")

        print("\n🎉 All validation tests passed!")
        print("✅ EmbeddingInterface implementation is working correctly")
        print("✅ Using numpy for efficient vector operations")
        print("✅ Factory pattern integrated successfully")

    asyncio.run(main())
