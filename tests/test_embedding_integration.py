"""
Integration test for OpenAI embedding provider with EmbeddingFactory.

This test validates that the provider is properly registered and can be created through the factory.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from bruno_llm.embedding_factory import EmbeddingFactory
from bruno_llm.exceptions import ConfigurationError, ProviderNotFoundError
from bruno_llm.providers.openai.embedding_provider import OpenAIEmbeddingProvider


class TestEmbeddingFactoryIntegration:
    """Integration tests for OpenAI provider with EmbeddingFactory."""

    def test_openai_provider_registration(self):
        """Test that OpenAI provider is properly registered."""
        providers = EmbeddingFactory.list_providers()
        assert "openai" in providers

    def test_create_openai_provider_via_factory(self):
        """Test creating OpenAI provider through factory."""
        provider = EmbeddingFactory.create(
            "openai", config={"api_key": "test-key-123", "model": "text-embedding-ada-002"}
        )

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.api_key == "test-key-123"
        assert provider.model == "text-embedding-ada-002"

    def test_create_openai_provider_with_kwargs(self):
        """Test creating OpenAI provider with kwargs."""
        provider = EmbeddingFactory.create(
            "openai",
            api_key="test-key-456",
            model="text-embedding-3-small",
            dimensions=512,
            timeout=60.0,
        )

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.api_key == "test-key-456"
        assert provider.model == "text-embedding-3-small"
        assert provider.get_dimension() == 512
        assert provider.timeout == 60.0

    def test_create_openai_provider_from_env(self):
        """Test creating OpenAI provider from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "env-test-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
            "OPENAI_ORG_ID": "org-test",
            "EMBEDDING_TIMEOUT": "45.0",
        }

        with patch.dict(os.environ, env_vars):
            provider = EmbeddingFactory.create_from_env("openai")

            assert isinstance(provider, OpenAIEmbeddingProvider)
            assert provider.api_key == "env-test-key"
            assert provider.model == "text-embedding-3-large"
            assert provider.organization == "org-test"
            assert provider.timeout == 45.0

    def test_create_nonexistent_provider(self):
        """Test creating nonexistent provider raises error."""
        with pytest.raises(
            ProviderNotFoundError, match="Embedding provider 'nonexistent' not found"
        ):
            EmbeddingFactory.create("nonexistent")

    def test_create_openai_provider_missing_api_key(self):
        """Test creating OpenAI provider without API key raises error."""
        with pytest.raises(
            ConfigurationError, match="Failed to create embedding provider.*missing.*api_key"
        ):
            EmbeddingFactory.create("openai", config={})

    def test_create_from_env_missing_api_key(self):
        """Test creating from env without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY.*required"):
                EmbeddingFactory.create_from_env("openai")

    def test_get_provider_info(self):
        """Test getting provider information."""
        info = EmbeddingFactory.get_provider_info("openai")

        assert info["name"] == "openai"
        assert info["class_name"] == "OpenAIEmbeddingProvider"
        assert "openai.embedding_provider" in info["module"]
        assert len(info["docstring"]) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_embedding_with_mock(self):
        """Test complete embedding workflow with mocked API."""
        # Create provider via factory
        provider = EmbeddingFactory.create(
            "openai", api_key="test-key", model="text-embedding-ada-002"
        )

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512, index=0)  # 1536 dimensions
        ]

        with patch.object(provider.client.embeddings, "create", return_value=mock_response):
            # Test single embedding
            embedding = await provider.embed_text("Hello world")
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)

            # Test batch embedding
            batch_mock = MagicMock()
            batch_mock.data = [
                MagicMock(embedding=[0.1] * 1536, index=0),
                MagicMock(embedding=[0.2] * 1536, index=1),
            ]

            with patch.object(provider.client.embeddings, "create", return_value=batch_mock):
                embeddings = await provider.embed_texts(["Hello", "World"])
                assert len(embeddings) == 2
                assert all(len(emb) == 1536 for emb in embeddings)

                # Test similarity calculation using numpy from base class
                similarity = provider.calculate_similarity(embeddings[0], embeddings[1])
                assert isinstance(similarity, float)
                assert -1.0 <= similarity <= 1.0
