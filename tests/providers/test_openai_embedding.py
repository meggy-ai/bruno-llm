"""
Tests for OpenAI Embedding Provider.

This module tests the OpenAIEmbeddingProvider implementation using mocks
to avoid requiring real API keys during testing.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from bruno_llm.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelNotFoundError,
    RateLimitError,
)
from bruno_llm.providers.openai.config import OpenAIEmbeddingConfig
from bruno_llm.providers.openai.embedding_provider import OpenAIEmbeddingProvider


class TestOpenAIEmbeddingProvider:
    """Test suite for OpenAIEmbeddingProvider."""

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI embedding response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512, index=0),  # 1536 dimensions
        ]
        return mock_response

    @pytest.fixture
    def mock_openai_response_512(self):
        """Mock OpenAI embedding response with 512 dimensions."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 170 + [0.1, 0.2], index=0),  # 512 dimensions
        ]
        return mock_response

    @pytest.fixture
    def mock_openai_batch_response(self):
        """Mock OpenAI batch embedding response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512, index=0),
            MagicMock(embedding=[0.4, 0.5, 0.6] * 512, index=1),
            MagicMock(embedding=[0.7, 0.8, 0.9] * 512, index=2),
        ]
        return mock_response

    @pytest.fixture
    def provider(self):
        """Create test provider instance."""
        return OpenAIEmbeddingProvider(api_key="test-key-123", model="text-embedding-ada-002")

    def test_init_basic(self):
        """Test basic provider initialization."""
        provider = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-ada-002")

        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-ada-002"
        assert provider.get_dimension() == 1536
        assert provider.batch_size <= 2048
        assert provider.timeout == 30.0

    def test_init_v3_model_with_dimensions(self):
        """Test initialization with v3 model and custom dimensions."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small", dimensions=512
        )

        assert provider.model == "text-embedding-3-small"
        assert provider.get_dimension() == 512

    def test_init_invalid_dimensions_for_ada(self):
        """Test that dimensions are ignored for ada-002 model."""
        with patch("bruno_llm.providers.openai.embedding_provider.logger") as mock_logger:
            provider = OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-ada-002",
                dimensions=512,  # Should be ignored
            )

            assert provider.get_dimension() == 1536  # Default for ada-002
            mock_logger.warning.assert_called_once()

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ConfigurationError, match="OpenAI API key is required"):
            OpenAIEmbeddingProvider(api_key="")

    @pytest.mark.asyncio
    async def test_embed_text_success(self, provider, mock_openai_response):
        """Test successful single text embedding."""
        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            result = await provider.embed_text("Hello world")

            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)
            mock_create.assert_called_once_with(input="Hello world", model="text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, provider):
        """Test embedding empty string returns zero vector."""
        result = await provider.embed_text("")

        assert len(result) == 1536
        assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_v3_model(self, mock_openai_response_512):
        """Test embedding with v3 model includes dimensions parameter."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small", dimensions=512
        )

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response_512

            result = await provider.embed_text("Hello world")

            assert len(result) == 512
            mock_create.assert_called_once_with(
                input="Hello world", model="text-embedding-3-small", dimensions=512
            )

    @pytest.mark.asyncio
    async def test_embed_texts_batch_success(self, provider, mock_openai_batch_response):
        """Test successful batch embedding."""
        texts = ["Hello", "World", "Test"]

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_batch_response

            results = await provider.embed_texts(texts)

            assert len(results) == 3
            assert all(len(embedding) == 1536 for embedding in results)
            mock_create.assert_called_once_with(input=texts, model="text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, provider):
        """Test embedding empty list returns empty results."""
        result = await provider.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_with_empty_strings(self, provider, mock_openai_response):
        """Test embedding with some empty strings."""
        texts = ["Hello", "", "World"]

        # Mock response for only non-empty texts
        mock_openai_response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),  # "Hello"
            MagicMock(embedding=[0.2] * 1536, index=1),  # "World"
        ]

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            results = await provider.embed_texts(texts)

            assert len(results) == 3
            assert results[0] == [0.1] * 1536  # "Hello"
            assert results[1] == [0.0] * 1536  # Empty string -> zero vector
            assert results[2] == [0.2] * 1536  # "World"

            # Should only call API with non-empty texts
            mock_create.assert_called_once_with(
                input=["Hello", "World"], model="text-embedding-ada-002"
            )

    @pytest.mark.asyncio
    async def test_embed_texts_large_batch_splitting(self, provider):
        """Test that large batches are split correctly."""
        # Set small batch size for testing
        provider.batch_size = 2
        texts = ["text1", "text2", "text3", "text4", "text5"]

        # Mock responses for each batch
        mock_response1 = MagicMock()
        mock_response1.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
            MagicMock(embedding=[0.2] * 1536, index=1),
        ]

        mock_response2 = MagicMock()
        mock_response2.data = [
            MagicMock(embedding=[0.3] * 1536, index=0),
            MagicMock(embedding=[0.4] * 1536, index=1),
        ]

        mock_response3 = MagicMock()
        mock_response3.data = [
            MagicMock(embedding=[0.5] * 1536, index=0),
        ]

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [mock_response1, mock_response2, mock_response3]

            results = await provider.embed_texts(texts)

            assert len(results) == 5
            assert mock_create.call_count == 3

            # Check batch calls
            calls = mock_create.call_args_list
            assert calls[0][1]["input"] == ["text1", "text2"]
            assert calls[1][1]["input"] == ["text3", "text4"]
            assert calls[2][1]["input"] == ["text5"]

    @pytest.mark.asyncio
    async def test_check_connection_success(self, provider, mock_openai_response):
        """Test successful connection check."""
        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            result = await provider.check_connection()

            assert result is True
            mock_create.assert_called_once_with(input="test", model="text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_check_connection_failure(self, provider):
        """Test connection check failure."""
        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            result = await provider.check_connection()

            assert result is False

    @pytest.mark.asyncio
    async def test_error_handling_auth_error(self, provider):
        """Test authentication error handling."""
        # Import here to avoid dependency issues
        try:
            from openai import AuthenticationError as OpenAIAuthError
        except ImportError:
            pytest.skip("OpenAI library not available")

        # Create mock response and body for OpenAI error
        mock_response = MagicMock()
        mock_response.status_code = 401

        auth_error = OpenAIAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = auth_error

            with pytest.raises(AuthenticationError, match="Invalid OpenAI API key"):
                await provider.embed_text("test")

    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, provider):
        """Test rate limit error handling."""
        try:
            from openai import RateLimitError as OpenAIRateLimit
        except ImportError:
            pytest.skip("OpenAI library not available")

        # Mock response with retry-after header
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}

        rate_error = OpenAIRateLimit(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = rate_error

            with pytest.raises(RateLimitError, match="OpenAI rate limit exceeded"):
                await provider.embed_text("test")

    @pytest.mark.asyncio
    async def test_error_handling_model_not_found(self, provider):
        """Test model not found error handling."""
        try:
            from openai import NotFoundError as OpenAINotFound
        except ImportError:
            pytest.skip("OpenAI library not available")

        # Mock response for not found error
        mock_response = MagicMock()
        mock_response.status_code = 404

        not_found_error = OpenAINotFound(
            message="Model not found",
            response=mock_response,
            body={"error": {"message": "Model not found"}},
        )

        with patch.object(
            provider.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = not_found_error

            with pytest.raises(ModelNotFoundError, match="OpenAI model.*not found"):
                await provider.embed_text("test")

    def test_from_env_success(self):
        """Test creating provider from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_ORG_ID": "test-org",
            "OPENAI_EMBEDDING_DIMENSIONS": "512",
            "EMBEDDING_TIMEOUT": "60.0",
            "EMBEDDING_BATCH_SIZE": "200",
        }

        with patch.dict(os.environ, env_vars):
            provider = OpenAIEmbeddingProvider.from_env()

            assert provider.api_key == "test-key"
            assert provider.model == "text-embedding-3-small"
            assert provider.organization == "test-org"
            assert provider.get_dimension() == 512
            assert provider.timeout == 60.0
            assert provider.batch_size == 200

    def test_from_env_missing_api_key(self):
        """Test from_env fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY.*required"):
                OpenAIEmbeddingProvider.from_env()

    def test_from_env_invalid_dimensions(self):
        """Test from_env handles invalid dimensions gracefully."""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_DIMENSIONS": "invalid",
        }

        with patch("bruno_llm.providers.openai.embedding_provider.logger") as mock_logger:
            with patch.dict(os.environ, env_vars):
                provider = OpenAIEmbeddingProvider.from_env()

                # Should use default dimensions and log warning
                assert provider.get_dimension() == 1536
                mock_logger.warning.assert_called()


class TestOpenAIEmbeddingConfig:
    """Test suite for OpenAIEmbeddingConfig."""

    def test_config_basic(self):
        """Test basic configuration creation."""
        config = OpenAIEmbeddingConfig(api_key="test-key")

        assert config.api_key.get_secret_value() == "test-key"
        assert config.model == "text-embedding-ada-002"
        assert config.timeout == 30.0
        assert config.batch_size == 100
        assert config.dimensions is None

    def test_config_with_dimensions(self):
        """Test configuration with custom dimensions."""
        config = OpenAIEmbeddingConfig(
            api_key="test-key", model="text-embedding-3-small", dimensions=512
        )

        assert config.dimensions == 512
        assert config.model == "text-embedding-3-small"

    def test_config_validation_dimensions_range(self):
        """Test dimension validation."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            OpenAIEmbeddingConfig(api_key="test-key", dimensions=0)

        with pytest.raises(ValidationError, match="Input should be less than or equal to 4096"):
            OpenAIEmbeddingConfig(api_key="test-key", dimensions=5000)

    def test_config_validation_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            OpenAIEmbeddingConfig(api_key="test-key", batch_size=0)

        with pytest.raises(ValidationError, match="Input should be less than or equal to 2048"):
            OpenAIEmbeddingConfig(api_key="test-key", batch_size=3000)

    def test_config_immutable(self):
        """Test configuration is immutable after creation."""
        config = OpenAIEmbeddingConfig(api_key="test-key")

        with pytest.raises(ValidationError):
            config.api_key = "new-key"
