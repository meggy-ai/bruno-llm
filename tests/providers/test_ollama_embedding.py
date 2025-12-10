"""
Tests for Ollama Embedding Provider.

This module tests the OllamaEmbeddingProvider implementation using mocks
to avoid requiring a running Ollama instance during testing.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from bruno_llm.exceptions import (
    InvalidResponseError,
    LLMError,
    ModelNotFoundError,
)
from bruno_llm.exceptions import (
    TimeoutError as LLMTimeoutError,
)
from bruno_llm.providers.ollama.config import OllamaEmbeddingConfig
from bruno_llm.providers.ollama.embedding_provider import OllamaEmbeddingProvider


class TestOllamaEmbeddingProvider:
    """Test suite for OllamaEmbeddingProvider."""

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama embedding response."""
        return {
            "embedding": [0.1, 0.2, 0.3] * 256,  # 768 dimensions for nomic-embed-text
        }

    @pytest.fixture
    def mock_ollama_batch_response(self):
        """Mock Ollama batch embedding response."""
        return {
            "embeddings": [
                [0.1, 0.2, 0.3] * 256,  # 768 dimensions
                [0.4, 0.5, 0.6] * 256,
                [0.7, 0.8, 0.9] * 256,
            ]
        }

    @pytest.fixture
    def provider(self):
        """Create test provider instance."""
        return OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

    def test_init_basic(self):
        """Test basic provider initialization."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434", model="nomic-embed-text"
        )

        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "nomic-embed-text"
        assert provider.get_dimension() == 768  # Known dimension for nomic-embed-text
        assert provider.timeout == 30.0
        assert provider.batch_size == 32

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        provider = OllamaEmbeddingProvider(
            base_url="http://192.168.1.100:11434",
            model="mxbai-embed-large",
            timeout=60.0,
            batch_size=16,
        )

        assert provider.base_url == "http://192.168.1.100:11434"
        assert provider.model == "mxbai-embed-large"
        assert provider.get_dimension() == 1024  # Known dimension for mxbai-embed-large
        assert provider.timeout == 60.0
        assert provider.batch_size == 16

    def test_init_unknown_model(self):
        """Test initialization with unknown model uses default dimension."""
        provider = OllamaEmbeddingProvider(model="custom-model")

        assert provider.model == "custom-model"
        assert provider.get_dimension() == 768  # Default dimension

    @pytest.mark.asyncio
    async def test_embed_text_success(self, provider, mock_ollama_response):
        """Test successful single text embedding."""
        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_ollama_response

            result = await provider.embed_text("Hello world")

            assert len(result) == 768
            assert all(isinstance(x, float) for x in result)
            mock_request.assert_called_once_with({"input": "Hello world"})

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self, provider):
        """Test embedding empty string returns zero vector."""
        result = await provider.embed_text("")

        assert len(result) == 768
        assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_dimension_update(self, provider):
        """Test that provider updates dimensions when model returns different size."""
        # Mock response with different dimensions
        mock_response = {"embedding": [0.1] * 1024}  # Different from expected 768

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await provider.embed_text("Hello world")

            assert len(result) == 1024
            assert provider.get_dimension() == 1024  # Should be updated

    @pytest.mark.asyncio
    async def test_embed_texts_batch_success(self, provider, mock_ollama_batch_response):
        """Test successful batch embedding."""
        texts = ["Hello", "World", "Test"]

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_ollama_batch_response

            results = await provider.embed_texts(texts)

            assert len(results) == 3
            assert all(len(embedding) == 768 for embedding in results)
            mock_request.assert_called_once_with({"input": texts})

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, provider):
        """Test embedding empty list returns empty results."""
        result = await provider.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_with_empty_strings(self, provider):
        """Test embedding with some empty strings."""
        texts = ["Hello", "", "World"]

        # Mock response for only non-empty texts
        mock_response = {
            "embeddings": [
                [0.1] * 768,  # "Hello"
                [0.2] * 768,  # "World"
            ]
        }

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            results = await provider.embed_texts(texts)

            assert len(results) == 3
            assert results[0] == [0.1] * 768  # "Hello"
            assert results[1] == [0.0] * 768  # Empty string -> zero vector
            assert results[2] == [0.2] * 768  # "World"

            # Should only call API with non-empty texts
            mock_request.assert_called_once_with({"input": ["Hello", "World"]})

    @pytest.mark.asyncio
    async def test_embed_texts_large_batch_splitting(self, provider):
        """Test that large batches are split correctly."""
        # Set small batch size for testing
        provider.batch_size = 2
        texts = ["text1", "text2", "text3", "text4", "text5"]

        # Mock responses for each batch
        mock_response1 = {
            "embeddings": [
                [0.1] * 768,
                [0.2] * 768,
            ]
        }

        mock_response2 = {
            "embeddings": [
                [0.3] * 768,
                [0.4] * 768,
            ]
        }

        mock_response3 = {
            "embeddings": [
                [0.5] * 768,
            ]
        }

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [mock_response1, mock_response2, mock_response3]

            results = await provider.embed_texts(texts)

            assert len(results) == 5
            assert mock_request.call_count == 3

            # Check batch calls
            calls = mock_request.call_args_list
            assert calls[0][0][0]["input"] == ["text1", "text2"]
            assert calls[1][0][0]["input"] == ["text3", "text4"]
            assert calls[2][0][0]["input"] == ["text5"]

    @pytest.mark.asyncio
    async def test_embed_text_single_embedding_response(self, provider):
        """Test handling single embedding response format (older Ollama versions)."""
        mock_response = {"embedding": [0.1] * 768}

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            # Test with _embed_batch directly
            results = await provider._embed_batch(["test"])

            assert len(results) == 1
            assert len(results[0]) == 768

    @pytest.mark.asyncio
    async def test_check_connection_success(self, provider, mock_ollama_response):
        """Test successful connection check."""
        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_ollama_response

            result = await provider.check_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_failure(self, provider):
        """Test connection check failure."""
        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = LLMError("Connection failed")

            result = await provider.check_connection()

            assert result is False

    @pytest.mark.asyncio
    async def test_make_request_timeout(self, provider):
        """Test request timeout handling."""
        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(LLMTimeoutError, match="Ollama request timeout"):
                await provider._make_request({"input": "test"})

    @pytest.mark.asyncio
    async def test_make_request_model_not_found(self, provider):
        """Test model not found error handling."""
        # Mock 404 response with model error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model not found"}

        http_error = httpx.HTTPStatusError(
            message="Not found", request=MagicMock(), response=mock_response
        )

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = http_error

            with pytest.raises(ModelNotFoundError, match="Ollama model.*not found"):
                await provider._make_request({"input": "test"})

    @pytest.mark.asyncio
    async def test_make_request_server_error(self, provider):
        """Test server error handling."""
        # Mock 500 response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}

        http_error = httpx.HTTPStatusError(
            message="Server error", request=MagicMock(), response=mock_response
        )

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = http_error

            with pytest.raises(LLMError, match="Ollama server error"):
                await provider._make_request({"input": "test"})

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, provider):
        """Test connection error handling."""
        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.RequestError("Connection failed")

            with pytest.raises(LLMError, match="Failed to connect to Ollama"):
                await provider._make_request({"input": "test"})

    @pytest.mark.asyncio
    async def test_invalid_response_no_embedding(self, provider):
        """Test handling of invalid response format."""
        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "ok"}  # No embedding field

            with pytest.raises(InvalidResponseError, match="No embedding in Ollama response"):
                await provider.embed_text("test")

    @pytest.mark.asyncio
    async def test_batch_embedding_count_mismatch(self, provider):
        """Test handling of embedding count mismatch in batch."""
        texts = ["text1", "text2", "text3"]

        # Mock response with wrong number of embeddings
        mock_response = {
            "embeddings": [
                [0.1] * 768,
                [0.2] * 768,
            ]  # Only 2 embeddings for 3 texts
        }

        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(InvalidResponseError, match="Expected 3 embeddings, got 2"):
                await provider._embed_batch(texts)

    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Test async context manager functionality."""
        async with provider as p:
            assert p == provider

        # Client should be closed after context manager exit
        # Note: We can't easily test this without mocking the client

    def test_from_env_success(self):
        """Test creating provider from environment variables."""
        env_vars = {
            "OLLAMA_BASE_URL": "http://test-server:11434",
            "OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large",
            "EMBEDDING_TIMEOUT": "60.0",
            "EMBEDDING_BATCH_SIZE": "16",
        }

        with patch.dict(os.environ, env_vars):
            provider = OllamaEmbeddingProvider.from_env()

            assert provider.base_url == "http://test-server:11434"
            assert provider.model == "mxbai-embed-large"
            assert provider.timeout == 60.0
            assert provider.batch_size == 16

    def test_from_env_defaults(self):
        """Test from_env with default values."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OllamaEmbeddingProvider.from_env()

            assert provider.base_url == "http://localhost:11434"
            assert provider.model == "nomic-embed-text"
            assert provider.timeout == 30.0
            assert provider.batch_size == 32

    def test_from_env_invalid_values(self):
        """Test from_env handles invalid environment values gracefully."""
        env_vars = {
            "EMBEDDING_TIMEOUT": "invalid",
            "EMBEDDING_BATCH_SIZE": "not-a-number",
        }

        with patch("bruno_llm.providers.ollama.embedding_provider.logger") as mock_logger:
            with patch.dict(os.environ, env_vars):
                provider = OllamaEmbeddingProvider.from_env()

                # Should use defaults and log warnings
                assert provider.timeout == 30.0
                assert provider.batch_size == 32
                assert mock_logger.warning.call_count == 2

    def test_from_env_with_kwargs_override(self):
        """Test from_env with kwargs override."""
        env_vars = {
            "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        }

        with patch.dict(os.environ, env_vars):
            provider = OllamaEmbeddingProvider.from_env(
                model="mxbai-embed-large",  # Override env var
                batch_size=64,
            )

            assert provider.model == "mxbai-embed-large"  # Kwargs override
            assert provider.batch_size == 64


class TestOllamaEmbeddingConfig:
    """Test suite for OllamaEmbeddingConfig."""

    def test_config_basic(self):
        """Test basic configuration creation."""
        config = OllamaEmbeddingConfig()

        assert config.base_url == "http://localhost:11434"
        assert config.model == "nomic-embed-text"
        assert config.timeout == 30.0
        assert config.batch_size == 32

    def test_config_custom(self):
        """Test configuration with custom values."""
        config = OllamaEmbeddingConfig(
            base_url="http://192.168.1.100:11434",
            model="mxbai-embed-large",
            timeout=60.0,
            batch_size=16,
        )

        assert config.base_url == "http://192.168.1.100:11434"
        assert config.model == "mxbai-embed-large"
        assert config.timeout == 60.0
        assert config.batch_size == 16

    def test_config_validation_timeout(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            OllamaEmbeddingConfig(timeout=-1.0)

    def test_config_validation_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
            OllamaEmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError, match="Input should be less than or equal to 128"):
            OllamaEmbeddingConfig(batch_size=200)

    def test_config_immutable(self):
        """Test configuration is immutable after creation."""
        config = OllamaEmbeddingConfig()

        with pytest.raises(ValidationError):
            config.model = "new-model"
