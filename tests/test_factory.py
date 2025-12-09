"""Tests for LLM factory."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from bruno_llm.exceptions import ConfigurationError, LLMError
from bruno_llm.factory import LLMFactory
from bruno_llm.providers.ollama import OllamaProvider
from bruno_llm.providers.openai import OpenAIProvider


def test_factory_list_providers():
    """Test listing registered providers."""
    providers = LLMFactory.list_providers()

    assert "ollama" in providers
    assert "openai" in providers
    assert len(providers) >= 2


def test_factory_is_registered():
    """Test checking if provider is registered."""
    assert LLMFactory.is_registered("ollama") is True
    assert LLMFactory.is_registered("openai") is True
    assert LLMFactory.is_registered("OLLAMA") is True  # Case insensitive
    assert LLMFactory.is_registered("nonexistent") is False


def test_factory_register():
    """Test registering a new provider."""

    class CustomProvider:
        def __init__(self, **kwargs):
            pass

    LLMFactory.register("custom", CustomProvider)

    assert LLMFactory.is_registered("custom")
    provider = LLMFactory.create("custom")
    assert isinstance(provider, CustomProvider)


def test_factory_create_ollama():
    """Test creating Ollama provider."""
    provider = LLMFactory.create("ollama", {"model": "llama2"})

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama2"


def test_factory_create_openai():
    """Test creating OpenAI provider."""
    provider = LLMFactory.create("openai", {"api_key": "sk-test", "model": "gpt-4"})

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4"


def test_factory_create_with_kwargs():
    """Test creating provider with kwargs instead of config dict."""
    provider = LLMFactory.create("ollama", model="mistral")

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "mistral"


def test_factory_create_merge_config_and_kwargs():
    """Test creating provider with both config dict and kwargs."""
    provider = LLMFactory.create(
        "ollama", config={"model": "llama2"}, base_url="http://localhost:11434"
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama2"


def test_factory_create_invalid_provider():
    """Test creating with invalid provider name."""
    with pytest.raises(ConfigurationError) as exc_info:
        LLMFactory.create("invalid_provider")

    assert "Provider 'invalid_provider' not found" in str(exc_info.value)
    assert "ollama" in str(exc_info.value)
    assert "openai" in str(exc_info.value)


def test_factory_create_invalid_config():
    """Test creating with invalid configuration."""
    with pytest.raises(ConfigurationError) as exc_info:
        # OpenAI requires api_key
        LLMFactory.create("openai", {"invalid_param": "value"})

    assert "Invalid configuration" in str(exc_info.value)


def test_factory_create_from_env_ollama():
    """Test creating Ollama from environment variables."""
    with patch.dict(
        os.environ,
        {"BRUNO_LLM_OLLAMA_MODEL": "llama2", "BRUNO_LLM_OLLAMA_BASE_URL": "http://localhost:11434"},
    ):
        provider = LLMFactory.create_from_env("ollama")

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama2"


def test_factory_create_from_env_openai():
    """Test creating OpenAI from environment variables."""
    with patch.dict(
        os.environ, {"BRUNO_LLM_OPENAI_API_KEY": "sk-test", "BRUNO_LLM_OPENAI_MODEL": "gpt-4"}
    ):
        provider = LLMFactory.create_from_env("openai")

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4"


def test_factory_create_from_env_custom_prefix():
    """Test creating from env with custom prefix."""
    with patch.dict(os.environ, {"MY_APP_OLLAMA_MODEL": "mistral"}):
        provider = LLMFactory.create_from_env("ollama", prefix="MY_APP")

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "mistral"


def test_factory_create_from_env_no_vars():
    """Test creating from env with no matching variables."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigurationError) as exc_info:
            LLMFactory.create_from_env("ollama")

        assert "No environment variables found" in str(exc_info.value)


def test_factory_create_from_env_case_insensitive():
    """Test env var names are converted to lowercase for config."""
    with patch.dict(
        os.environ,
        {"BRUNO_LLM_OLLAMA_MODEL": "llama2", "BRUNO_LLM_OLLAMA_BASE_URL": "http://localhost:11434"},
    ):
        provider = LLMFactory.create_from_env("ollama")

        # Config keys should work (converted to lowercase)
        assert provider.model == "llama2"


@pytest.mark.asyncio
async def test_factory_create_with_fallback_first_succeeds():
    """Test fallback with first provider succeeding."""
    with patch.object(OllamaProvider, "check_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        provider = await LLMFactory.create_with_fallback(
            providers=["ollama", "openai"],
            configs=[{"model": "llama2"}, {"api_key": "sk-test", "model": "gpt-4"}],
        )

        assert isinstance(provider, OllamaProvider)
        mock_check.assert_called_once()


@pytest.mark.asyncio
async def test_factory_create_with_fallback_second_succeeds():
    """Test fallback with first provider failing, second succeeding."""
    with patch.object(OllamaProvider, "check_connection", new_callable=AsyncMock) as mock_ollama:
        mock_ollama.return_value = False

        with patch.object(
            OpenAIProvider, "check_connection", new_callable=AsyncMock
        ) as mock_openai:
            mock_openai.return_value = True

            provider = await LLMFactory.create_with_fallback(
                providers=["ollama", "openai"],
                configs=[{"model": "llama2"}, {"api_key": "sk-test", "model": "gpt-4"}],
            )

            assert isinstance(provider, OpenAIProvider)
            mock_ollama.assert_called_once()
            mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_factory_create_with_fallback_all_fail():
    """Test fallback with all providers failing."""
    with patch.object(OllamaProvider, "check_connection", new_callable=AsyncMock) as mock_ollama:
        mock_ollama.return_value = False

        with patch.object(
            OpenAIProvider, "check_connection", new_callable=AsyncMock
        ) as mock_openai:
            mock_openai.return_value = False

            with pytest.raises(LLMError) as exc_info:
                await LLMFactory.create_with_fallback(
                    providers=["ollama", "openai"],
                    configs=[{"model": "llama2"}, {"api_key": "sk-test", "model": "gpt-4"}],
                )

            assert "All providers failed to connect" in str(exc_info.value)
            assert "ollama" in str(exc_info.value)
            assert "openai" in str(exc_info.value)


@pytest.mark.asyncio
async def test_factory_create_with_fallback_no_configs():
    """Test fallback without explicit configs."""
    with patch.object(OllamaProvider, "check_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        provider = await LLMFactory.create_with_fallback(providers=["ollama"])

        assert isinstance(provider, OllamaProvider)


@pytest.mark.asyncio
async def test_factory_create_with_fallback_empty_providers():
    """Test fallback with no providers specified."""
    with pytest.raises(ConfigurationError) as exc_info:
        await LLMFactory.create_with_fallback(providers=[])

    assert "No providers specified" in str(exc_info.value)


@pytest.mark.asyncio
async def test_factory_create_with_fallback_mismatched_configs():
    """Test fallback with mismatched providers and configs length."""
    with pytest.raises(ConfigurationError) as exc_info:
        await LLMFactory.create_with_fallback(
            providers=["ollama", "openai"],
            configs=[{"model": "llama2"}],  # Only one config
        )

    assert "configs length" in str(exc_info.value)


@pytest.mark.asyncio
async def test_factory_create_with_fallback_exception_handling():
    """Test fallback handles provider creation exceptions."""
    with patch.object(LLMFactory, "create") as mock_create:
        # First provider raises exception
        mock_create.side_effect = [Exception("Provider 1 failed"), OllamaProvider(model="llama2")]

        with patch.object(OllamaProvider, "check_connection", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True

            provider = await LLMFactory.create_with_fallback(
                providers=["openai", "ollama"],
                configs=[{"api_key": "sk-test"}, {"model": "llama2"}],
            )

            assert isinstance(provider, OllamaProvider)
