"""
Factory for creating embedding provider instances.

This module provides a factory pattern for creating embedding providers,
following the same design as LLMFactory but for EmbeddingInterface implementations.
Uses external libraries and follows the bruno-llm architecture patterns.
"""

import os
from typing import Any, Callable, Optional

from bruno_core.interfaces import EmbeddingInterface
from bruno_llm.exceptions import ConfigurationError, ProviderNotFoundError


class EmbeddingFactory:
    """
    Factory for creating embedding provider instances.

    Provides multiple ways to instantiate embedding providers:
    - Direct creation with configuration
    - Environment-based configuration
    - Provider discovery and listing

    Examples:
        >>> # Direct creation
        >>> embedder = EmbeddingFactory.create(
        ...     provider="openai",
        ...     config={"api_key": "sk-...", "model": "text-embedding-ada-002"}
        ... )

        >>> # From environment variables
        >>> embedder = EmbeddingFactory.create_from_env(provider="openai")

        >>> # List available providers
        >>> providers = EmbeddingFactory.list_providers()
    """

    _providers: dict[str, Callable[..., EmbeddingInterface]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Callable[..., EmbeddingInterface]) -> None:
        """
        Register an embedding provider class with the factory.

        Args:
            name: Provider name (e.g., "openai", "huggingface", "ollama")
            provider_class: Provider class or factory function

        Example:
            >>> EmbeddingFactory.register("custom", CustomEmbeddingProvider)
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls, provider: str, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> EmbeddingInterface:
        """
        Create an embedding provider instance.

        Args:
            provider: Provider name ("openai", "huggingface", "ollama", etc.)
            config: Configuration dictionary (optional)
            **kwargs: Additional arguments passed to provider

        Returns:
            Configured EmbeddingInterface instance

        Raises:
            ProviderNotFoundError: If provider not registered
            ConfigurationError: If configuration is invalid

        Examples:
            >>> embedder = EmbeddingFactory.create(
            ...     "openai",
            ...     {"api_key": "sk-...", "model": "text-embedding-ada-002"}
            ... )
            >>> embedder = EmbeddingFactory.create(
            ...     "huggingface",
            ...     model="sentence-transformers/all-MiniLM-L6-v2"
            ... )
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ProviderNotFoundError(
                f"Embedding provider '{provider}' not found. " f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_lower]

        # Merge config dict and kwargs
        final_config = config.copy() if config else {}
        final_config.update(kwargs)

        try:
            return provider_class(**final_config)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create embedding provider '{provider}': {e}"
            ) from e

    @classmethod
    def create_from_env(cls, provider: str, prefix: Optional[str] = None) -> EmbeddingInterface:
        """
        Create embedding provider from environment variables.

        Environment variables are read using the pattern:
        {PREFIX}_{PROVIDER}_EMBEDDING_{SETTING}

        Args:
            provider: Provider name
            prefix: Environment variable prefix (default: "BRUNO_LLM")

        Returns:
            Configured EmbeddingInterface instance

        Raises:
            ConfigurationError: If required env vars missing

        Environment Variables by Provider:
            OpenAI:
                - BRUNO_LLM_OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY
                - BRUNO_LLM_OPENAI_EMBEDDING_MODEL (optional)
                - BRUNO_LLM_OPENAI_EMBEDDING_ORG_ID (optional)

            HuggingFace:
                - BRUNO_LLM_HUGGINGFACE_EMBEDDING_MODEL (optional)
                - BRUNO_LLM_HUGGINGFACE_EMBEDDING_DEVICE (optional)

            Ollama:
                - BRUNO_LLM_OLLAMA_EMBEDDING_BASE_URL (optional)
                - BRUNO_LLM_OLLAMA_EMBEDDING_MODEL (optional)

        Examples:
            >>> # With environment variables set
            >>> embedder = EmbeddingFactory.create_from_env("openai")
        """
        prefix = prefix or "BRUNO_LLM"
        provider_upper = provider.upper()
        env_prefix = f"{prefix}_{provider_upper}_EMBEDDING_"

        # Collect all matching environment variables
        config: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                setting_name = key[len(env_prefix) :].lower()
                config[setting_name] = value

        # Check for provider-specific common environment variables
        if provider.lower() == "openai":
            # Check for standard OpenAI environment variables
            if "api_key" not in config and "OPENAI_API_KEY" in os.environ:
                config["api_key"] = os.environ["OPENAI_API_KEY"]
            if "org_id" not in config and "OPENAI_ORG_ID" in os.environ:
                config["org_id"] = os.environ["OPENAI_ORG_ID"]

        # Check for common embedding environment variables
        if "timeout" not in config and "EMBEDDING_TIMEOUT" in os.environ:
            try:
                config["timeout"] = float(os.environ["EMBEDDING_TIMEOUT"])
            except ValueError:
                pass

        if "batch_size" not in config and "EMBEDDING_BATCH_SIZE" in os.environ:
            try:
                config["batch_size"] = int(os.environ["EMBEDDING_BATCH_SIZE"])
            except ValueError:
                pass

        return cls.create(provider, config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List all registered embedding providers.

        Returns:
            List of provider names

        Example:
            >>> providers = EmbeddingFactory.list_providers()
            >>> print(providers)
            ['openai', 'huggingface', 'ollama']
        """
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """
        Check if an embedding provider is registered.

        Args:
            provider: Provider name

        Returns:
            True if provider is registered

        Example:
            >>> if EmbeddingFactory.is_registered("openai"):
            ...     embedder = EmbeddingFactory.create("openai", config)
        """
        return provider.lower() in cls._providers

    @classmethod
    def get_provider_info(cls, provider: str) -> dict[str, Any]:
        """
        Get information about a registered provider.

        Args:
            provider: Provider name

        Returns:
            Provider information dictionary

        Raises:
            ProviderNotFoundError: If provider not registered

        Example:
            >>> info = EmbeddingFactory.get_provider_info("openai")
            >>> print(info["class_name"])
        """
        if not cls.is_registered(provider):
            raise ProviderNotFoundError(f"Provider '{provider}' not registered")

        provider_class = cls._providers[provider.lower()]

        return {
            "name": provider,
            "class_name": provider_class.__name__,
            "module": provider_class.__module__,
            "docstring": provider_class.__doc__ or "",
        }


def _register_builtin_providers():
    """Register all built-in embedding providers if available."""

    # Register OpenAI embedding provider
    try:
        from bruno_llm.providers.openai import OpenAIEmbeddingProvider

        EmbeddingFactory.register("openai", OpenAIEmbeddingProvider)
    except ImportError:
        pass

    # Register HuggingFace embedding provider
    try:
        from bruno_llm.providers.huggingface import HuggingFaceEmbeddingProvider

        EmbeddingFactory.register("huggingface", HuggingFaceEmbeddingProvider)
    except ImportError:
        pass

    # Register Ollama embedding provider
    try:
        from bruno_llm.providers.ollama import OllamaEmbeddingProvider

        EmbeddingFactory.register("ollama", OllamaEmbeddingProvider)
    except ImportError:
        pass


# Auto-register built-in providers on import
_register_builtin_providers()
