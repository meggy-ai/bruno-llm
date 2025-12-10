"""Factory for creating LLM provider instances."""

import os
from typing import Any, Callable, Optional

from bruno_core.interfaces import EmbeddingInterface, LLMInterface
from bruno_llm.exceptions import ConfigurationError, LLMError


class LLMFactory:
    """
    Factory for creating LLM provider instances.

    Provides multiple ways to instantiate providers:
    - Direct creation with configuration
    - Environment-based configuration
    - Fallback chain for resilience

    Examples:
        >>> # Direct creation
        >>> llm = LLMFactory.create(
        ...     provider="ollama",
        ...     config={"model": "llama2"}
        ... )

        >>> # From environment variables
        >>> llm = LLMFactory.create_from_env(provider="openai")

        >>> # With fallback
        >>> llm = LLMFactory.create_with_fallback(
        ...     providers=["openai", "ollama"],
        ...     configs=[openai_config, ollama_config]
        ... )
    """

    _providers: dict[str, Callable[..., LLMInterface]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Callable[..., LLMInterface]) -> None:
        """
        Register a provider class with the factory.

        Args:
            name: Provider name (e.g., "ollama", "openai")
            provider_class: Provider class or factory function
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls, provider: str, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> LLMInterface:
        """
        Create a provider instance.

        Args:
            provider: Provider name ("ollama", "openai", etc.)
            config: Configuration dictionary (optional)
            **kwargs: Additional arguments passed to provider

        Returns:
            Configured LLMInterface instance

        Raises:
            ConfigurationError: If provider not found or config invalid

        Examples:
            >>> llm = LLMFactory.create("ollama", {"model": "llama2"})
            >>> llm = LLMFactory.create("openai", api_key="sk-...", model="gpt-4")
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ConfigurationError(
                f"Provider '{provider}' not found. Available providers: {available}"
            )

        provider_class = cls._providers[provider_lower]

        # Merge config dict and kwargs
        final_config = config.copy() if config else {}
        final_config.update(kwargs)

        try:
            return provider_class(**final_config)
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration for provider '{provider}': {e}") from e

    @classmethod
    def create_from_env(cls, provider: str, prefix: Optional[str] = None) -> LLMInterface:
        """
        Create provider from environment variables.

        Environment variables are read using the pattern:
        {PREFIX}_{PROVIDER}_{SETTING}

        Args:
            provider: Provider name
            prefix: Environment variable prefix (default: "BRUNO_LLM")

        Returns:
            Configured LLMInterface instance

        Raises:
            ConfigurationError: If required env vars missing

        Examples:
            >>> # With BRUNO_LLM_OPENAI_API_KEY=sk-...
            >>> # and BRUNO_LLM_OPENAI_MODEL=gpt-4
            >>> llm = LLMFactory.create_from_env("openai")

            >>> # Custom prefix
            >>> # MY_APP_OLLAMA_MODEL=llama2
            >>> llm = LLMFactory.create_from_env("ollama", prefix="MY_APP")
        """
        prefix = prefix or "BRUNO_LLM"
        provider_upper = provider.upper()
        env_prefix = f"{prefix}_{provider_upper}_"

        # Collect all matching environment variables
        config: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                setting_name = key[len(env_prefix) :].lower()
                config[setting_name] = value

        if not config:
            raise ConfigurationError(
                f"No environment variables found for provider '{provider}'. "
                f"Expected variables starting with {env_prefix}"
            )

        return cls.create(provider, config)

    @classmethod
    async def create_with_fallback(
        cls, providers: list[str], configs: Optional[list[dict[str, Any]]] = None
    ) -> LLMInterface:
        """
        Create provider with fallback chain.

        Tries each provider in order until one connects successfully.

        Args:
            providers: List of provider names in priority order
            configs: Optional list of configurations (must match providers length)

        Returns:
            First successfully connected LLMInterface instance

        Raises:
            LLMError: If all providers fail to connect

        Examples:
            >>> llm = await LLMFactory.create_with_fallback(
            ...     providers=["openai", "ollama"],
            ...     configs=[
            ...         {"api_key": "sk-...", "model": "gpt-4"},
            ...         {"model": "llama2"}
            ...     ]
            ... )
        """
        if not providers:
            raise ConfigurationError("No providers specified for fallback")

        if configs and len(configs) != len(providers):
            raise ConfigurationError(
                f"configs length ({len(configs)}) must match providers length ({len(providers)})"
            )

        errors = []

        for i, provider_name in enumerate(providers):
            try:
                config = configs[i] if configs else {}
                provider = cls.create(provider_name, config)

                # Test connection
                if await provider.check_connection():
                    return provider
                else:
                    errors.append(f"{provider_name}: Connection check failed")

            except Exception as e:
                errors.append(f"{provider_name}: {e}")
                continue

        # All providers failed
        error_details = "; ".join(errors)
        raise LLMError(
            f"All providers failed to connect. Tried: {', '.join(providers)}. "
            f"Errors: {error_details}"
        )

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List all registered providers.

        Returns:
            List of provider names
        """
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider: Provider name

        Returns:
            True if provider is registered
        """
        return provider.lower() in cls._providers


# Register built-in providers
def _register_builtin_providers():
    """Register all built-in providers."""
    try:
        from bruno_llm.providers.ollama import OllamaProvider

        LLMFactory.register("ollama", OllamaProvider)
    except ImportError:
        pass

    try:
        from bruno_llm.providers.openai import OpenAIProvider

        LLMFactory.register("openai", OpenAIProvider)
    except ImportError:
        pass


# Auto-register on import
_register_builtin_providers()


class UnifiedProviderFactory:
    """
    Unified factory for creating both LLM and Embedding providers.

    This factory provides a single interface for creating any type of provider,
    integrating both LLMFactory and EmbeddingFactory.

    Examples:
        >>> # Create LLM provider
        >>> llm = UnifiedProviderFactory.create_llm("openai", {"api_key": "sk-..."})

        >>> # Create embedding provider
        >>> embedder = UnifiedProviderFactory.create_embedding("openai", {"api_key": "sk-..."})

        >>> # List all available providers
        >>> all_providers = UnifiedProviderFactory.list_all_providers()
    """

    @classmethod
    def create_llm(
        cls, provider: str, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> LLMInterface:
        """
        Create an LLM provider instance.

        Args:
            provider: Provider name
            config: Configuration dictionary
            **kwargs: Additional configuration arguments

        Returns:
            LLM provider instance
        """
        return LLMFactory.create(provider, config, **kwargs)

    @classmethod
    def create_embedding(
        cls, provider: str, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> EmbeddingInterface:
        """
        Create an embedding provider instance.

        Args:
            provider: Provider name
            config: Configuration dictionary
            **kwargs: Additional configuration arguments

        Returns:
            Embedding provider instance

        Raises:
            ImportError: If embedding support not available
        """
        try:
            from bruno_llm.embedding_factory import EmbeddingFactory

            return EmbeddingFactory.create(provider, config, **kwargs)
        except ImportError as e:
            raise ConfigurationError(
                "Embedding support not available. Please ensure embedding providers are installed."
            ) from e

    @classmethod
    def list_llm_providers(cls) -> list[str]:
        """List all available LLM providers."""
        return LLMFactory.list_providers()

    @classmethod
    def list_embedding_providers(cls) -> list[str]:
        """List all available embedding providers."""
        try:
            from bruno_llm.embedding_factory import EmbeddingFactory

            return EmbeddingFactory.list_providers()
        except ImportError:
            return []

    @classmethod
    def list_all_providers(cls) -> dict[str, list[str]]:
        """
        List all available providers by type.

        Returns:
            Dictionary with 'llm' and 'embedding' keys mapping to provider lists
        """
        return {
            "llm": cls.list_llm_providers(),
            "embedding": cls.list_embedding_providers(),
        }
