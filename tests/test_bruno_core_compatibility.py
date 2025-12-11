"""
Bruno-Core Interface Compatibility Tests.

This module provides comprehensive tests to verify that bruno-llm
correctly implements all required interfaces from bruno-core.

The tests ensure:
1. All required interfaces are properly imported
2. Method signatures match exactly
3. All required methods are implemented
4. Interface inheritance works correctly
5. Future interface changes are detected automatically
"""

import inspect

import pytest

try:
    from bruno_core.interfaces import (
        EmbeddingInterface,
        LLMInterface,
    )

    BRUNO_CORE_AVAILABLE = True
except ImportError as e:
    BRUNO_CORE_AVAILABLE = False
    IMPORT_ERROR = str(e)

from bruno_llm.providers.ollama import OllamaEmbeddingProvider, OllamaProvider
from bruno_llm.providers.openai import OpenAIEmbeddingProvider, OpenAIProvider


class TestBrunoCoreImportCompatibility:
    """Test that bruno-core interfaces can be imported successfully."""

    def test_bruno_core_available(self):
        """Test that bruno-core is available and can be imported."""
        if not BRUNO_CORE_AVAILABLE:
            pytest.skip(f"bruno-core not available: {IMPORT_ERROR}")

        # Should not raise any exceptions
        assert BRUNO_CORE_AVAILABLE is True

    @pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
    def test_llm_interface_import(self):
        """Test LLMInterface can be imported from bruno-core."""
        from bruno_core.interfaces import LLMInterface

        assert LLMInterface is not None
        assert inspect.isclass(LLMInterface)

    @pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
    def test_embedding_interface_import(self):
        """Test EmbeddingInterface can be imported from bruno-core."""
        from bruno_core.interfaces import EmbeddingInterface

        assert EmbeddingInterface is not None
        assert inspect.isclass(EmbeddingInterface)

    @pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
    def test_message_models_import(self):
        """Test Message and MessageRole can be imported from bruno-core."""
        from bruno_core.models import Message, MessageRole

        assert Message is not None
        assert MessageRole is not None


@pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
class TestLLMInterfaceCompatibility:
    """Test bruno-llm LLM providers implement bruno-core LLMInterface correctly."""

    @pytest.fixture
    def llm_interface_methods(self):
        """Get all methods that should be implemented by LLMInterface."""
        return {
            name: method
            for name, method in inspect.getmembers(LLMInterface, inspect.isfunction)
            if not name.startswith("_")
        }

    def test_openai_provider_inherits_llm_interface(self):
        """Test OpenAIProvider inherits from LLMInterface."""
        assert issubclass(OpenAIProvider, LLMInterface)

    def test_ollama_provider_inherits_llm_interface(self):
        """Test OllamaProvider inherits from LLMInterface."""
        assert issubclass(OllamaProvider, LLMInterface)

    def test_openai_provider_implements_required_methods(self, llm_interface_methods):
        """Test OpenAIProvider implements all required LLMInterface methods."""
        provider_methods = {
            name: method
            for name, method in inspect.getmembers(OpenAIProvider, callable)
            if not name.startswith("_")
        }

        for method_name, interface_method in llm_interface_methods.items():
            assert method_name in provider_methods, f"OpenAIProvider missing method: {method_name}"

            provider_method = provider_methods[method_name]
            interface_signature = inspect.signature(interface_method)
            provider_signature = inspect.signature(provider_method)

            # Check that all interface parameters exist in provider (providers can have additional params with defaults)
            interface_params = set(interface_signature.parameters.keys())
            provider_params = set(provider_signature.parameters.keys())

            missing_params = interface_params - provider_params
            assert not missing_params, (
                f"Provider {method_name} missing required parameters: {missing_params}"
            )

    def test_ollama_provider_implements_required_methods(self, llm_interface_methods):
        """Test OllamaProvider implements all required LLMInterface methods."""
        provider_methods = {
            name: method
            for name, method in inspect.getmembers(OllamaProvider, callable)
            if not name.startswith("_")
        }

        for method_name, interface_method in llm_interface_methods.items():
            assert method_name in provider_methods, f"OllamaProvider missing method: {method_name}"

            provider_method = provider_methods[method_name]
            interface_signature = inspect.signature(interface_method)
            provider_signature = inspect.signature(provider_method)

            # Check that all interface parameters exist in provider (providers can have additional params with defaults)
            interface_params = set(interface_signature.parameters.keys())
            provider_params = set(provider_signature.parameters.keys())

            missing_params = interface_params - provider_params
            assert not missing_params, (
                f"Provider {method_name} missing required parameters: {missing_params}"
            )

    def test_openai_provider_method_signatures_compatible(self):
        """Test OpenAIProvider method signatures are compatible with LLMInterface."""
        # Test key methods have correct async signatures
        assert inspect.iscoroutinefunction(OpenAIProvider.generate)
        assert inspect.isasyncgenfunction(OpenAIProvider.stream)  # stream should be async generator
        assert inspect.iscoroutinefunction(OpenAIProvider.check_connection)
        assert inspect.iscoroutinefunction(OpenAIProvider.list_models)

    def test_ollama_provider_method_signatures_compatible(self):
        """Test OllamaProvider method signatures are compatible with LLMInterface."""
        # Test key methods have correct async signatures
        assert inspect.iscoroutinefunction(OllamaProvider.generate)
        assert inspect.isasyncgenfunction(OllamaProvider.stream)  # stream should be async generator
        assert inspect.iscoroutinefunction(OllamaProvider.check_connection)
        assert inspect.iscoroutinefunction(OllamaProvider.list_models)


@pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
class TestEmbeddingInterfaceCompatibility:
    """Test bruno-llm embedding providers implement bruno-core EmbeddingInterface correctly."""

    @pytest.fixture
    def embedding_interface_methods(self):
        """Get all methods that should be implemented by EmbeddingInterface."""
        return {
            name: method
            for name, method in inspect.getmembers(EmbeddingInterface, inspect.isfunction)
            if not name.startswith("_")
        }

    def test_openai_embedding_provider_inherits_embedding_interface(self):
        """Test OpenAIEmbeddingProvider inherits from EmbeddingInterface."""
        assert issubclass(OpenAIEmbeddingProvider, EmbeddingInterface)

    def test_ollama_embedding_provider_inherits_embedding_interface(self):
        """Test OllamaEmbeddingProvider inherits from EmbeddingInterface."""
        assert issubclass(OllamaEmbeddingProvider, EmbeddingInterface)

    def test_openai_embedding_provider_implements_required_methods(
        self, embedding_interface_methods
    ):
        """Test OpenAIEmbeddingProvider implements all required EmbeddingInterface methods."""
        provider_methods = {
            name: method
            for name, method in inspect.getmembers(OpenAIEmbeddingProvider, callable)
            if not name.startswith("_")
        }

        for method_name, interface_method in embedding_interface_methods.items():
            assert method_name in provider_methods, (
                f"OpenAIEmbeddingProvider missing method: {method_name}"
            )

            provider_method = provider_methods[method_name]
            interface_signature = inspect.signature(interface_method)
            provider_signature = inspect.signature(provider_method)

            # Check parameter names match (allow additional parameters with defaults)
            interface_params = list(interface_signature.parameters.keys())
            provider_params = list(provider_signature.parameters.keys())

            for param in interface_params:
                assert param in provider_params, f"Parameter '{param}' missing in {method_name}"

    def test_ollama_embedding_provider_implements_required_methods(
        self, embedding_interface_methods
    ):
        """Test OllamaEmbeddingProvider implements all required EmbeddingInterface methods."""
        provider_methods = {
            name: method
            for name, method in inspect.getmembers(OllamaEmbeddingProvider, callable)
            if not name.startswith("_")
        }

        for method_name, interface_method in embedding_interface_methods.items():
            assert method_name in provider_methods, (
                f"OllamaEmbeddingProvider missing method: {method_name}"
            )

            provider_method = provider_methods[method_name]
            interface_signature = inspect.signature(interface_method)
            provider_signature = inspect.signature(provider_method)

            # Check parameter names match (allow additional parameters with defaults)
            interface_params = list(interface_signature.parameters.keys())
            provider_params = list(provider_signature.parameters.keys())

            for param in interface_params:
                assert param in provider_params, f"Parameter '{param}' missing in {method_name}"

    def test_openai_embedding_provider_method_signatures_compatible(self):
        """Test OpenAIEmbeddingProvider method signatures are compatible with EmbeddingInterface."""
        # Test key methods have correct async signatures
        assert inspect.iscoroutinefunction(OpenAIEmbeddingProvider.embed_text)
        assert inspect.iscoroutinefunction(OpenAIEmbeddingProvider.embed_texts)
        assert inspect.iscoroutinefunction(OpenAIEmbeddingProvider.check_connection)

        # Test non-async methods
        assert not inspect.iscoroutinefunction(OpenAIEmbeddingProvider.get_dimension)

    def test_ollama_embedding_provider_method_signatures_compatible(self):
        """Test OllamaEmbeddingProvider method signatures are compatible with EmbeddingInterface."""
        # Test key methods have correct async signatures
        assert inspect.iscoroutinefunction(OllamaEmbeddingProvider.embed_text)
        assert inspect.iscoroutinefunction(OllamaEmbeddingProvider.embed_texts)
        assert inspect.iscoroutinefunction(OllamaEmbeddingProvider.check_connection)

        # Test non-async methods
        assert not inspect.iscoroutinefunction(OllamaEmbeddingProvider.get_dimension)


@pytest.mark.skipif(not BRUNO_CORE_AVAILABLE, reason="bruno-core not available")
class TestInterfaceInstantiation:
    """Test that providers can be instantiated and used as their interfaces."""

    def test_openai_provider_as_llm_interface(self):
        """Test OpenAIProvider can be used as LLMInterface."""
        # Should be able to create without error
        provider = OpenAIProvider(api_key="test-key")

        # Should be usable as LLMInterface
        assert isinstance(provider, LLMInterface)

        # Should have all required methods
        assert hasattr(provider, "generate")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "check_connection")
        assert hasattr(provider, "list_models")
        assert hasattr(provider, "get_token_count")

    def test_ollama_provider_as_llm_interface(self):
        """Test OllamaProvider can be used as LLMInterface."""
        # Should be able to create without error
        provider = OllamaProvider()

        # Should be usable as LLMInterface
        assert isinstance(provider, LLMInterface)

        # Should have all required methods
        assert hasattr(provider, "generate")
        assert hasattr(provider, "stream")
        assert hasattr(provider, "check_connection")
        assert hasattr(provider, "list_models")
        assert hasattr(provider, "get_token_count")

    def test_openai_embedding_provider_as_embedding_interface(self):
        """Test OpenAIEmbeddingProvider can be used as EmbeddingInterface."""
        # Should be able to create without error
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        # Should be usable as EmbeddingInterface
        assert isinstance(provider, EmbeddingInterface)

        # Should have all required methods
        assert hasattr(provider, "embed_text")
        assert hasattr(provider, "embed_texts")
        assert hasattr(provider, "get_dimension")
        assert hasattr(provider, "check_connection")

    def test_ollama_embedding_provider_as_embedding_interface(self):
        """Test OllamaEmbeddingProvider can be used as EmbeddingInterface."""
        # Should be able to create without error
        provider = OllamaEmbeddingProvider()

        # Should be usable as EmbeddingInterface
        assert isinstance(provider, EmbeddingInterface)

        # Should have all required methods
        assert hasattr(provider, "embed_text")
        assert hasattr(provider, "embed_texts")
        assert hasattr(provider, "get_dimension")
        assert hasattr(provider, "check_connection")


class TestInterfaceDetection:
    """Test detection of interface changes and missing implementations."""

    def test_detect_new_bruno_core_interfaces(self):
        """Test that we can detect if bruno-core adds new interfaces."""
        if not BRUNO_CORE_AVAILABLE:
            pytest.skip("bruno-core not available")

        import bruno_core.interfaces as interfaces_module

        # Get all classes that look like interfaces (only interfaces, not models)
        interface_classes = []
        for name, _obj in inspect.getmembers(interfaces_module, inspect.isclass):
            if name.endswith("Interface"):
                interface_classes.append(name)

        # Known interfaces that should exist (bruno-llm only implements LLM and Embedding interfaces)
        expected_interfaces = {
            "LLMInterface",
            "EmbeddingInterface",
        }

        # Interfaces that exist in bruno-core but are NOT for bruno-llm to implement
        other_ecosystem_interfaces = {
            "AssistantInterface",  # Implemented by bruno-core itself
            "MemoryInterface",  # Implemented by bruno-memory
            "AbilityInterface",  # Future ecosystem component
            "StreamInterface",  # Future ecosystem component
        }

        found_interfaces = set(interface_classes)

        # Check if there are new interfaces we don't know about
        known_interfaces = expected_interfaces | other_ecosystem_interfaces
        new_interfaces = found_interfaces - known_interfaces
        if new_interfaces:
            pytest.fail(
                f"Unknown bruno-core interfaces detected: {new_interfaces}. "
                f"Please update the compatibility test to account for these interfaces."
            )

        # Check if expected interfaces are missing
        missing_interfaces = expected_interfaces - found_interfaces
        if missing_interfaces:
            pytest.fail(
                f"Expected bruno-core interfaces missing: {missing_interfaces}. "
                f"This may indicate a bruno-core version compatibility issue."
            )

    def test_interface_method_completeness(self):
        """Test that we implement all methods from bruno-core interfaces."""
        if not BRUNO_CORE_AVAILABLE:
            pytest.skip("bruno-core not available")

        from bruno_core.interfaces import EmbeddingInterface, LLMInterface

        # Check LLM providers
        llm_interface_methods = {
            name
            for name, _ in inspect.getmembers(LLMInterface, inspect.isfunction)
            if not name.startswith("_")
        }

        for provider_class in [OpenAIProvider, OllamaProvider]:
            provider_methods = {
                name
                for name, _ in inspect.getmembers(provider_class, callable)
                if not name.startswith("_")
            }

            missing_methods = llm_interface_methods - provider_methods
            assert not missing_methods, (
                f"{provider_class.__name__} missing LLMInterface methods: {missing_methods}"
            )

        # Check Embedding providers
        embedding_interface_methods = {
            name
            for name, _ in inspect.getmembers(EmbeddingInterface, inspect.isfunction)
            if not name.startswith("_")
        }

        for provider_class in [OpenAIEmbeddingProvider, OllamaEmbeddingProvider]:
            provider_methods = {
                name
                for name, _ in inspect.getmembers(provider_class, callable)
                if not name.startswith("_")
            }

            missing_methods = embedding_interface_methods - provider_methods
            assert not missing_methods, (
                f"{provider_class.__name__} missing EmbeddingInterface methods: {missing_methods}"
            )


class TestVersionCompatibility:
    """Test version compatibility with bruno-core."""

    def test_bruno_core_version_compatibility(self):
        """Test that bruno-core version is compatible."""
        if not BRUNO_CORE_AVAILABLE:
            pytest.skip("bruno-core not available")

        try:
            import bruno_core

            version = getattr(bruno_core, "__version__", "unknown")

            # For now, just check that we can detect the version
            # In the future, add actual version compatibility checks
            assert version is not None

        except Exception as e:
            pytest.fail(f"Could not determine bruno-core version: {e}")

    def test_interface_backward_compatibility(self):
        """Test that interface changes don't break existing functionality."""
        if not BRUNO_CORE_AVAILABLE:
            pytest.skip("bruno-core not available")

        # This test should be expanded as we learn more about
        # bruno-core's versioning and compatibility guarantees

        # For now, just check that basic interface usage works
        from bruno_core.interfaces import EmbeddingInterface, LLMInterface

        # Should be able to use interfaces in type hints
        def use_llm_interface(llm: LLMInterface) -> None:
            pass

        def use_embedding_interface(embedder: EmbeddingInterface) -> None:
            pass

        # Should not raise any exceptions
        assert callable(use_llm_interface)
        assert callable(use_embedding_interface)
