"""
bruno-llm: LLM provider implementations for bruno-core.

This package provides production-ready LLM provider implementations that
integrate seamlessly with the bruno-core framework. All providers implement
the LLMInterface contract from bruno-core.

Available Providers:
    - OllamaProvider: Local LLM inference via Ollama
    - OpenAIProvider: OpenAI GPT models

Example:
    >>> from bruno_llm.providers.ollama import OllamaProvider
    >>> llm = OllamaProvider(model="llama2")
    >>> response = await llm.generate([Message(role="user", content="Hello")])
    
    >>> from bruno_llm.providers.openai import OpenAIProvider
    >>> llm = OpenAIProvider(api_key="sk-...", model="gpt-4")
    >>> response = await llm.generate([Message(role="user", content="Hello")])
"""

from bruno_llm.__version__ import (
    __version__,
    __author__,
    __email__,
    __license__,
    __description__,
)

# Public API will be populated as providers are implemented
from bruno_llm.factory import LLMFactory

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "LLMFactory",
]
