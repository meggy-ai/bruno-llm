"""Ollama provider implementation."""

import asyncio
import json
from typing import AsyncIterator, Dict, Any, List, Optional

import httpx

from bruno_core.interfaces import LLMInterface
from bruno_core.models import Message, MessageRole

from bruno_llm.base import BaseProvider
from bruno_llm.base.token_counter import SimpleTokenCounter
from bruno_llm.exceptions import (
    LLMError,
    ModelNotFoundError,
    TimeoutError as LLMTimeoutError,
    StreamError,
    InvalidResponseError,
)
from bruno_llm.providers.ollama.config import OllamaConfig


class OllamaProvider(BaseProvider, LLMInterface):
    """
    Ollama provider for local LLM inference.
    
    Ollama runs models locally without API keys. Requires Ollama
    to be installed and running on the specified base_url.
    
    Args:
        base_url: Ollama API endpoint (default: http://localhost:11434)
        model: Model name (default: llama2)
        timeout: Request timeout in seconds (default: 30.0)
        **kwargs: Additional configuration parameters
    
    Examples:
        >>> provider = OllamaProvider(model="llama2")
        >>> response = await provider.generate([
        ...     Message(role=MessageRole.USER, content="Hello")
        ... ])
        
        >>> # Streaming
        >>> async for chunk in provider.stream([
        ...     Message(role=MessageRole.USER, content="Tell me a story")
        ... ]):
        ...     print(chunk, end="")
        
    See Also:
        - https://ollama.ai/
        - bruno-core LLMInterface documentation
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: float = 30.0,
        **kwargs: Any
    ):
        """Initialize Ollama provider."""
        # Create config
        config = OllamaConfig(
            base_url=base_url,
            model=model,
            timeout=timeout,
            **kwargs
        )
        
        # Initialize base provider
        super().__init__(
            provider_name="ollama",
            max_retries=3,
            timeout=timeout,
        )
        
        # Store config
        self._config = config
        self._model = config.model
        
        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout),
        )
        
        # Token counter (simple estimation for local models)
        self._token_counter = SimpleTokenCounter()
        
    @property
    def model(self) -> str:
        """Get current model name."""
        return self._model
    
    @property
    def config(self) -> OllamaConfig:
        """Get provider configuration."""
        return self._config
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert bruno-core messages to Ollama format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
    def _build_request(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Build Ollama API request."""
        request = {
            "model": self._model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }
        
        # Add optional parameters
        options: Dict[str, Any] = {}
        
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            options["top_p"] = self._config.top_p
        if self._config.top_k is not None:
            options["top_k"] = self._config.top_k
        if self._config.num_predict is not None:
            options["num_predict"] = self._config.num_predict
        if self._config.stop is not None:
            request["stop"] = self._config.stop
        
        # Override with kwargs
        options.update(kwargs)
        
        if options:
            request["options"] = options
        
        return request
    
    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        """
        Generate a complete response from Ollama.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            request = self._build_request(messages, stream=False, **kwargs)
            
            response = await self._client.post(
                "/api/chat",
                json=request,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response content
            if "message" not in data:
                raise InvalidResponseError("No 'message' in response")
            
            content = data["message"].get("content", "")
            return content
            
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{self._model}' not found. "
                    f"Run 'ollama pull {self._model}' to download it."
                ) from e
            raise LLMError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to connect to Ollama at {self._config.base_url}. "
                f"Make sure Ollama is running: {e}"
            ) from e
        except (KeyError, json.JSONDecodeError) as e:
            raise InvalidResponseError(f"Invalid response format: {e}") from e
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from Ollama.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Yields:
            Response text chunks
            
        Raises:
            StreamError: If streaming fails
        """
        try:
            request = self._build_request(messages, stream=True, **kwargs)
            
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=request,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Check if streaming is done
                        if data.get("done", False):
                            break
                        
                        # Extract content chunk
                        if "message" in data:
                            content = data["message"].get("content", "")
                            if content:
                                yield content
                                
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
                        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{self._model}' not found. "
                    f"Run 'ollama pull {self._model}' to download it."
                ) from e
            raise StreamError(f"Streaming failed: {e}") from e
        except httpx.RequestError as e:
            raise StreamError(
                f"Failed to connect to Ollama: {e}"
            ) from e
        except Exception as e:
            raise StreamError(f"Unexpected streaming error: {e}") from e
    
    async def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
            
        Raises:
            LLMError: If request fails
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            return [model["name"] for model in models]
            
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to list models: {e}"
            ) from e
        except (KeyError, json.JSONDecodeError) as e:
            raise InvalidResponseError(
                f"Invalid response format: {e}"
            ) from e
    
    async def check_connection(self) -> bool:
        """
        Check if Ollama is accessible.
        
        Returns:
            True if Ollama is running and accessible
        """
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        return self._token_counter.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            "provider": "ollama",
            "model": self._model,
            "base_url": self._config.base_url,
            "temperature": self._config.temperature,
            "max_tokens": self._config.num_predict,
        }
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self._client.aclose()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
