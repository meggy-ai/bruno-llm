"""OpenAI provider implementation."""

from typing import AsyncIterator, Dict, Any, List, Optional

from openai import (
    AsyncOpenAI,
    OpenAIError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
    NotFoundError as OpenAINotFoundError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from bruno_core.interfaces import LLMInterface
from bruno_core.models import Message, MessageRole

from bruno_llm.base import BaseProvider, CostTracker, PRICING_OPENAI
from bruno_llm.base.token_counter import create_token_counter
from bruno_llm.exceptions import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    TimeoutError as LLMTimeoutError,
    StreamError,
    InvalidResponseError,
)
from bruno_llm.providers.openai.config import OpenAIConfig


class OpenAIProvider(BaseProvider, LLMInterface):
    """
    OpenAI provider for GPT models.
    
    Provides access to OpenAI's GPT models (GPT-4, GPT-3.5-turbo, etc.)
    via the official OpenAI API. Requires an API key.
    
    Args:
        api_key: OpenAI API key (required)
        model: Model name (default: gpt-4)
        organization: Organization ID (optional)
        timeout: Request timeout in seconds (default: 30.0)
        **kwargs: Additional configuration parameters
    
    Examples:
        >>> provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
        >>> response = await provider.generate([
        ...     Message(role=MessageRole.USER, content="Hello")
        ... ])
        
        >>> # Streaming
        >>> async for chunk in provider.stream([
        ...     Message(role=MessageRole.USER, content="Tell me a story")
        ... ]):
        ...     print(chunk, end="")
        
        >>> # With cost tracking
        >>> provider = OpenAIProvider(api_key="sk-...", track_cost=True)
        >>> await provider.generate([...])
        >>> report = provider.cost_tracker.get_usage_report()
        
    See Also:
        - https://platform.openai.com/docs/api-reference
        - bruno-core LLMInterface documentation
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        organization: Optional[str] = None,
        timeout: float = 30.0,
        track_cost: bool = True,
        **kwargs: Any
    ):
        """Initialize OpenAI provider."""
        # Create config
        config = OpenAIConfig(
            api_key=api_key,
            model=model,
            organization=organization,
            timeout=timeout,
            **kwargs
        )
        
        # Initialize base provider
        super().__init__(
            provider_name="openai",
            max_retries=config.max_retries,
            timeout=timeout,
        )
        
        # Store config
        self._config = config
        self._model = config.model
        
        # Create OpenAI client
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            organization=config.organization,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,  # We handle retries in BaseProvider
        )
        
        # Token counter (tiktoken for accurate counting)
        self._token_counter = create_token_counter("openai", model=model)
        
        # Cost tracker
        self._track_cost = track_cost
        if track_cost:
            self.cost_tracker = CostTracker(
                provider_name="openai",
                pricing=PRICING_OPENAI,
            )
    
    @property
    def model(self) -> str:
        """Get current model name."""
        return self._model
    
    @property
    def config(self) -> OpenAIConfig:
        """Get provider configuration."""
        return self._config
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert bruno-core messages to OpenAI format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
    def _build_request_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Build OpenAI API request parameters."""
        params: Dict[str, Any] = {
            "model": self._model,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
        }
        
        # Add optional parameters
        if self._config.max_tokens is not None:
            params["max_tokens"] = self._config.max_tokens
        if self._config.presence_penalty != 0.0:
            params["presence_penalty"] = self._config.presence_penalty
        if self._config.frequency_penalty != 0.0:
            params["frequency_penalty"] = self._config.frequency_penalty
        if self._config.stop is not None:
            params["stop"] = self._config.stop
        
        # Override with kwargs
        params.update(kwargs)
        
        return params
    
    def _track_usage(
        self,
        messages: List[Message],
        response_text: str,
        completion: Optional[ChatCompletion] = None
    ) -> None:
        """Track token usage and costs."""
        if not self._track_cost:
            return
        
        # Get token counts from response or estimate
        if completion and hasattr(completion, 'usage') and completion.usage:
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
        else:
            # Estimate if usage not available
            input_text = " ".join(msg.content for msg in messages)
            input_tokens = self._token_counter.count_tokens(input_text)
            output_tokens = self._token_counter.count_tokens(response_text)
        
        # Track in cost tracker
        self.cost_tracker.track_request(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    
    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        """
        Generate a complete response from OpenAI.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
            LLMTimeoutError: If request times out
            LLMError: For other API errors
        """
        try:
            params = self._build_request_params(**kwargs)
            
            completion: ChatCompletion = await self._client.chat.completions.create(
                messages=self._format_messages(messages),
                **params
            )
            
            # Extract response content
            if not completion.choices:
                raise InvalidResponseError("No choices in response")
            
            content = completion.choices[0].message.content or ""
            
            # Track usage
            self._track_usage(messages, content, completion)
            
            return content
            
        except APITimeoutError as e:
            raise LLMTimeoutError(f"Request timed out: {e}") from e
        except APIConnectionError as e:
            raise LLMError(f"Connection error: {e}") from e
        except OpenAIError as e:
            # Parse OpenAI-specific errors by exception type
            if isinstance(e, OpenAIAuthError):
                raise AuthenticationError(f"Invalid API key: {e}") from e
            elif isinstance(e, OpenAIRateLimitError):
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            elif isinstance(e, OpenAINotFoundError):
                raise ModelNotFoundError(f"Model '{self._model}' not found: {e}") from e
            else:
                raise LLMError(f"OpenAI API error: {e}") from e
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from OpenAI.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Yields:
            Response text chunks
            
        Raises:
            StreamError: If streaming fails
        """
        try:
            params = self._build_request_params(stream=True, **kwargs)
            
            stream = await self._client.chat.completions.create(
                messages=self._format_messages(messages),
                **params
            )
            
            full_response = []
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                content = delta.content
                
                if content:
                    full_response.append(content)
                    yield content
            
            # Track usage after streaming completes
            response_text = "".join(full_response)
            self._track_usage(messages, response_text)
            
        except APITimeoutError as e:
            raise StreamError(f"Stream timed out: {e}") from e
        except APIConnectionError as e:
            raise StreamError(f"Connection error: {e}") from e
        except OpenAIError as e:
            # Parse OpenAI-specific errors by exception type
            if isinstance(e, OpenAIAuthError):
                raise AuthenticationError(f"Invalid API key: {e}") from e
            elif isinstance(e, OpenAIRateLimitError):
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            elif isinstance(e, OpenAINotFoundError):
                raise ModelNotFoundError(f"Model '{self._model}' not found: {e}") from e
            else:
                raise StreamError(f"Streaming failed: {e}") from e
        except Exception as e:
            raise StreamError(f"Unexpected streaming error: {e}") from e
    
    async def list_models(self) -> List[str]:
        """
        List available OpenAI models.
        
        Returns:
            List of model IDs
            
        Raises:
            LLMError: If request fails
        """
        try:
            models = await self._client.models.list()
            return [model.id for model in models.data]
        except OpenAIError as e:
            raise LLMError(f"Failed to list models: {e}") from e
    
    async def check_connection(self) -> bool:
        """
        Check if OpenAI API is accessible.
        
        Returns:
            True if API is accessible with valid credentials
        """
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
    
    def get_token_count(self, text: str) -> int:
        """
        Get accurate token count for text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Exact token count
        """
        return self._token_counter.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            "provider": "openai",
            "model": self._model,
            "base_url": self._config.base_url,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        
        # Add cost tracking info if enabled
        if self._track_cost:
            info["cost_tracking"] = {
                "enabled": True,
                "total_cost": self.cost_tracker.get_total_cost(),
                "total_requests": self.cost_tracker.get_request_count(),
            }
        
        return info
    
    async def close(self) -> None:
        """Close OpenAI client and cleanup resources."""
        await self._client.close()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
