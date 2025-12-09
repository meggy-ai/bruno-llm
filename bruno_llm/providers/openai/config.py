"""Configuration for OpenAI provider."""

from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI provider.

    OpenAI provides GPT models via API. Requires an API key from
    https://platform.openai.com/api-keys

    Args:
        api_key: OpenAI API key (required)
        model: Model name (default: gpt-4)
        organization: Organization ID (optional)
        base_url: API base URL (default: https://api.openai.com/v1)
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum retry attempts (default: 3)
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        top_p: Nucleus sampling threshold (default: 1.0)
        max_tokens: Maximum tokens to generate (default: None)
        presence_penalty: Presence penalty -2.0 to 2.0 (default: 0.0)
        frequency_penalty: Frequency penalty -2.0 to 2.0 (default: 0.0)
        stop: Stop sequences (default: None)

    Examples:
        >>> config = OpenAIConfig(api_key="sk-...", model="gpt-4")
        >>> config = OpenAIConfig(
        ...     api_key="sk-...",
        ...     model="gpt-3.5-turbo",
        ...     temperature=0.5,
        ...     max_tokens=1000
        ... )
    """

    api_key: SecretStr = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Model name to use")
    organization: Optional[str] = Field(default=None, description="Organization ID")
    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    timeout: float = Field(default=30.0, ge=0.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")

    model_config = {"frozen": True}  # Immutable after creation
