"""Configuration for Ollama provider."""

from pydantic import BaseModel, Field
from typing import Optional


class OllamaConfig(BaseModel):
    """Configuration for Ollama provider.
    
    Ollama runs models locally without API keys. Requires Ollama
    to be installed and running on the specified base_url.
    
    Args:
        base_url: Ollama API endpoint (default: http://localhost:11434)
        model: Model name (default: llama2)
        timeout: Request timeout in seconds (default: 30.0)
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        top_p: Nucleus sampling threshold (default: 0.9)
        top_k: Top-k sampling parameter (default: 40)
        num_predict: Maximum tokens to generate (default: None for unlimited)
        stop: Stop sequences (default: None)
        
    Examples:
        >>> config = OllamaConfig(model="llama2")
        >>> config = OllamaConfig(
        ...     base_url="http://192.168.1.100:11434",
        ...     model="mistral",
        ...     temperature=0.5
        ... )
    """
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    model: str = Field(
        default="llama2",
        description="Model name to use"
    )
    timeout: float = Field(
        default=30.0,
        ge=0.0,
        description="Request timeout in seconds"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling parameter"
    )
    num_predict: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )
    stop: Optional[list[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    
    model_config = {"frozen": True}  # Immutable after creation
