"""
Provider Abstraction Layer

This package provides a unified interface for interacting with multiple AI
providers. It includes abstract base classes, data models, and error handling
for seamless multi-provider orchestration.

Core Components:
- Provider: Abstract base class for provider implementations
- ProviderRequest: Standardized request model
- ProviderResponse: Standardized response model
- ProviderError: Exception hierarchy for error handling
- RateLimitConfig: Rate limiting configuration
- RetryConfig: Retry logic configuration

Usage Example:
    ```python
    from quorum_mcp.providers import (
        Provider,
        ProviderRequest,
        ProviderResponse,
        ProviderError,
    )

    # Create a request
    request = ProviderRequest(
        query="What is the capital of France?",
        system_prompt="You are a helpful assistant.",
        max_tokens=1000,
        temperature=0.7,
    )

    # Send to provider (assuming provider is implemented)
    try:
        response = await provider.send_request(request)
        print(response.content)
    except ProviderError as e:
        print(f"Error: {e}")
    ```
"""

from quorum_mcp.providers.anthropic_provider import AnthropicProvider
from quorum_mcp.providers.base import (
    # Base classes
    Provider,
    ProviderAuthenticationError,
    ProviderConnectionError,
    # Error hierarchy
    ProviderError,
    ProviderInvalidRequestError,
    ProviderModelError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    # Data models
    ProviderRequest,
    ProviderResponse,
    ProviderTimeoutError,
    ProviderType,
    # Configuration models
    RateLimitConfig,
    RetryConfig,
)
from quorum_mcp.providers.openai_provider import OpenAIProvider

__all__ = [
    # Base classes
    "Provider",
    # Provider implementations
    "AnthropicProvider",
    "OpenAIProvider",
    # Data models
    "ProviderRequest",
    "ProviderResponse",
    "ProviderType",
    # Configuration models
    "RateLimitConfig",
    "RetryConfig",
    # Error hierarchy
    "ProviderError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderConnectionError",
    "ProviderInvalidRequestError",
    "ProviderModelError",
    "ProviderQuotaExceededError",
]

# Version information
__version__ = "0.1.0"
