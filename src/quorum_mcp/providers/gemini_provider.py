"""
Google Gemini Provider Implementation

This module provides integration with Google's Gemini AI models via the
google-genai SDK (unified Gen AI Python SDK).

Supported Models:
- gemini-2.5-flash (default) - Fast, cost-effective
- gemini-2.5-pro - Advanced reasoning
- gemini-1.5-pro - Stable, 2M context window
- gemini-1.5-flash - Fast, legacy

Features:
- Async/await support via client.aio
- Token counting via count_tokens
- Cost tracking based on usage metadata
- Error mapping to Provider exceptions
"""

import asyncio
import os
from typing import Any

from google import genai
from google.genai import types

from quorum_mcp.providers.base import (
    Provider,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderModelError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeoutError,
    RateLimitConfig,
    RetryConfig,
)


class GeminiProvider(Provider):
    """
    Google Gemini AI Provider.

    Integrates with Google's Gemini models via the google-genai SDK.
    """

    # Model pricing (per 1M tokens)
    MODEL_PRICING = {
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    # Context windows
    CONTEXT_WINDOWS = {
        "gemini-2.5-flash": 200000,
        "gemini-2.5-pro": 200000,
        "gemini-1.5-pro": 2000000,  # 2M tokens - largest available
        "gemini-1.5-flash": 1000000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env var)
            model: Model identifier (default: gemini-2.5-flash)
            vertexai: Use Vertex AI instead of Gemini Developer API
            project: GCP project ID (required for Vertex AI)
            location: GCP location (required for Vertex AI)
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Raises:
            ProviderAuthenticationError: If API key is missing
        """
        # Get API key from parameter or environment (may be None for Vertex AI)
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")

        # Store vertex AI settings before validation
        self.vertexai = vertexai
        self.project = project
        self.location = location

        # Validate configuration
        if not vertexai and not resolved_api_key:
            raise ProviderAuthenticationError(
                "GOOGLE_API_KEY environment variable not set | Provider: gemini"
            )

        if vertexai and (not self.project or not self.location):
            raise ProviderAuthenticationError(
                "Vertex AI requires project and location | Provider: gemini"
            )

        # Initialize base class (use "vertex-ai" as api_key if using Vertex AI)
        super().__init__(
            api_key=resolved_api_key if not vertexai else "vertex-ai",
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        # Initialize client
        try:
            if vertexai:
                self.client = genai.Client(
                    vertexai=True, project=self.project, location=self.location
                )
            else:
                self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ProviderAuthenticationError(f"Failed to initialize Gemini client: {e}")

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send request to Gemini and return structured response.

        Args:
            request: Provider request with prompt, context, and parameters

        Returns:
            ProviderResponse with content, tokens, and cost

        Raises:
            Various ProviderError subclasses for different error conditions
        """
        try:
            # Determine model (from request or use instance default)
            model = request.model or self.model

            # Build contents (query + context)
            if request.context:
                contents = f"{request.context}\n\n{request.query}"
            else:
                contents = request.query

            # Build configuration
            config_dict: dict[str, Any] = {}

            if request.max_tokens:
                config_dict["max_output_tokens"] = request.max_tokens

            if request.temperature is not None:
                config_dict["temperature"] = request.temperature

            if request.system_prompt:
                config_dict["system_instruction"] = request.system_prompt

            # Create config object if we have settings
            config = types.GenerateContentConfig(**config_dict) if config_dict else None

            # Call async API
            response = await self.client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )

            # Extract content
            if not response.text:
                raise ProviderError("Empty response from Gemini")

            content = response.text

            # Extract token usage
            tokens_input = response.usage_metadata.prompt_token_count
            tokens_output = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

            # Calculate cost
            cost = self.get_cost(tokens_input, tokens_output)

            # Build metadata
            metadata = {
                "model": model,
                "finish_reason": (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else None
                ),
                "total_tokens": total_tokens,
            }

            return ProviderResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                metadata=metadata,
            )

        except asyncio.TimeoutError as e:
            raise ProviderTimeoutError(f"Request timed out | Provider: gemini") from e

        except Exception as e:
            # Map Gemini-specific errors to Provider errors
            error_message = str(e).lower()

            if "api key" in error_message or "authentication" in error_message:
                raise ProviderAuthenticationError(
                    f"Authentication failed: {e} | Provider: gemini"
                ) from e

            if "quota" in error_message or "resource exhausted" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {e} | Provider: gemini"
                ) from e

            if "rate limit" in error_message or "too many requests" in error_message:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded: {e} | Provider: gemini"
                ) from e

            if "invalid" in error_message or "not found" in error_message:
                if "model" in error_message:
                    raise ProviderModelError(
                        f"Invalid model: {e} | Provider: gemini"
                    ) from e
                raise ProviderInvalidRequestError(
                    f"Invalid request: {e} | Provider: gemini"
                ) from e

            if "connection" in error_message or "network" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {e} | Provider: gemini"
                ) from e

            # Generic error
            raise ProviderError(f"Gemini API error: {e}") from e

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Gemini's token counting API.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            response = await self.client.aio.models.count_tokens(
                model=self.model, contents=text
            )
            return response.total_tokens

        except Exception as e:
            # Fallback: estimate ~4 chars per token
            return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate cost based on token usage.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing for current model (or use default)
        pricing = self.MODEL_PRICING.get(
            self.model, self.MODEL_PRICING["gemini-2.5-flash"]
        )

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """Get provider identifier."""
        return "gemini"

    def get_model_info(self) -> dict:
        """
        Get information about the current model.

        Returns:
            Dictionary with model details
        """
        pricing = self.MODEL_PRICING.get(
            self.model, self.MODEL_PRICING["gemini-2.5-flash"]
        )
        context_window = self.CONTEXT_WINDOWS.get(
            self.model, self.CONTEXT_WINDOWS["gemini-2.5-flash"]
        )

        return {
            "provider": self.get_provider_name(),
            "model": self.model,
            "context_window": context_window,
            "pricing": pricing,
            "vertexai": self.vertexai,
        }

    @staticmethod
    def list_available_models() -> list[str]:
        """
        List available Gemini models.

        Returns:
            List of model identifiers
        """
        return list(GeminiProvider.MODEL_PRICING.keys())

    async def aclose(self) -> None:
        """
        Close the async client and release resources.

        Should be called when done using the provider.
        """
        try:
            await self.client.aio.aclose()
        except Exception:
            pass  # Ignore errors during cleanup
