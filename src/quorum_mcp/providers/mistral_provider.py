"""
Mistral AI Provider Implementation for Quorum-MCP

This module implements the Provider interface for Mistral AI's API.
It provides async API integration with proper authentication, token counting,
and comprehensive error handling.

Key Features:
- Support for all Mistral AI models (Large, Medium, Code, Pixtral)
- Accurate token counting
- Built-in cost calculation per model
- Comprehensive error mapping
- Async/await throughout

Supported Models (as of 2025):
- mistral-large-latest: Latest Mistral Large model
- mistral-medium-latest: Latest Mistral Medium model
- codestral-latest: Code generation optimized
- pixtral-large-latest: Multimodal (vision) model
- mistral-small-latest: Smaller, faster model
"""

import os
import time
from typing import Any

from mistralai import Mistral
from mistralai.models import (
    ChatCompletionResponse,
    SDKError,
    HTTPValidationError,
)

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


class MistralProvider(Provider):
    """
    Mistral AI provider implementation.

    This provider handles communication with Mistral AI's API, including:
    - Message formatting for chat completions
    - Token counting (estimated)
    - Cost calculation based on model-specific pricing
    - Error handling and mapping to provider exceptions

    Supported Models:
    - mistral-large-latest (flagship model)
    - mistral-medium-latest (balanced performance)
    - codestral-latest (code generation)
    - pixtral-large-latest (vision capabilities)
    - mistral-small-latest (fast and efficient)

    Pricing (per 1M tokens, as of 2025):
    - Large models: $2.00 input / $6.00 output
    - Medium models: $0.40 input / $2.00 output
    - Code models: $0.30 input / $0.90 output
    - Small models: $0.20 input / $0.60 output
    """

    # Model pricing in dollars per 1M tokens
    MODEL_PRICING = {
        "mistral-large-latest": {"input": 2.0, "output": 6.0},
        "mistral-large-2411": {"input": 2.0, "output": 6.0},
        "pixtral-large-latest": {"input": 2.0, "output": 6.0},
        "pixtral-large-2411": {"input": 2.0, "output": 6.0},
        "mistral-medium-latest": {"input": 0.4, "output": 2.0},
        "mistral-medium-3": {"input": 0.4, "output": 2.0},
        "devstral-medium": {"input": 0.4, "output": 2.0},
        "codestral-latest": {"input": 0.3, "output": 0.9},
        "codestral-2508": {"input": 0.3, "output": 0.9},
        "mistral-small-latest": {"input": 0.2, "output": 0.6},
        "mistral-small-2409": {"input": 0.2, "output": 0.6},
    }

    # Model context windows
    MODEL_CONTEXT = {
        "mistral-large-latest": 128000,
        "mistral-large-2411": 128000,
        "pixtral-large-latest": 128000,
        "pixtral-large-2411": 128000,
        "mistral-medium-latest": 32000,
        "mistral-medium-3": 32000,
        "devstral-medium": 32000,
        "codestral-latest": 256000,
        "codestral-2508": 256000,
        "mistral-small-latest": 32000,
        "mistral-small-2409": 32000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-large-latest",
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize the Mistral AI provider.

        Args:
            api_key: Mistral AI API key. If None, reads from MISTRAL_API_KEY env var.
            model: Default model to use (default: mistral-large-latest)
            rate_limit_config: Configuration for rate limiting
            retry_config: Configuration for retry logic

        Raises:
            ProviderAuthenticationError: If no API key is provided or found
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise ProviderAuthenticationError(
                "Mistral AI API key not provided. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="mistral",
            )

        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to Mistral AI's API and return the response.

        This method:
        1. Validates the request
        2. Formats it for Mistral's chat completion API
        3. Makes the API call with error handling
        4. Parses the response and extracts token counts
        5. Calculates costs based on usage

        Args:
            request: The standardized request to send

        Returns:
            The standardized response from Mistral AI

        Raises:
            ProviderError: If the request fails for any reason
        """
        # Validate request
        await self.validate_request(request)

        # Check rate limits
        await self.check_rate_limits()

        # Use request model or fall back to provider default
        model = request.model or self.model

        # Format messages using helper method
        messages = self._format_messages(request)

        # Prepare API call parameters
        api_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Add optional parameters
        if request.top_p is not None:
            api_params["top_p"] = request.top_p

        # Make the API call with timing
        start_time = time.time()

        try:
            response: ChatCompletionResponse = await self.client.chat.complete_async(
                **api_params,
            )
        except HTTPValidationError as e:
            # Validation errors (bad request, invalid model, etc.)
            error_detail = str(e)
            if "model" in error_detail.lower():
                raise ProviderModelError(
                    f"Invalid model: {error_detail}",
                    provider="mistral",
                    original_error=e,
                ) from e
            else:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {error_detail}",
                    provider="mistral",
                    original_error=e,
                ) from e
        except SDKError as e:
            # SDK errors (authentication, rate limit, connection, etc.)
            error_message = str(e).lower()

            if "authentication" in error_message or "api key" in error_message or "401" in error_message:
                raise ProviderAuthenticationError(
                    f"Authentication failed: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
            elif "rate limit" in error_message or "429" in error_message:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
            elif "timeout" in error_message:
                raise ProviderTimeoutError(
                    f"Request timed out: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
            elif "connection" in error_message or "network" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
            elif "quota" in error_message or "insufficient" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
            else:
                raise ProviderError(
                    f"SDK error: {str(e)}",
                    provider="mistral",
                    original_error=e,
                ) from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected error: {str(e)}",
                provider="mistral",
                original_error=e,
            ) from e

        end_time = time.time()
        latency = end_time - start_time

        # Extract response content
        if not response.choices or len(response.choices) == 0:
            raise ProviderError(
                "No choices returned in response",
                provider="mistral",
            )

        content = response.choices[0].message.content or ""

        # Extract token counts from usage
        tokens_input = response.usage.prompt_tokens if response.usage else 0
        tokens_output = response.usage.completion_tokens if response.usage else 0

        # Calculate cost
        cost = self.get_cost(tokens_input, tokens_output)

        # Build metadata
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model,
            "response_id": response.id,
        }

        # Create and return standardized response
        return ProviderResponse(
            content=content,
            model=model,
            provider=self.get_provider_name(),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            latency=latency,
            metadata=metadata,
        )

    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.

        Note: Mistral doesn't provide a native token counting API, so we use
        a rough estimation of 4 characters per token (similar to tiktoken).

        Args:
            text: The text to tokenize

        Returns:
            Estimated number of tokens in the text
        """
        # Rough estimation: ~4 characters per token
        # This matches the general rule that 1 token ≈ 0.75 words ≈ 4 chars
        return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate the cost of an API call based on token usage.

        Uses model-specific pricing from MODEL_PRICING. If the model
        is not found, falls back to mistral-large-latest pricing.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing for the current model, or fall back to large model
        pricing = self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING["mistral-large-latest"],
        )

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            "mistral"
        """
        return "mistral"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model metadata including:
            - model: Model identifier
            - provider: Provider name
            - context_window: Maximum context length in tokens
            - pricing: Pricing information per 1M tokens
        """
        pricing = self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING["mistral-large-latest"],
        )

        context_window = self.MODEL_CONTEXT.get(
            self.model,
            self.MODEL_CONTEXT["mistral-large-latest"],
        )

        return {
            "provider": "mistral",
            "model": self.model,
            "context_window": context_window,
            "pricing": {
                "input": pricing["input"],
                "output": pricing["output"],
            },
        }

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List all available Mistral AI models.

        Returns:
            List of model identifiers
        """
        return list(cls.MODEL_PRICING.keys())

    def _validate_model(self, model: str | None) -> None:
        """
        Validate that a model is supported.

        Args:
            model: Model identifier to validate, or None for default

        Raises:
            ProviderModelError: If model is not supported
        """
        # None is valid (uses default)
        if model is None:
            return

        # Check if model is in supported models
        if model not in self.MODEL_PRICING:
            raise ProviderModelError(
                f"Unsupported model: {model}. Available models: {self.list_available_models()}",
                provider=self.get_provider_name(),
            )

    def _format_messages(self, request: ProviderRequest) -> list[dict[str, str]]:
        """
        Format messages for Mistral AI API.

        Args:
            request: The provider request

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Build user message with context if provided
        user_content = request.query
        if request.context:
            user_content = f"{request.context}\n\n{request.query}"

        messages.append({"role": "user", "content": user_content})

        return messages
