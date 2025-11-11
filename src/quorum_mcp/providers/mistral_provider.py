"""
Mistral AI Provider for Quorum-MCP

This module implements the Mistral AI provider, supporting high-performance LLM inference
with the best pricing in the market and excellent quality-to-cost ratio.

Key Features:
- Best-in-class pricing ($0.27-2.70 per million tokens)
- Strong performance competitive with GPT-4 and Claude
- European AI sovereignty (GDPR compliant)
- Multiple model sizes (7B to Mixtral 8x22B)
- OpenAI-compatible API for easy migration
"""

import logging
import os
from datetime import datetime, timezone

from mistralai import Mistral

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

logger = logging.getLogger(__name__)


class MistralProvider(Provider):
    """
    Provider implementation for Mistral AI.

    Mistral AI offers best-in-class pricing with strong performance,
    making it ideal for cost-sensitive deployments and European customers.
    """

    DEFAULT_MODEL = "mistral-large-latest"

    # Model pricing (per million tokens) - Based on Mistral AI's 2025 pricing
    MODEL_PRICING = {
        # Large models (Most capable)
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-large-2411": {"input": 2.00, "output": 6.00},
        # Medium models (Balanced)
        "mistral-medium-latest": {"input": 2.70, "output": 8.10},
        # Small models (Fast and economical)
        "mistral-small-latest": {"input": 0.20, "output": 0.60},
        "mistral-small-2412": {"input": 0.20, "output": 0.60},
        # Mixtral models (Open-weight)
        "open-mixtral-8x22b": {"input": 2.00, "output": 6.00},
        "open-mixtral-8x7b": {"input": 0.70, "output": 0.70},
        # Ministral models (Edge optimized)
        "ministral-8b-latest": {"input": 0.10, "output": 0.10},
        "ministral-3b-latest": {"input": 0.04, "output": 0.04},
        # Codestral (Code-specialized)
        "codestral-latest": {"input": 0.20, "output": 0.60},
        # Default for unknown models
        "default": {"input": 1.00, "output": 3.00},
    }

    # Context windows for models
    MODEL_CONTEXT_WINDOWS = {
        "mistral-large-latest": 128000,
        "mistral-large-2411": 128000,
        "mistral-medium-latest": 32000,
        "mistral-small-latest": 32000,
        "mistral-small-2412": 32000,
        "open-mixtral-8x22b": 64000,
        "open-mixtral-8x7b": 32000,
        "ministral-8b-latest": 128000,
        "ministral-3b-latest": 128000,
        "codestral-latest": 32000,
        "default": 32000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Mistral AI provider.

        Args:
            api_key: Mistral AI API key (or from MISTRAL_API_KEY env var)
            model: Model identifier (default: mistral-large-latest)
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Raises:
            ProviderAuthenticationError: If API key is not provided and not in environment
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ProviderAuthenticationError(
                    "MISTRAL_API_KEY environment variable not set",
                    provider="mistral",
                )

        # Use default model if not specified
        if model is None:
            model = self.DEFAULT_MODEL

        # Initialize base class
        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        # Initialize Mistral client
        self.client = Mistral(api_key=api_key)

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to Mistral AI and return the response.

        This method handles:
        - Request validation
        - Rate limiting checks
        - Request formatting for Mistral AI API
        - API call with retry logic
        - Response parsing and token counting
        - Error mapping to Provider exceptions

        Args:
            request: Provider request with prompt, context, and parameters

        Returns:
            ProviderResponse with content, tokens, and cost

        Raises:
            Various ProviderError subclasses for different error conditions
        """
        try:
            # Check rate limits
            await self.check_rate_limits()

            # Determine model (from request or use instance default)
            model = request.model or self.model

            # Build messages for chat API
            messages = []

            # Add system prompt if provided
            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })

            # Build user message (context + query)
            user_content = request.query
            if request.context:
                user_content = f"{request.context}\n\n{request.query}"

            messages.append({
                "role": "user",
                "content": user_content
            })

            # Start timing
            start_time = datetime.now(timezone.utc)

            # Make API call using Mistral's async method
            response = await self.client.chat.complete_async(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            # Calculate latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Extract response content
            if not response.choices or len(response.choices) == 0:
                raise ProviderError(
                    "No choices returned in response",
                    provider="mistral",
                )

            content = response.choices[0].message.content or ""

            # Get token counts from response
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

            # Create response object
            return ProviderResponse(
                content=content,
                model=model,
                provider=self.get_provider_name(),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                latency=latency,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata,
            )

        except Exception as e:
            # Map Mistral AI exceptions to Provider exceptions
            error_message = str(e).lower()

            # Try to get status code if available
            status_code = 0
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'http_status'):
                status_code = e.http_status

            if status_code == 401 or "authentication" in error_message or "api key" in error_message or "unauthorized" in error_message:
                raise ProviderAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="mistral",
                ) from e

            if status_code == 429 or "rate limit" in error_message or "too many requests" in error_message:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="mistral",
                ) from e

            if "timeout" in error_message or "timed out" in error_message:
                raise ProviderTimeoutError(
                    f"Request timeout: {e}",
                    provider="mistral",
                ) from e

            if "model" in error_message and ("not found" in error_message or "unavailable" in error_message):
                raise ProviderModelError(
                    f"Model not found or not available: {e}",
                    provider="mistral",
                ) from e

            if "quota" in error_message or "insufficient" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {e}",
                    provider="mistral",
                ) from e

            if status_code == 400 or "invalid" in error_message or "bad request" in error_message:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {e}",
                    provider="mistral",
                ) from e

            if "connection" in error_message or "network" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {e}",
                    provider="mistral",
                ) from e

            # Generic error for anything else
            raise ProviderError(
                f"Mistral AI request failed: {e}",
                provider="mistral",
            ) from e

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Mistral AI doesn't provide a tokenize API, so we use estimation
        based on typical token-to-character ratios.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text (estimated)
        """
        # Mistral models use similar tokenization to GPT models
        # Estimate ~4 chars per token as a reasonable approximation
        return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate cost for token usage.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Total cost in USD
        """
        # Get pricing for current model
        pricing = self.MODEL_PRICING.get(self.model, self.MODEL_PRICING["default"])

        # Calculate cost (pricing is per million tokens)
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            Provider identifier string
        """
        return "mistral"

    def get_model_info(self) -> dict:
        """
        Return information about the current model.

        Returns:
            Dictionary with model details including pricing and context window
        """
        # Get pricing and context window for current model
        pricing = self.MODEL_PRICING.get(self.model, self.MODEL_PRICING["default"])
        context_window = self.MODEL_CONTEXT_WINDOWS.get(
            self.model, self.MODEL_CONTEXT_WINDOWS["default"]
        )

        return {
            "name": self.model,
            "provider": "mistral",
            "context_window": context_window,
            "pricing": {
                "input_per_1m": pricing["input"],
                "output_per_1m": pricing["output"],
                "currency": "USD",
            },
            "features": {
                "best_pricing": True,
                "european_ai": True,
                "gdpr_compliant": True,
            },
        }

    async def aclose(self) -> None:
        """
        Close the async client and release resources.

        Should be called when done using the provider to properly cleanup
        HTTP connections and other resources.
        """
        try:
            # Mistral client may not have explicit close method
            # Check if it has one and call it
            if hasattr(self.client, 'close'):
                await self.client.close()
            elif hasattr(self.client, 'aclose'):
                await self.client.aclose()
        except Exception as e:
            logger.warning(
                f"Error closing Mistral AI client: {e}",
                exc_info=True
            )

    @staticmethod
    def list_available_models() -> list[str]:
        """
        List available Mistral AI models.

        Returns:
            List of model identifiers
        """
        return [
            model for model in MistralProvider.MODEL_PRICING.keys()
            if model != "default"
        ]
