"""
Cohere Provider for Quorum-MCP

This module implements the Cohere provider, supporting enterprise-grade LLM inference
with excellent RAG (Retrieval-Augmented Generation) capabilities and competitive pricing.

Key Features:
- Enterprise focus with production-ready reliability
- Excellent RAG capabilities with retrieval-optimized models
- Competitive pricing ($0.15-3.00 per million tokens)
- Command R and Command R+ models for enhanced reasoning
- Free tier available for prototyping
"""

import logging
import os
from datetime import datetime, timezone

import cohere

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


class CohereProvider(Provider):
    """
    Provider implementation for Cohere.

    Cohere offers enterprise-grade LLMs with excellent RAG capabilities,
    competitive pricing, and production-ready reliability.
    """

    DEFAULT_MODEL = "command-r-plus"

    # Model pricing (per million tokens) - Based on Cohere's 2025 pricing
    MODEL_PRICING = {
        # Command R+ (Most capable)
        "command-r-plus": {"input": 3.00, "output": 15.00},
        # Command R (Optimized balance)
        "command-r": {"input": 0.50, "output": 1.50},
        # Command (Legacy)
        "command": {"input": 1.00, "output": 2.00},
        # Command Light (Fast, economical)
        "command-light": {"input": 0.30, "output": 0.60},
        # Command Nightly (Latest experimental)
        "command-nightly": {"input": 1.00, "output": 2.00},
        # Default for unknown models
        "default": {"input": 1.00, "output": 2.00},
    }

    # Context windows for models
    MODEL_CONTEXT_WINDOWS = {
        "command-r-plus": 128000,
        "command-r": 128000,
        "command": 4096,
        "command-light": 4096,
        "command-nightly": 128000,
        "default": 4096,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Cohere provider.

        Args:
            api_key: Cohere API key (or from COHERE_API_KEY env var)
            model: Model identifier (default: command-r-plus)
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Raises:
            ProviderAuthenticationError: If API key is not provided and not in environment
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ProviderAuthenticationError(
                    "COHERE_API_KEY environment variable not set",
                    provider="cohere",
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

        # Initialize Cohere client
        self.client = cohere.AsyncClientV2(api_key=api_key)

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to Cohere and return the response.

        This method handles:
        - Request validation
        - Rate limiting checks
        - Request formatting for Cohere API
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

            # Add system message if provided
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

            # Make API call
            response = await self.client.chat(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                p=request.top_p,  # Cohere uses 'p' instead of 'top_p'
            )

            # Calculate latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Extract response content
            if not response.message or not response.message.content:
                raise ProviderError(
                    "No content returned in response",
                    provider="cohere",
                )

            # Get content from response
            content_blocks = response.message.content
            if isinstance(content_blocks, list):
                # Extract text from content blocks
                content = " ".join([
                    block.text for block in content_blocks
                    if hasattr(block, 'text')
                ])
            else:
                content = str(content_blocks)

            # Get token counts from response
            tokens_input = response.usage.tokens.input_tokens if response.usage else 0
            tokens_output = response.usage.tokens.output_tokens if response.usage else 0

            # Calculate cost
            cost = self.get_cost(tokens_input, tokens_output)

            # Build metadata
            metadata = {
                "finish_reason": response.finish_reason,
                "model": model,
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

        except cohere.core.ApiError as e:
            # Map Cohere API errors to Provider exceptions
            error_message = str(e).lower()
            status_code = getattr(e, 'status_code', 0)

            if status_code == 401 or "authentication" in error_message or "api key" in error_message:
                raise ProviderAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="cohere",
                ) from e

            if status_code == 429 or "rate limit" in error_message or "too many requests" in error_message:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="cohere",
                ) from e

            if status_code == 408 or "timeout" in error_message or "timed out" in error_message:
                raise ProviderTimeoutError(
                    f"Request timeout: {e}",
                    provider="cohere",
                ) from e

            if "model" in error_message and ("not found" in error_message or "unavailable" in error_message):
                raise ProviderModelError(
                    f"Model not found or not available: {e}",
                    provider="cohere",
                ) from e

            if "quota" in error_message or "insufficient" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {e}",
                    provider="cohere",
                ) from e

            if status_code == 400 or "invalid" in error_message:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {e}",
                    provider="cohere",
                ) from e

            # Generic API error
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
            ) from e

        except Exception as e:
            # Handle connection and other errors
            error_message = str(e).lower()

            if "connection" in error_message or "network" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {e}",
                    provider="cohere",
                ) from e

            # Generic error for anything else
            raise ProviderError(
                f"Cohere request failed: {e}",
                provider="cohere",
            ) from e

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Cohere's tokenize API.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            # Use Cohere's tokenize endpoint for accurate counting
            response = await self.client.tokenize(
                text=text,
                model=self.model,
            )
            return len(response.tokens) if response.tokens else 0
        except Exception as e:
            logger.warning(f"Token counting failed, using estimation: {e}")
            # Fallback: estimate ~4 chars per token
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
        return "cohere"

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
            "provider": "cohere",
            "context_window": context_window,
            "pricing": {
                "input_per_1m": pricing["input"],
                "output_per_1m": pricing["output"],
                "currency": "USD",
            },
            "features": {
                "rag_optimized": True,
                "enterprise_ready": True,
                "free_tier": True,
            },
        }

    async def aclose(self) -> None:
        """
        Close the async client and release resources.

        Should be called when done using the provider to properly cleanup
        HTTP connections and other resources.
        """
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(
                f"Error closing Cohere client: {e}",
                exc_info=True
            )

    @staticmethod
    def list_available_models() -> list[str]:
        """
        List available Cohere models.

        Returns:
            List of model identifiers
        """
        return [
            model for model in CohereProvider.MODEL_PRICING.keys()
            if model != "default"
        ]
