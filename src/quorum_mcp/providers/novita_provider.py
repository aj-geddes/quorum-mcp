"""
Novita AI Provider for Quorum-MCP

This module implements the Novita AI provider, supporting ultra-low-cost LLM inference
through an OpenAI-compatible API. Novita AI offers competitive pricing and multiple
models including Llama, DeepSeek, and others.

Key Features:
- OpenAI-compatible API (easy migration)
- Ultra-low pricing (as low as $0.04/M tokens)
- Multiple open-source models available
- Serverless integration with dynamic scaling
"""

import logging
import os
from datetime import datetime, timezone

from openai import AsyncOpenAI

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


class NovitaProvider(Provider):
    """
    Provider implementation for Novita AI.

    Novita AI offers ultra-low-cost LLM inference with OpenAI-compatible API,
    making it easy to integrate and cost-effective for high-volume applications.
    """

    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"

    # Model pricing (per million tokens) - Based on Novita AI's competitive pricing
    MODEL_PRICING = {
        # Llama Models
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.04, "output": 0.04},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.02},
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.06, "output": 0.06},
        "meta-llama/llama-3.1-405b-instruct": {"input": 0.20, "output": 0.20},
        # DeepSeek Models
        "deepseek/deepseek-r1": {"input": 0.04, "output": 0.04},
        "deepseek/deepseek-v3": {"input": 0.04, "output": 0.04},
        # Qwen Models
        "qwen/qwen-2.5-72b-instruct": {"input": 0.06, "output": 0.06},
        "qwen/qwen-2.5-coder-32b-instruct": {"input": 0.04, "output": 0.04},
        # Mixtral Models
        "mistralai/mixtral-8x7b-instruct": {"input": 0.06, "output": 0.06},
        "mistralai/mixtral-8x22b-instruct": {"input": 0.12, "output": 0.12},
        # Default for unknown models
        "default": {"input": 0.10, "output": 0.10},
    }

    # Context windows for models
    MODEL_CONTEXT_WINDOWS = {
        "meta-llama/llama-3.3-70b-instruct": 128000,
        "meta-llama/llama-3.1-8b-instruct": 128000,
        "meta-llama/llama-3.1-70b-instruct": 128000,
        "meta-llama/llama-3.1-405b-instruct": 128000,
        "deepseek/deepseek-r1": 64000,
        "deepseek/deepseek-v3": 64000,
        "qwen/qwen-2.5-72b-instruct": 32768,
        "qwen/qwen-2.5-coder-32b-instruct": 32768,
        "mistralai/mixtral-8x7b-instruct": 32768,
        "mistralai/mixtral-8x22b-instruct": 64000,
        "default": 32768,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Novita AI provider.

        Args:
            api_key: Novita AI API key (or from NOVITA_API_KEY env var)
            model: Model identifier (default: meta-llama/llama-3.3-70b-instruct)
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Raises:
            ProviderAuthenticationError: If API key is not provided and not in environment
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("NOVITA_API_KEY")
            if not api_key:
                raise ProviderAuthenticationError(
                    "NOVITA_API_KEY environment variable not set",
                    provider="novita",
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

        # Initialize OpenAI client with Novita AI base URL
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.novita.ai/openai",
        )

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to Novita AI and return the response.

        This method handles:
        - Request validation
        - Rate limiting checks
        - Request formatting for Novita AI API (OpenAI-compatible)
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
                messages.append({"role": "system", "content": request.system_prompt})

            # Build user message (context + query)
            user_content = request.query
            if request.context:
                user_content = f"{request.context}\n\n{request.query}"

            messages.append({"role": "user", "content": user_content})

            # Start timing
            start_time = datetime.now(timezone.utc)

            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
                timeout=request.timeout,
            )

            # Calculate latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Extract response content
            if not response.choices or len(response.choices) == 0:
                raise ProviderError(
                    "No choices returned in response",
                    provider="novita",
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
            # Map Novita AI/OpenAI exceptions to Provider exceptions
            error_message = str(e).lower()

            if "authentication" in error_message or "api key" in error_message or "401" in error_message:
                raise ProviderAuthenticationError(
                    f"Authentication failed: {e}",
                    provider="novita",
                ) from e

            if "rate limit" in error_message or "429" in error_message:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="novita",
                ) from e

            if "timeout" in error_message or "timed out" in error_message:
                raise ProviderTimeoutError(
                    f"Request timeout: {e}",
                    provider="novita",
                ) from e

            if "model" in error_message and "not found" in error_message:
                raise ProviderModelError(
                    f"Model not found or not available: {e}",
                    provider="novita",
                ) from e

            if "quota" in error_message or "insufficient" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {e}",
                    provider="novita",
                ) from e

            if "invalid" in error_message or "400" in error_message:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {e}",
                    provider="novita",
                ) from e

            if "connection" in error_message or "network" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {e}",
                    provider="novita",
                ) from e

            # Generic error for anything else
            raise ProviderError(
                f"Novita AI request failed: {e}",
                provider="novita",
            ) from e

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken (same as OpenAI).

        Since Novita AI uses OpenAI-compatible models, we use the same
        tokenization approach as OpenAI (cl100k_base encoding).

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            import tiktoken

            # Use cl100k_base encoding (same as GPT-4/GPT-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
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
        return "novita"

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
            "provider": "novita",
            "context_window": context_window,
            "pricing": {
                "input_per_1m": pricing["input"],
                "output_per_1m": pricing["output"],
                "currency": "USD",
            },
            "api_compatible": "OpenAI",
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
                f"Error closing Novita AI client: {e}",
                exc_info=True
            )

    @staticmethod
    def list_available_models() -> list[str]:
        """
        List available Novita AI models.

        Returns:
            List of model identifiers
        """
        return [
            model for model in NovitaProvider.MODEL_PRICING.keys()
            if model != "default"
        ]
