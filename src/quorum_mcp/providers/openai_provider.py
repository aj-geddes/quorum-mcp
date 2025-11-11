"""
OpenAI Provider Implementation for Quorum-MCP

This module implements the Provider interface for OpenAI's GPT-4 models.
It provides async API integration, accurate token counting using tiktoken,
and comprehensive error handling.

Key Features:
- GPT-4 and GPT-4-turbo support
- Accurate token counting with tiktoken
- Built-in cost calculation
- Comprehensive error mapping from OpenAI SDK
- Async/await throughout
"""

import os
import time
from typing import Any

import tiktoken
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
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


class OpenAIProvider(Provider):
    """
    OpenAI provider implementation for GPT-4 models.

    This provider handles communication with OpenAI's API, including:
    - Message formatting for chat completions
    - Token counting using tiktoken
    - Cost calculation based on model-specific pricing
    - Error handling and mapping to provider exceptions

    Supported Models:
    - gpt-4o (default, latest GPT-4 Optimized)
    - gpt-4o-mini (smaller, faster GPT-4o)
    - gpt-4-turbo
    - gpt-4
    - gpt-3.5-turbo

    Pricing (as of Nov 2024):
    - GPT-4o: $2.50/1M input tokens, $10/1M output tokens
    - GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
    - GPT-4-turbo: $10/1M input tokens, $30/1M output tokens
    - GPT-4: $30/1M input tokens, $60/1M output tokens
    - GPT-3.5-turbo: $0.50/1M input tokens, $1.50/1M output tokens
    """

    # Model pricing in dollars per 1M tokens
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-32k": {"input": 60.0, "output": 120.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    # Model context windows
    MODEL_CONTEXT = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Default model to use (default: gpt-4-turbo-preview)
            rate_limit_config: Configuration for rate limiting
            retry_config: Configuration for retry logic

        Raises:
            ProviderAuthenticationError: If no API key is provided or found
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ProviderAuthenticationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="openai",
            )

        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Initialize tiktoken encoder for token counting
        # cl100k_base is the encoding used by GPT-4 and GPT-3.5-turbo
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize tiktoken encoding: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to OpenAI's API and return the response.

        This method:
        1. Validates the request
        2. Formats it for OpenAI's chat completion API
        3. Makes the API call with error handling
        4. Parses the response and extracts token counts
        5. Calculates costs based on usage

        Args:
            request: The standardized request to send

        Returns:
            The standardized response from OpenAI

        Raises:
            ProviderError: If the request fails for any reason
        """
        # Validate request
        await self.validate_request(request)

        # Check rate limits
        await self.check_rate_limits()

        # Use request model or fall back to provider default
        model = request.model or self.model

        # Build messages array for OpenAI API
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Add context if provided (as a system message)
        if request.context:
            context_content = f"Context: {request.context}"
            messages.append({"role": "system", "content": context_content})

        # Add the user query
        messages.append({"role": "user", "content": request.query})

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

        # Add any additional metadata as extra parameters
        for key, value in request.metadata.items():
            if key not in api_params and key not in ["timeout"]:
                api_params[key] = value

        # Make the API call with timing
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                timeout=request.timeout,
                **api_params,
            )
        except AuthenticationError as e:
            raise ProviderAuthenticationError(
                f"Authentication failed: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e
        except RateLimitError as e:
            # Try to extract retry_after from headers
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass

            raise ProviderRateLimitError(
                f"Rate limit exceeded: {str(e)}",
                provider="openai",
                original_error=e,
                retry_after=retry_after,
            ) from e
        except APITimeoutError as e:
            raise ProviderTimeoutError(
                f"Request timed out: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e
        except APIConnectionError as e:
            raise ProviderConnectionError(
                f"Connection error: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e
        except BadRequestError as e:
            # Check if this is a model error
            error_message = str(e).lower()
            if "model" in error_message or "does not exist" in error_message:
                raise ProviderModelError(
                    f"Invalid model: {str(e)}",
                    provider="openai",
                    original_error=e,
                ) from e
            else:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {str(e)}",
                    provider="openai",
                    original_error=e,
                ) from e
        except APIError as e:
            # Check if this is a quota error
            error_message = str(e).lower()
            if "quota" in error_message or "insufficient" in error_message:
                raise ProviderQuotaExceededError(
                    f"Quota exceeded: {str(e)}",
                    provider="openai",
                    original_error=e,
                ) from e
            else:
                raise ProviderError(
                    f"API error: {str(e)}",
                    provider="openai",
                    original_error=e,
                ) from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected error: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e

        end_time = time.time()
        latency = end_time - start_time

        # Extract response content
        if not response.choices or len(response.choices) == 0:
            raise ProviderError(
                "No choices returned in response",
                provider="openai",
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
        Count the number of tokens in a given text using tiktoken.

        This method uses the cl100k_base encoding which is used by
        GPT-4 and GPT-3.5-turbo models.

        Args:
            text: The text to tokenize

        Returns:
            The number of tokens in the text
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            raise ProviderError(
                f"Failed to count tokens: {str(e)}",
                provider="openai",
                original_error=e,
            ) from e

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate the cost of an API call based on token usage.

        Uses model-specific pricing from MODEL_PRICING. If the model
        is not found, falls back to GPT-4-turbo pricing.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing for the current model, or fall back to gpt-4-turbo
        pricing = self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING["gpt-4-turbo-preview"],
        )

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            "openai"
        """
        return "openai"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model metadata including:
            - name: Model identifier
            - provider: Provider name
            - context_window: Maximum context length in tokens
            - pricing: Pricing information per 1M tokens
        """
        pricing = self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING["gpt-4-turbo-preview"],
        )

        context_window = self.MODEL_CONTEXT.get(
            self.model,
            self.MODEL_CONTEXT["gpt-4-turbo-preview"],
        )

        return {
            "name": self.model,
            "provider": "openai",
            "context_window": context_window,
            "pricing": {
                "input_per_1m": pricing["input"],
                "output_per_1m": pricing["output"],
                "currency": "USD",
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
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Error closing OpenAI client: {e}",
                exc_info=True
            )
