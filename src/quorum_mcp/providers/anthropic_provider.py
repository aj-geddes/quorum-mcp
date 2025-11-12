"""
Anthropic Claude Provider Implementation

This module provides the concrete implementation for Anthropic's Claude API
integration, supporting Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models.

Features:
- Full async/await support
- Streaming and non-streaming responses
- Token counting and cost calculation
- Comprehensive error mapping
- Retry logic with rate limit handling
- Support for system prompts and context
"""

import os
import time
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message

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


class AnthropicProvider(Provider):
    """
    Anthropic Claude API provider implementation.

    This provider integrates with Anthropic's Claude models through their official
    Python SDK. It handles authentication, request formatting, response parsing,
    token counting, and cost calculation.

    Supported Models:
        - claude-3-5-sonnet-20241022 (default)
        - claude-3-5-sonnet-20240620
        - claude-3-opus-20240229
        - claude-3-sonnet-20240229
        - claude-3-haiku-20240307

    Pricing (as of November 2024):
        - Claude 3.5 Sonnet: $3/1M input, $15/1M output tokens
        - Claude 3 Opus: $15/1M input, $75/1M output tokens
        - Claude 3 Sonnet: $3/1M input, $15/1M output tokens
        - Claude 3 Haiku: $0.25/1M input, $1.25/1M output tokens

    Example:
        ```python
        provider = AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-20241022"
        )

        request = ProviderRequest(
            query="What is the capital of France?",
            system_prompt="You are a helpful assistant.",
            max_tokens=1000,
        )

        response = await provider.send_request(request)
        print(response.content)
        ```
    """

    # Model pricing in USD per million tokens
    MODEL_PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    # Model context windows
    MODEL_CONTEXT_WINDOWS = {
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    # Default model
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, will be read from ANTHROPIC_API_KEY
                environment variable.
            model: Model to use. Defaults to claude-3-5-sonnet-20241022.
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Raises:
            ProviderAuthenticationError: If API key is not provided and not in environment
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ProviderAuthenticationError(
                    "ANTHROPIC_API_KEY environment variable not set",
                    provider="anthropic",
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

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=api_key)

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to Claude API and return the response.

        This method handles:
        - Request validation
        - Rate limiting checks
        - Request formatting for Claude API
        - API call with retry logic
        - Response parsing and token counting
        - Error mapping to Provider exceptions

        Args:
            request: The standardized request to send

        Returns:
            The standardized response from Claude

        Raises:
            ProviderError: If the request fails for any reason
        """
        # Validate request
        await self.validate_request(request)

        # Check rate limits
        await self.check_rate_limits()

        # Determine model to use
        model = request.model or self.model

        # Validate model
        if model not in self.MODEL_PRICING:
            raise ProviderModelError(
                f"Model '{model}' not supported. Supported models: {list(self.MODEL_PRICING.keys())}",
                provider=self.get_provider_name(),
            )

        # Format messages for Claude API
        messages = self._format_messages(request)

        # Prepare system prompt
        system_prompt = self._format_system_prompt(request)

        # Start timing
        start_time = time.time()

        # Make API call with retry logic
        attempt = 0
        last_error = None

        while attempt <= self.retry_config.max_retries:
            try:
                # Call Claude API
                response = await self._call_api(
                    model=model,
                    messages=messages,
                    system=system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    timeout=request.timeout,
                    metadata=request.metadata,
                )

                # Calculate latency
                latency = time.time() - start_time

                # Extract content
                content = self._extract_content(response)

                # Get token counts
                tokens_input = response.usage.input_tokens
                tokens_output = response.usage.output_tokens

                # Calculate cost
                cost = self.get_cost(tokens_input, tokens_output, model)

                # Build response
                return ProviderResponse(
                    content=content,
                    model=response.model,
                    provider=self.get_provider_name(),
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost=cost,
                    latency=latency,
                    metadata={
                        "stop_reason": response.stop_reason,
                        "stop_sequence": response.stop_sequence,
                        "response_id": response.id,
                    },
                )

            except Exception as e:
                # Map exception to Provider error
                provider_error = self._map_exception(e)
                last_error = provider_error

                # Try to handle retry
                try:
                    delay = await self.handle_retry(provider_error, attempt)
                    if delay > 0:
                        await self._async_sleep(delay)
                        attempt += 1
                        continue
                except ProviderError:
                    # No retry, raise the error
                    raise provider_error from e

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error

    def _format_messages(self, request: ProviderRequest) -> list[dict[str, str]]:
        """
        Format messages for Claude API.

        Claude expects messages in the format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]

        Args:
            request: The provider request

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add context as a user message if provided
        if request.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{request.context}\n\nPlease use this context to answer the following query.",
                }
            )

        # Add the main query
        messages.append({"role": "user", "content": request.query})

        return messages

    def _format_system_prompt(self, request: ProviderRequest) -> str | None:
        """
        Format system prompt for Claude API.

        Args:
            request: The provider request

        Returns:
            System prompt string or None
        """
        return request.system_prompt if request.system_prompt else None

    async def _call_api(
        self,
        model: str,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        timeout: float,
        metadata: dict[str, Any],
    ) -> Message:
        """
        Make the actual API call to Claude.

        Args:
            model: Model to use
            messages: Formatted messages
            system: System prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            timeout: Request timeout
            metadata: Additional metadata

        Returns:
            Claude API message response

        Raises:
            Exception: Any exception from the Anthropic SDK
        """
        # Build API parameters
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }

        # Add optional parameters
        if system:
            params["system"] = system

        if top_p is not None:
            params["top_p"] = top_p

        # Add metadata if provided
        if metadata:
            # Extract Anthropic-specific metadata
            if "user_id" in metadata:
                params["metadata"] = {"user_id": metadata["user_id"]}

        # Make the API call
        response = await self.client.messages.create(**params)

        return response

    def _extract_content(self, response: Message) -> str:
        """
        Extract text content from Claude response.

        Args:
            response: Claude API message response

        Returns:
            Extracted text content
        """
        # Claude returns content as a list of content blocks
        # Concatenate all text blocks
        content_parts = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)

        return "".join(content_parts)

    def _map_exception(self, error: Exception) -> ProviderError:
        """
        Map Anthropic SDK exceptions to Provider exceptions.

        Args:
            error: The original exception

        Returns:
            Mapped ProviderError
        """
        provider_name = self.get_provider_name()

        # Authentication errors
        if isinstance(error, anthropic.AuthenticationError):
            return ProviderAuthenticationError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Rate limit errors
        if isinstance(error, anthropic.RateLimitError):
            # Try to extract retry_after from headers
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    try:
                        retry_after = float(retry_after)
                    except (ValueError, TypeError):
                        retry_after = None

            return ProviderRateLimitError(
                str(error),
                provider=provider_name,
                original_error=error,
                retry_after=retry_after,
            )

        # Invalid request errors
        if isinstance(error, anthropic.BadRequestError):
            return ProviderInvalidRequestError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Not found / invalid model errors
        if isinstance(error, anthropic.NotFoundError):
            return ProviderModelError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Permission / quota errors
        if isinstance(error, anthropic.PermissionDeniedError):
            return ProviderQuotaExceededError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Timeout errors
        if isinstance(error, anthropic.APITimeoutError):
            return ProviderTimeoutError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Connection errors
        if isinstance(error, anthropic.APIConnectionError):
            return ProviderConnectionError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Generic API errors
        if isinstance(error, anthropic.APIError):
            return ProviderError(
                str(error),
                provider=provider_name,
                original_error=error,
            )

        # Unknown errors
        return ProviderError(
            f"Unexpected error: {str(error)}",
            provider=provider_name,
            original_error=error,
        )

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Anthropic's token counting.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        # Use Anthropic's token counting API
        try:
            # Format as a message for token counting
            result = await self.client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            # Fallback to approximation if API call fails
            # Claude uses roughly 4 characters per token on average
            return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int, model: str | None = None) -> float:
        """
        Calculate the cost of an API call.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            model: Model used (defaults to self.model)

        Returns:
            Cost in USD
        """
        # Use specified model or default
        model = model or self.model

        # Get pricing for model
        pricing = self.MODEL_PRICING.get(model)
        if not pricing:
            # Use Claude 3.5 Sonnet pricing as default
            pricing = self.MODEL_PRICING[self.DEFAULT_MODEL]

        # Calculate cost (pricing is per million tokens)
        input_cost = (tokens_input / 1_000_000) * pricing["input"]
        output_cost = (tokens_output / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            "anthropic"
        """
        return "anthropic"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        context_window = self.MODEL_CONTEXT_WINDOWS.get(self.model, 200000)
        pricing = self.MODEL_PRICING.get(self.model, self.MODEL_PRICING[self.DEFAULT_MODEL])

        return {
            "provider": self.get_provider_name(),
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
        List all available Anthropic models.

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

    async def _async_sleep(self, seconds: float) -> None:
        """
        Async sleep helper for testing and retry logic.

        Args:
            seconds: Time to sleep in seconds
        """
        import asyncio

        await asyncio.sleep(seconds)

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
                f"Error closing Anthropic client: {e}",
                exc_info=True
            )
