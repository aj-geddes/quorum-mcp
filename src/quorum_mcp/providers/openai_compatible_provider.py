"""
OpenAI-Compatible API Provider for Quorum-MCP

This provider enables connection to ANY local or remote LLM that implements
the OpenAI-compatible API standard. This includes:

Local Providers:
- LM Studio (http://localhost:1234/v1)
- text-generation-webui (http://localhost:5000/v1)
- LocalAI (http://localhost:8080/v1)
- vLLM (custom endpoint)
- llama.cpp server (http://localhost:8080/v1)
- Ollama (via OpenAI compatibility mode)
- TabbyAPI (ExLlamaV2)

Cloud Providers:
- OpenRouter (https://openrouter.ai/api/v1)
- Together AI (https://api.together.xyz/v1)
- Anyscale (https://api.endpoints.anyscale.com/v1)
- Deep Infra (https://api.deepinfra.com/v1/openai)

Key Features:
- Universal compatibility with OpenAI API standard
- Custom base URL configuration
- Optional API key support (local endpoints often don't require)
- Full async/await support
- Token counting with tiktoken
- Cost tracking (configurable or free for local)
"""

import os
import time
from typing import Any

import tiktoken
from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError, APITimeoutError

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


class OpenAICompatibleProvider(Provider):
    """
    Universal provider for any OpenAI-compatible API endpoint.

    This provider allows connection to local LLM servers (LM Studio, text-generation-webui,
    LocalAI, vLLM, etc.) or cloud providers that implement the OpenAI API standard.

    Example Usage:
        # LM Studio
        lm_studio = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
            provider_name="lm-studio",
            api_key="not-needed"  # LM Studio doesn't require key
        )

        # text-generation-webui
        textgen = OpenAICompatibleProvider(
            base_url="http://localhost:5000/v1",
            model="TheBloke/Llama-2-13B-GGUF",
            provider_name="textgen-webui"
        )

        # OpenRouter
        openrouter = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            provider_name="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            pricing={"input": 3.0, "output": 15.0}  # $/1M tokens
        )
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        provider_name: str = "openai-compatible",
        api_key: str | None = None,
        pricing: dict[str, float] | None = None,
        context_window: int = 32000,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize the OpenAI-compatible provider.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:1234/v1")
            model: Model name/identifier to use
            provider_name: Human-readable provider name for logging (default: "openai-compatible")
            api_key: API key (optional for local endpoints, required for cloud)
            pricing: Cost per 1M tokens {"input": X, "output": Y}, None for free (default: None)
            context_window: Maximum context length in tokens (default: 32000)
            rate_limit_config: Configuration for rate limiting
            retry_config: Configuration for retry logic
        """
        # Use dummy key if none provided (local endpoints don't need it)
        if api_key is None:
            api_key = "not-needed"

        super().__init__(
            api_key=api_key,
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        self.base_url = base_url
        self.provider_name = provider_name
        self.pricing = pricing  # None = free (local), dict = paid (cloud)
        self.context_window = context_window

        # Initialize OpenAI client with custom base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Initialize tiktoken for token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken not available
            self.encoding = None

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send request to OpenAI-compatible API endpoint.

        Args:
            request: The standardized request to send

        Returns:
            The standardized response

        Raises:
            ProviderError: If the request fails
        """
        # Validate request
        await self.validate_request(request)

        # Check rate limits
        await self.check_rate_limits()

        # Use request model or fall back to provider default
        model = request.model or self.model

        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        user_content = request.query
        if request.context:
            user_content = f"{request.context}\n\n{request.query}"
        messages.append({"role": "user", "content": user_content})

        # Prepare API call parameters
        api_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.top_p is not None:
            api_params["top_p"] = request.top_p

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
                provider=self.provider_name,
                original_error=e,
            ) from e
        except RateLimitError as e:
            raise ProviderRateLimitError(
                f"Rate limit exceeded: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e
        except APITimeoutError as e:
            raise ProviderTimeoutError(
                f"Request timed out: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e
        except APIError as e:
            raise ProviderError(
                f"API error: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e
        except Exception as e:
            # Catch connection errors
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                raise ProviderConnectionError(
                    f"Cannot connect to {self.base_url}: {str(e)}",
                    provider=self.provider_name,
                    original_error=e,
                ) from e
            raise ProviderError(
                f"Unexpected error: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e

        end_time = time.time()
        latency = end_time - start_time

        # Extract response content
        if not response.choices or len(response.choices) == 0:
            raise ProviderError(
                "No choices returned in response",
                provider=self.provider_name,
            )

        content = response.choices[0].message.content or ""

        # Extract token counts
        tokens_input = response.usage.prompt_tokens if response.usage else 0
        tokens_output = response.usage.completion_tokens if response.usage else 0

        # Calculate cost
        cost = self.get_cost(tokens_input, tokens_output)

        # Build metadata
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model,
            "response_id": response.id,
            "base_url": self.base_url,
        }

        return ProviderResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            latency=latency,
            metadata=metadata,
        )

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        if self.encoding:
            try:
                tokens = self.encoding.encode(text)
                return len(tokens)
            except Exception:
                pass

        # Fallback: rough estimation
        return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate cost based on token usage.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Cost in USD (0.0 if no pricing configured)
        """
        if self.pricing is None:
            return 0.0  # Free for local endpoints

        input_cost = (tokens_input / 1_000_000) * self.pricing.get("input", 0.0)
        output_cost = (tokens_output / 1_000_000) * self.pricing.get("output", 0.0)

        return input_cost + output_cost

    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.provider_name

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "context_window": self.context_window,
            "base_url": self.base_url,
            "pricing": self.pricing or {"input": 0.0, "output": 0.0},
            "local": "localhost" in self.base_url or "127.0.0.1" in self.base_url,
        }

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List available models.

        Note: For OpenAI-compatible providers, models are endpoint-specific.
        This returns a generic list of common configurations.
        """
        return [
            "local-model",  # Generic local model
            "lm-studio",
            "textgen-webui",
            "localai",
        ]
