"""
Ollama Provider Implementation

This module provides integration with Ollama for running local LLMs.

Supported Models:
- llama3.2 (default) - Fast, efficient local inference
- llama3.1 - Advanced reasoning
- mistral - Efficient 7B model
- mixtral - Mixture of experts model
- qwen3 - Alibaba's efficient model
- deepseek-r1 - Reasoning focused
- gemma3 - Google's local model

Features:
- Async/await support via AsyncClient
- Zero-cost local inference
- Streaming support
- Error mapping to Provider exceptions
- Model availability checking
"""

import os
from typing import Any

from ollama import AsyncClient, ResponseError

from quorum_mcp.providers.base import (
    Provider,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderModelError,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeoutError,
    RateLimitConfig,
    RetryConfig,
)


class OllamaProvider(Provider):
    """
    Ollama Local LLM Provider.

    Integrates with Ollama for running LLMs locally with zero cost.
    """

    # Popular models with their context windows
    MODEL_CONTEXT_WINDOWS = {
        "llama3.2": 128000,
        "llama3.1": 128000,
        "mistral": 32000,
        "mixtral": 32000,
        "qwen3": 32000,
        "deepseek-r1": 64000,
        "gemma3": 8192,
    }

    def __init__(
        self,
        model: str = "llama3.2",
        host: str | None = None,
        timeout: float = 120.0,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model identifier (default: llama3.2)
            host: Ollama server host (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 120.0)
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration

        Note:
            Ollama server must be running locally. Start with: ollama serve
        """
        # Store host and timeout before initialization
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

        # Initialize base class (Ollama doesn't use API keys for local inference)
        super().__init__(
            api_key="local",  # Placeholder for local inference
            model=model,
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
        )

        # Initialize async client
        try:
            self.client = AsyncClient(host=self.host, timeout=self.timeout)
        except Exception as e:
            raise ProviderConnectionError(
                f"Failed to initialize Ollama client: {e} | Provider: ollama"
            )

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send request to Ollama and return structured response.

        Args:
            request: Provider request with prompt, context, and parameters

        Returns:
            ProviderResponse with content, tokens, and cost ($0 for local)

        Raises:
            Various ProviderError subclasses for different error conditions
        """
        try:
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

            # Build options
            options = {}
            if request.temperature is not None:
                options["temperature"] = request.temperature
            if request.max_tokens:
                options["num_predict"] = request.max_tokens

            # Call async API
            response = await self.client.chat(
                model=model,
                messages=messages,
                options=options if options else None,
                stream=False,
            )

            # Extract content
            content = response["message"]["content"]

            # Estimate token usage (Ollama doesn't provide token counts)
            # Using ~4 chars per token as rough estimate
            input_text = (request.system_prompt or "") + user_content
            tokens_input = len(input_text) // 4
            tokens_output = len(content) // 4

            # Local inference is free!
            cost = 0.0

            # Build metadata
            metadata = {
                "model": model,
                "host": self.host,
                "eval_count": response.get("eval_count"),
                "eval_duration": response.get("eval_duration"),
                "total_duration": response.get("total_duration"),
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

        except ResponseError as e:
            # Map Ollama ResponseError to Provider errors
            error_message = str(e.error).lower()

            if e.status_code == 404:
                raise ProviderModelError(
                    f"Model not found: {model}. Pull with 'ollama pull {model}' | Provider: ollama"
                ) from e

            if "not found" in error_message or "invalid" in error_message:
                raise ProviderInvalidRequestError(
                    f"Invalid request: {e.error} | Provider: ollama"
                ) from e

            # Generic response error
            raise ProviderError(f"Ollama API error: {e.error}") from e

        except ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama server at {self.host}. "
                f"Is Ollama running? Start with 'ollama serve' | Provider: ollama"
            ) from e

        except TimeoutError as e:
            raise ProviderTimeoutError(f"Request timed out | Provider: ollama") from e

        except Exception as e:
            # Generic error
            error_message = str(e).lower()

            if "connection" in error_message or "refused" in error_message:
                raise ProviderConnectionError(
                    f"Connection error: {e}. Is Ollama running? | Provider: ollama"
                ) from e

            raise ProviderError(f"Ollama error: {e}") from e

    async def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Note:
            Ollama doesn't provide a token counting API, so we estimate
            using ~4 characters per token as a rough approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        # Rough estimation: ~4 chars per token
        return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate cost based on token usage.

        Note:
            Local inference with Ollama is always free!

        Args:
            tokens_input: Number of input tokens (ignored)
            tokens_output: Number of output tokens (ignored)

        Returns:
            Cost in USD (always 0.0)
        """
        return 0.0

    def get_provider_name(self) -> str:
        """Get provider identifier."""
        return "ollama"

    def get_model_info(self) -> dict:
        """
        Get information about the current model.

        Returns:
            Dictionary with model details
        """
        context_window = self.MODEL_CONTEXT_WINDOWS.get(self.model, 32000)

        return {
            "provider": self.get_provider_name(),
            "model": self.model,
            "context_window": context_window,
            "pricing": {"input": 0.0, "output": 0.0},
            "host": self.host,
            "local": True,
        }

    @staticmethod
    def list_available_models() -> list[str]:
        """
        List popular Ollama models.

        Returns:
            List of model identifiers
        """
        return list(OllamaProvider.MODEL_CONTEXT_WINDOWS.keys())

    async def check_availability(self) -> dict[str, Any]:
        """
        Check if Ollama server is running and model is available.

        Returns:
            Dictionary with availability status
        """
        try:
            # Try to list models to verify server is running
            models_response = await self.client.list()

            # Check if our model is available
            available_models = [m["model"] for m in models_response.get("models", [])]
            model_available = any(self.model in m for m in available_models)

            return {
                "server_running": True,
                "model_available": model_available,
                "available_models": available_models,
                "host": self.host,
            }

        except Exception as e:
            return {
                "server_running": False,
                "model_available": False,
                "error": str(e),
                "host": self.host,
            }

    async def aclose(self) -> None:
        """
        Close the async client and release resources.

        Should be called when done using the provider.
        """
        try:
            await self.client.aclose()
        except Exception:
            pass  # Ignore errors during cleanup
