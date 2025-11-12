"""
Provider Abstraction Layer for Quorum-MCP

This module defines the abstract base classes and data models for AI provider
integration. It provides a unified interface for interacting with multiple AI
providers (Anthropic Claude, OpenAI GPT-4, Google Gemini, Mistral, etc.).

Key Design Principles:
- Async-first architecture for concurrent API calls
- Built-in retry and rate limiting hooks
- Comprehensive error handling
- Token counting and cost tracking
- Extensible for future providers
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from quorum_mcp.rate_limiter import ProviderRateLimiter
    from quorum_mcp.budget import BudgetManager
    from quorum_mcp.benchmark import BenchmarkTracker


class ProviderType(str, Enum):
    """Enumeration of supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MISTRAL = "mistral"
    CUSTOM = "custom"


class HealthStatus(str, Enum):
    """Provider health status levels."""

    HEALTHY = "healthy"  # Provider is fully operational
    DEGRADED = "degraded"  # Provider is working but with issues (slow, rate limited)
    UNHEALTHY = "unhealthy"  # Provider is not operational


class ProviderError(Exception):
    """Base exception class for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
        retry_after: float | None = None,
    ):
        """
        Initialize a ProviderError.

        Args:
            message: Human-readable error message
            provider: Name of the provider that raised the error
            original_error: The underlying exception if wrapping another error
            retry_after: Suggested time in seconds before retrying (if applicable)
        """
        self.message = message
        self.provider = provider
        self.original_error = original_error
        self.retry_after = retry_after
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a formatted error message."""
        parts = [self.message]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.original_error:
            parts.append(f"Underlying error: {str(self.original_error)}")
        if self.retry_after:
            parts.append(f"Retry after: {self.retry_after}s")
        return " | ".join(parts)


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails (API key invalid, etc.)."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""

    pass


class ProviderConnectionError(ProviderError):
    """Raised when unable to connect to provider API."""

    pass


class ProviderInvalidRequestError(ProviderError):
    """Raised when the request format is invalid for the provider."""

    pass


class ProviderModelError(ProviderError):
    """Raised when the requested model is unavailable or invalid."""

    pass


class ProviderQuotaExceededError(ProviderError):
    """Raised when API quota/budget is exceeded."""

    pass


class ProviderRequest(BaseModel):
    """
    Standardized request model for AI provider calls.

    This model encapsulates all parameters needed to make a request to any
    AI provider, abstracting away provider-specific differences.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="allow",  # Allow provider-specific fields
    )

    query: str = Field(
        ...,
        description="The user's query or prompt to send to the AI provider",
        min_length=1,
    )

    system_prompt: str | None = Field(
        default=None,
        description="System-level instructions that guide the AI's behavior",
    )

    context: str | None = Field(
        default=None,
        description="Additional context or background information for the query",
    )

    model: str | None = Field(
        default=None,
        description="Specific model to use (e.g., 'claude-3-opus', 'gpt-4-turbo')",
    )

    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens in the response",
        gt=0,
        le=200000,
    )

    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0.0 to 2.0)",
        ge=0.0,
        le=2.0,
    )

    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        gt=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for tracking or provider-specific options",
    )

    def to_provider_format(self, provider_type: ProviderType) -> dict[str, Any]:
        """
        Convert this request to a provider-specific format.

        This method can be overridden by subclasses to handle provider-specific
        transformations.

        Args:
            provider_type: The target provider type

        Returns:
            Dictionary in provider-specific format
        """
        return self.model_dump(exclude_none=True)


class ProviderResponse(BaseModel):
    """
    Standardized response model from AI provider calls.

    This model normalizes responses from different providers into a consistent
    format for downstream processing and consensus building.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
    )

    content: str = Field(
        ...,
        description="The main response content from the AI provider",
    )

    confidence: float | None = Field(
        default=None,
        description="Provider's confidence score (0.0 to 1.0) if available",
        ge=0.0,
        le=1.0,
    )

    model: str = Field(
        ...,
        description="The actual model that generated the response",
    )

    provider: str = Field(
        ...,
        description="Name of the provider that generated this response",
    )

    tokens_input: int = Field(
        default=0,
        description="Number of tokens in the input/prompt",
        ge=0,
    )

    tokens_output: int = Field(
        default=0,
        description="Number of tokens in the output/response",
        ge=0,
    )

    cost: float | None = Field(
        default=None,
        description="Estimated cost of this API call in USD",
        ge=0.0,
    )

    latency: float = Field(
        default=0.0,
        description="Response latency in seconds",
        ge=0.0,
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the response was received",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific metadata",
    )

    error: str | None = Field(
        default=None,
        description="Error message if the request failed (for partial failures)",
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert response to a dictionary.

        Returns:
            Dictionary representation of the response
        """
        return self.model_dump()


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    model_config = ConfigDict(frozen=True)

    requests_per_minute: int | None = Field(
        default=None,
        description="Maximum requests per minute (None for unlimited)",
        ge=1,
    )

    tokens_per_minute: int | None = Field(
        default=None,
        description="Maximum tokens per minute (None for unlimited)",
        ge=1,
    )

    concurrent_requests: int = Field(
        default=5,
        description="Maximum concurrent requests",
        ge=1,
    )


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
    )

    base_delay: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff",
        gt=0,
    )

    max_delay: float = Field(
        default=60.0,
        description="Maximum delay in seconds between retries",
        gt=0,
    )

    exponential_base: float = Field(
        default=2.0,
        description="Base for exponential backoff calculation",
        gt=1.0,
    )

    retry_on_timeout: bool = Field(
        default=True,
        description="Whether to retry on timeout errors",
    )

    retry_on_rate_limit: bool = Field(
        default=True,
        description="Whether to retry on rate limit errors",
    )

    retry_on_server_error: bool = Field(
        default=True,
        description="Whether to retry on server errors (5xx)",
    )


class HealthCheckResult(BaseModel):
    """Result of a provider health check."""

    model_config = ConfigDict(frozen=False)

    status: HealthStatus = Field(
        ...,
        description="Overall health status of the provider",
    )

    response_time: float | None = Field(
        default=None,
        description="Response time in seconds for the health check",
        ge=0.0,
    )

    error: str | None = Field(
        default=None,
        description="Error message if health check failed",
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health check details (model availability, rate limits, etc.)",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed",
    )

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_usable(self) -> bool:
        """Check if provider is usable (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class Provider(ABC):
    """
    Abstract base class for AI provider implementations.

    This class defines the interface that all provider implementations must follow.
    It includes methods for sending requests, counting tokens, calculating costs,
    and managing provider-specific configurations.

    Subclasses must implement all abstract methods to integrate a new provider.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
        rate_limiter: "ProviderRateLimiter | None" = None,
        budget_manager: "BudgetManager | None" = None,
        benchmark_tracker: "BenchmarkTracker | None" = None,
    ):
        """
        Initialize a provider.

        Args:
            api_key: API key for authenticating with the provider
            model: Default model to use for this provider
            rate_limit_config: Configuration for rate limiting (deprecated, use rate_limiter)
            retry_config: Configuration for retry logic
            rate_limiter: Rate limiter instance for this provider
            budget_manager: Budget manager for cost tracking
            benchmark_tracker: Benchmark tracker for performance metrics
        """
        self.api_key = api_key
        self.model = model
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()

        # Advanced system integrations
        self.rate_limiter = rate_limiter
        self.budget_manager = budget_manager
        self.benchmark_tracker = benchmark_tracker

        # Internal state for rate limiting (legacy, for backwards compatibility)
        self._request_count = 0
        self._token_count = 0
        self._last_reset = datetime.now(timezone.utc)

    @abstractmethod
    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """
        Send a request to the AI provider and return the response.

        This method should handle:
        - Converting the ProviderRequest to provider-specific format
        - Making the actual API call
        - Handling errors and wrapping them in ProviderError subclasses
        - Converting the provider response to ProviderResponse format
        - Token counting and cost calculation

        Args:
            request: The standardized request to send

        Returns:
            The standardized response from the provider

        Raises:
            ProviderError: If the request fails for any reason
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.

        Different providers use different tokenization schemes. This method
        should use the provider's specific tokenizer or estimation method.

        Args:
            text: The text to tokenize

        Returns:
            The number of tokens in the text
        """
        pass

    @abstractmethod
    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """
        Calculate the cost of an API call based on token usage.

        Providers have different pricing models. This method should implement
        the specific pricing calculation for the provider.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens

        Returns:
            Cost in USD
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., "anthropic", "openai")
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model metadata (name, context window, etc.)
        """
        pass

    async def validate_request(self, request: ProviderRequest) -> None:
        """
        Validate a request before sending it to the provider.

        This method can be overridden to add provider-specific validation.

        Args:
            request: The request to validate

        Raises:
            ProviderInvalidRequestError: If the request is invalid
        """
        if not request.query or not request.query.strip():
            raise ProviderInvalidRequestError(
                "Query cannot be empty",
                provider=self.get_provider_name(),
            )

        if request.max_tokens <= 0:
            raise ProviderInvalidRequestError(
                f"max_tokens must be positive, got {request.max_tokens}",
                provider=self.get_provider_name(),
            )

    async def check_rate_limits(self, estimated_tokens: int = 0) -> None:
        """
        Check if rate limits allow making another request.

        This is a hook for implementing rate limiting logic. Uses the new
        rate limiter system if available, falls back to legacy implementation.

        Args:
            estimated_tokens: Estimated tokens for the request (for token-based rate limiting)

        Raises:
            ProviderRateLimitError: If rate limits are exceeded
        """
        # Use new rate limiter system if available
        if self.rate_limiter is not None:
            try:
                await self.rate_limiter.acquire(tokens=estimated_tokens)
            except Exception as e:
                raise ProviderRateLimitError(
                    str(e),
                    provider=self.get_provider_name(),
                    retry_after=60.0,
                )
            return

        # Legacy rate limiting (for backwards compatibility)
        # Reset counters if a minute has passed
        now = datetime.now(timezone.utc)
        if (now - self._last_reset).total_seconds() >= 60:
            self._request_count = 0
            self._token_count = 0
            self._last_reset = now

        # Check request rate limit
        if self.rate_limit_config.requests_per_minute is not None:
            if self._request_count >= self.rate_limit_config.requests_per_minute:
                raise ProviderRateLimitError(
                    "Request rate limit exceeded",
                    provider=self.get_provider_name(),
                    retry_after=60.0,
                )

        self._request_count += 1

    async def check_budget(self, estimated_cost: float) -> None:
        """
        Check if budget allows making another request.

        Args:
            estimated_cost: Estimated cost of the request in USD

        Raises:
            ProviderQuotaExceededError: If budget is exceeded
        """
        if self.budget_manager is None:
            return

        provider_name = self.get_provider_name()
        allowed, reason = await self.budget_manager.check_budget(provider_name, estimated_cost)

        if not allowed:
            raise ProviderQuotaExceededError(
                reason or "Budget exceeded",
                provider=provider_name,
            )

    async def record_cost(
        self,
        cost: float,
        tokens_input: int,
        tokens_output: int,
        model: str | None = None,
    ) -> None:
        """
        Record the cost of a request for budget tracking.

        Args:
            cost: Actual cost of the request in USD
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            model: Model name (defaults to self.model)
        """
        if self.budget_manager is None:
            return

        provider_name = self.get_provider_name()
        await self.budget_manager.record_cost(
            provider=provider_name,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model or self.model,
        )

    async def record_benchmark(
        self,
        latency: float,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        success: bool,
        error_type: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Record performance benchmarking data for this request.

        Args:
            latency: Request latency in seconds
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            cost: Cost of the request in USD
            success: Whether the request succeeded
            error_type: Type of error if request failed
            model: Model name (defaults to self.model)
        """
        if self.benchmark_tracker is None:
            return

        provider_name = self.get_provider_name()
        await self.benchmark_tracker.record_request(
            provider=provider_name,
            model=model or self.model,
            latency=latency,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            success=success,
            error_type=error_type,
        )

    async def handle_retry(
        self,
        error: ProviderError,
        attempt: int,
    ) -> float:
        """
        Determine if a request should be retried and calculate the delay.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before retry (0 means no retry)

        Raises:
            ProviderError: If the error should not be retried
        """
        if attempt >= self.retry_config.max_retries:
            raise error

        # Determine if this error type should be retried
        should_retry = False

        if isinstance(error, ProviderTimeoutError) and self.retry_config.retry_on_timeout:
            should_retry = True
        elif isinstance(error, ProviderRateLimitError) and self.retry_config.retry_on_rate_limit:
            should_retry = True
            # Use provider-suggested retry_after if available
            if error.retry_after:
                return min(error.retry_after, self.retry_config.max_delay)
        elif isinstance(error, ProviderConnectionError) and self.retry_config.retry_on_server_error:
            should_retry = True

        if not should_retry:
            raise error

        # Calculate exponential backoff delay
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base**attempt)
        delay = min(delay, self.retry_config.max_delay)

        return delay

    async def check_health(self) -> HealthCheckResult:
        """
        Check the health status of this provider.

        This method performs a lightweight health check by sending a minimal test
        request to the provider. Subclasses can override this to implement
        provider-specific health checks.

        The default implementation sends a simple test query and measures:
        - Connectivity (can we reach the API?)
        - Authentication (is our API key valid?)
        - Response time (how fast is the provider responding?)

        Returns:
            HealthCheckResult with status, response time, and any errors

        Example:
            >>> provider = OpenAIProvider()
            >>> health = await provider.check_health()
            >>> if health.is_healthy():
            ...     print("Provider is ready!")
        """
        import time

        start_time = time.time()
        details: dict[str, Any] = {
            "provider": self.get_provider_name(),
            "model": getattr(self, "model", None),
        }

        try:
            # Create a minimal test request
            test_request = ProviderRequest(
                query="test",  # Minimal query
                max_tokens=5,  # Minimal tokens
                temperature=0.0,  # Deterministic
                timeout=10.0,  # Short timeout for health check
            )

            # Attempt to send the request
            response = await self.send_request(test_request)

            # Calculate response time
            response_time = time.time() - start_time
            details["response_time"] = response_time
            details["tokens_used"] = response.tokens_input + response.tokens_output

            # Determine status based on response time
            if response_time < 2.0:
                status = HealthStatus.HEALTHY
            elif response_time < 5.0:
                status = HealthStatus.DEGRADED
                details["reason"] = "Slow response time"
            else:
                status = HealthStatus.DEGRADED
                details["reason"] = "Very slow response time"

            return HealthCheckResult(
                status=status,
                response_time=response_time,
                details=details,
            )

        except ProviderAuthenticationError as e:
            # Authentication failed - provider is unhealthy
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=f"Authentication failed: {str(e)}",
                details={**details, "error_type": "authentication"},
            )

        except ProviderRateLimitError as e:
            # Rate limited - provider is degraded but usable after wait
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                response_time=time.time() - start_time,
                error=f"Rate limited: {str(e)}",
                details={
                    **details,
                    "error_type": "rate_limit",
                    "retry_after": e.retry_after,
                },
            )

        except ProviderConnectionError as e:
            # Cannot connect - provider is unhealthy
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=f"Connection failed: {str(e)}",
                details={**details, "error_type": "connection"},
            )

        except ProviderTimeoutError as e:
            # Timeout - provider is degraded or unhealthy
            response_time = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error=f"Request timed out: {str(e)}",
                details={**details, "error_type": "timeout"},
            )

        except ProviderError as e:
            # Other provider error - provider is likely unhealthy
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=f"Provider error: {str(e)}",
                details={**details, "error_type": "provider_error"},
            )

        except Exception as e:
            # Unexpected error - provider is unhealthy
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                error=f"Unexpected error: {str(e)}",
                details={**details, "error_type": "unexpected"},
            )

    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model})"
