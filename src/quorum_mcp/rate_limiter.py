"""
Rate Limiting System for Quorum-MCP

Implements token bucket algorithm for rate limiting API requests and tokens
per provider. Ensures compliance with provider rate limits and prevents
API throttling.

Features:
- Per-provider rate limiting
- Request and token-based limits
- Async/await compatible
- Thread-safe with asyncio locks
- Automatic limit reset
- Rate limit status queries
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict

from quorum_mcp.providers.base import ProviderRateLimitError, RateLimitConfig


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Implements the token bucket algorithm where tokens are added at a fixed
    rate and consumed when making requests. When the bucket is empty,
    requests are denied until tokens refill.
    """

    capacity: float  # Maximum tokens in bucket
    refill_rate: float  # Tokens added per second
    tokens: float = field(init=False)  # Current tokens available
    last_refill: float = field(init=False)  # Last refill timestamp
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        """Initialize bucket to full capacity."""
        self.tokens = self.capacity
        self.last_refill = time.time()

    async def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    async def consume(self, tokens: float) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self.lock:
            await self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float, timeout: float = 60.0):
        """
        Wait until enough tokens are available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait in seconds

        Raises:
            ProviderRateLimitError: If timeout is reached
        """
        start_time = time.time()

        while True:
            if await self.consume(tokens):
                return

            # Check timeout
            if time.time() - start_time > timeout:
                raise ProviderRateLimitError(
                    f"Rate limit timeout: unable to acquire {tokens} tokens within {timeout}s"
                )

            # Calculate wait time until we have enough tokens
            async with self.lock:
                await self._refill()
                if self.tokens >= tokens:
                    continue

                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                wait_time = min(wait_time, 1.0)  # Wait at most 1 second at a time

            await asyncio.sleep(wait_time)

    async def get_available(self) -> float:
        """Get number of available tokens."""
        async with self.lock:
            await self._refill()
            return self.tokens

    async def get_wait_time(self, tokens: float) -> float:
        """
        Get estimated wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if immediately available)
        """
        async with self.lock:
            await self._refill()
            if self.tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate


@dataclass
class ProviderRateLimiter:
    """
    Rate limiter for a single provider.

    Manages both request-based and token-based rate limiting using
    token bucket algorithm.
    """

    provider_name: str
    config: RateLimitConfig
    request_bucket: TokenBucket | None = field(init=False, default=None)
    token_bucket: TokenBucket | None = field(init=False, default=None)

    def __post_init__(self):
        """Initialize rate limiting buckets."""
        # Request rate limiting (requests per minute)
        if self.config.requests_per_minute:
            self.request_bucket = TokenBucket(
                capacity=self.config.requests_per_minute,
                refill_rate=self.config.requests_per_minute / 60.0,  # per second
            )

        # Token rate limiting (tokens per minute)
        if self.config.tokens_per_minute:
            self.token_bucket = TokenBucket(
                capacity=self.config.tokens_per_minute,
                refill_rate=self.config.tokens_per_minute / 60.0,  # per second
            )

    async def acquire(self, tokens: int = 0):
        """
        Acquire rate limit permission for a request.

        Args:
            tokens: Number of tokens (for token-based limiting)

        Raises:
            ProviderRateLimitError: If rate limit cannot be acquired
        """
        # Check request rate limit
        if self.request_bucket:
            if not await self.request_bucket.consume(1.0):
                wait_time = await self.request_bucket.get_wait_time(1.0)
                raise ProviderRateLimitError(
                    f"Request rate limit exceeded for {self.provider_name}. "
                    f"Wait {wait_time:.1f}s or reduce request rate.",
                    provider=self.provider_name,
                    retry_after=wait_time,
                )

        # Check token rate limit
        if self.token_bucket and tokens > 0:
            if not await self.token_bucket.consume(float(tokens)):
                wait_time = await self.token_bucket.get_wait_time(float(tokens))
                raise ProviderRateLimitError(
                    f"Token rate limit exceeded for {self.provider_name}. "
                    f"Wait {wait_time:.1f}s or reduce token usage.",
                    provider=self.provider_name,
                    retry_after=wait_time,
                )

    async def wait_for_capacity(self, tokens: int = 0, timeout: float = 60.0):
        """
        Wait until rate limit capacity is available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time in seconds

        Raises:
            ProviderRateLimitError: If timeout is reached
        """
        # Wait for request capacity
        if self.request_bucket:
            await self.request_bucket.wait_for_tokens(1.0, timeout)

        # Wait for token capacity
        if self.token_bucket and tokens > 0:
            await self.token_bucket.wait_for_tokens(float(tokens), timeout)

    async def get_status(self) -> Dict[str, any]:
        """
        Get current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        status = {
            "provider": self.provider_name,
            "requests_per_minute": self.config.requests_per_minute,
            "tokens_per_minute": self.config.tokens_per_minute,
        }

        if self.request_bucket:
            available = await self.request_bucket.get_available()
            status["requests_available"] = int(available)
            status["requests_capacity"] = int(self.request_bucket.capacity)
            status["requests_utilization"] = 1.0 - (available / self.request_bucket.capacity)

        if self.token_bucket:
            available = await self.token_bucket.get_available()
            status["tokens_available"] = int(available)
            status["tokens_capacity"] = int(self.token_bucket.capacity)
            status["tokens_utilization"] = 1.0 - (available / self.token_bucket.capacity)

        return status


class RateLimiterManager:
    """
    Manages rate limiters for all providers.

    Provides centralized rate limiting across the entire system,
    tracking limits per provider.
    """

    def __init__(self):
        """Initialize the rate limiter manager."""
        self._limiters: Dict[str, ProviderRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def register_provider(
        self,
        provider_name: str,
        config: RateLimitConfig
    ) -> ProviderRateLimiter:
        """
        Register a provider with rate limiting.

        Args:
            provider_name: Name of the provider
            config: Rate limit configuration

        Returns:
            ProviderRateLimiter instance
        """
        async with self._lock:
            if provider_name in self._limiters:
                return self._limiters[provider_name]

            limiter = ProviderRateLimiter(provider_name, config)
            self._limiters[provider_name] = limiter
            return limiter

    async def get_limiter(self, provider_name: str) -> ProviderRateLimiter | None:
        """
        Get rate limiter for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderRateLimiter if registered, None otherwise
        """
        return self._limiters.get(provider_name)

    async def acquire(self, provider_name: str, tokens: int = 0):
        """
        Acquire rate limit for a provider.

        Args:
            provider_name: Name of the provider
            tokens: Number of tokens to consume

        Raises:
            ProviderRateLimitError: If rate limit exceeded
        """
        limiter = await self.get_limiter(provider_name)
        if limiter:
            await limiter.acquire(tokens)

    async def get_all_status(self) -> Dict[str, Dict]:
        """
        Get status of all provider rate limiters.

        Returns:
            Dictionary mapping provider names to their status
        """
        status = {}
        for provider_name, limiter in self._limiters.items():
            status[provider_name] = await limiter.get_status()
        return status


# Global rate limiter manager instance
_rate_limiter_manager: RateLimiterManager | None = None


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get the global rate limiter manager instance."""
    global _rate_limiter_manager
    if _rate_limiter_manager is None:
        _rate_limiter_manager = RateLimiterManager()
    return _rate_limiter_manager
