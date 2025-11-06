"""
Unit tests for Provider Health Monitoring.

Tests cover:
- HealthStatus enum
- HealthCheckResult model
- Provider check_health() method
- Health check for various error conditions
- Integration with different providers
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import time

from quorum_mcp.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderConnectionError,
    ProviderTimeoutError,
    HealthStatus,
    HealthCheckResult,
)
from quorum_mcp.providers.openai_provider import OpenAIProvider
from quorum_mcp.providers.mistral_provider import MistralProvider


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test that HealthStatus has expected values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_health_status_comparison(self):
        """Test HealthStatus comparison."""
        assert HealthStatus.HEALTHY != HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED != HealthStatus.UNHEALTHY


class TestHealthCheckResult:
    """Test HealthCheckResult model."""

    def test_health_check_result_healthy(self):
        """Test creating a healthy health check result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            response_time=0.5,
            details={"provider": "openai"},
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 0.5
        assert result.error is None
        assert result.is_healthy()
        assert result.is_usable()

    def test_health_check_result_degraded(self):
        """Test creating a degraded health check result."""
        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            response_time=3.5,
            error="Slow response",
            details={"reason": "high latency"},
        )

        assert result.status == HealthStatus.DEGRADED
        assert not result.is_healthy()
        assert result.is_usable()  # Degraded is still usable
        assert result.error == "Slow response"

    def test_health_check_result_unhealthy(self):
        """Test creating an unhealthy health check result."""
        result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            response_time=10.0,
            error="Connection failed",
            details={"error_type": "connection"},
        )

        assert result.status == HealthStatus.UNHEALTHY
        assert not result.is_healthy()
        assert not result.is_usable()
        assert result.error == "Connection failed"

    def test_health_check_result_timestamp(self):
        """Test that timestamp is automatically set."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
        )

        assert result.timestamp is not None


class TestProviderHealthCheck:
    """Test Provider.check_health() method."""

    @pytest.mark.asyncio
    async def test_check_health_success_fast(self):
        """Test health check with fast successful response."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock a fast successful response
        mock_response = ProviderResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=0.5,
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, return_value=mock_response):
            health = await provider.check_health()

            assert health.status == HealthStatus.HEALTHY
            assert health.response_time < 2.0
            assert health.error is None
            assert health.is_healthy()
            assert health.is_usable()
            assert health.details["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_check_health_success_slow(self):
        """Test health check with slow but successful response."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock a slow successful response
        mock_response = ProviderResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=3.5,
        )

        async def slow_response(request):
            await asyncio.sleep(3.5)
            return mock_response

        import asyncio

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=slow_response):
            health = await provider.check_health()

            # Should be degraded due to slow response
            assert health.status == HealthStatus.DEGRADED
            assert health.response_time >= 2.0
            assert health.error is None  # No error, just slow
            assert not health.is_healthy()
            assert health.is_usable()  # Degraded is still usable
            assert "reason" in health.details

    @pytest.mark.asyncio
    async def test_check_health_authentication_error(self):
        """Test health check with authentication error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock authentication error
        mock_error = ProviderAuthenticationError(
            "Invalid API key",
            provider="openai",
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            assert health.status == HealthStatus.UNHEALTHY
            assert health.error is not None
            assert "Authentication failed" in health.error
            assert not health.is_healthy()
            assert not health.is_usable()
            assert health.details["error_type"] == "authentication"

    @pytest.mark.asyncio
    async def test_check_health_rate_limit_error(self):
        """Test health check with rate limit error."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = MistralProvider()

        # Mock rate limit error
        mock_error = ProviderRateLimitError(
            "Rate limit exceeded",
            provider="mistral",
            retry_after=60.0,
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            # Rate limit means degraded (usable after waiting)
            assert health.status == HealthStatus.DEGRADED
            assert health.error is not None
            assert "Rate limited" in health.error
            assert not health.is_healthy()
            assert health.is_usable()  # Can use after waiting
            assert health.details["error_type"] == "rate_limit"
            assert health.details["retry_after"] == 60.0

    @pytest.mark.asyncio
    async def test_check_health_connection_error(self):
        """Test health check with connection error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock connection error
        mock_error = ProviderConnectionError(
            "Cannot connect to API",
            provider="openai",
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            assert health.status == HealthStatus.UNHEALTHY
            assert health.error is not None
            assert "Connection failed" in health.error
            assert not health.is_healthy()
            assert not health.is_usable()
            assert health.details["error_type"] == "connection"

    @pytest.mark.asyncio
    async def test_check_health_timeout_error(self):
        """Test health check with timeout error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock timeout error
        mock_error = ProviderTimeoutError(
            "Request timed out",
            provider="openai",
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            assert health.status == HealthStatus.UNHEALTHY
            assert health.error is not None
            assert "Request timed out" in health.error
            assert not health.is_healthy()
            assert not health.is_usable()
            assert health.details["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_check_health_generic_provider_error(self):
        """Test health check with generic provider error."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = MistralProvider()

        # Mock generic provider error
        mock_error = ProviderError(
            "Something went wrong",
            provider="mistral",
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            assert health.status == HealthStatus.UNHEALTHY
            assert health.error is not None
            assert "Provider error" in health.error
            assert not health.is_healthy()
            assert not health.is_usable()
            assert health.details["error_type"] == "provider_error"

    @pytest.mark.asyncio
    async def test_check_health_unexpected_error(self):
        """Test health check with unexpected error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()

        # Mock unexpected error
        mock_error = Exception("Unexpected error occurred")

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=mock_error):
            health = await provider.check_health()

            assert health.status == HealthStatus.UNHEALTHY
            assert health.error is not None
            assert "Unexpected error" in health.error
            assert not health.is_healthy()
            assert not health.is_usable()
            assert health.details["error_type"] == "unexpected"

    @pytest.mark.asyncio
    async def test_check_health_includes_details(self):
        """Test that health check includes provider details."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(model="gpt-4o-mini")

        mock_response = ProviderResponse(
            content="test",
            model="gpt-4o-mini",
            provider="openai",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=0.3,
        )

        with patch.object(provider, "send_request", new_callable=AsyncMock, return_value=mock_response):
            health = await provider.check_health()

            assert health.details["provider"] == "openai"
            assert health.details["model"] == "gpt-4o-mini"
            assert "tokens_used" in health.details
            assert health.details["tokens_used"] == 2  # 1 input + 1 output

    @pytest.mark.asyncio
    async def test_check_health_measures_response_time(self):
        """Test that health check accurately measures response time."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = MistralProvider()

        mock_response = ProviderResponse(
            content="test",
            model="mistral-large-latest",
            provider="mistral",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=1.0,
        )

        async def measured_response(request):
            import asyncio

            await asyncio.sleep(1.0)
            return mock_response

        with patch.object(provider, "send_request", new_callable=AsyncMock, side_effect=measured_response):
            start = time.time()
            health = await provider.check_health()
            elapsed = time.time() - start

            # Response time should be approximately 1 second
            assert health.response_time >= 1.0
            assert health.response_time <= 1.5  # Allow some margin
            assert elapsed >= 1.0


class TestHealthCheckIntegration:
    """Test health check integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_providers_health_check(self):
        """Test checking health of multiple providers."""
        providers = []

        # OpenAI provider
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            openai_provider = OpenAIProvider()
            providers.append(openai_provider)

        # Mistral provider
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            mistral_provider = MistralProvider()
            providers.append(mistral_provider)

        # Mock responses for both
        mock_response = ProviderResponse(
            content="test",
            model="test-model",
            provider="test",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=0.5,
        )

        for provider in providers:
            with patch.object(provider, "send_request", new_callable=AsyncMock, return_value=mock_response):
                health = await provider.check_health()
                assert health.is_healthy()

    @pytest.mark.asyncio
    async def test_health_check_concurrent(self):
        """Test concurrent health checks."""
        import asyncio

        mock_response = ProviderResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            tokens_input=1,
            tokens_output=1,
            cost=0.0,
            latency=0.5,
        )

        providers = []
        for _ in range(3):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                provider = OpenAIProvider()
                # Patch send_request before adding to list
                provider.send_request = AsyncMock(return_value=mock_response)
                providers.append(provider)

        # Run health checks concurrently
        tasks = [provider.check_health() for provider in providers]
        results = await asyncio.gather(*tasks)

        # All should be healthy
        assert all(r.is_healthy() for r in results)
        assert len(results) == 3
