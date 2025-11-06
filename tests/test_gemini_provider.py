"""
Unit tests for GeminiProvider.

Tests cover:
- Provider initialization
- Request sending and response handling
- Token counting with Gemini API
- Cost calculation
- Error handling and mapping
- Model validation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from quorum_mcp.providers.gemini_provider import GeminiProvider
from quorum_mcp.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
    ProviderModelError,
    ProviderQuotaExceededError,
)


@pytest.fixture
def gemini_provider():
    """Create GeminiProvider instance for testing."""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        return GeminiProvider()


@pytest.fixture
def sample_request():
    """Create sample provider request."""
    return ProviderRequest(
        query="What is Python?",
        context="Programming language",
        system_prompt="You are a helpful assistant.",
        max_tokens=1000,
        temperature=0.7,
    )


@pytest.fixture
def mock_gemini_response():
    """Create mock Gemini API response."""
    mock_response = Mock()
    mock_response.text = "Python is a high-level programming language."
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 30
    mock_response.usage_metadata.total_token_count = 80
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].finish_reason = Mock()
    mock_response.candidates[0].finish_reason.name = "STOP"
    return mock_response


class TestGeminiProviderInit:
    """Test GeminiProvider initialization."""

    def test_init_with_api_key_env(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider()
            assert provider.api_key == "test-key"
            assert provider.model == "gemini-2.5-flash"

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        provider = GeminiProvider(api_key="param-key")
        assert provider.api_key == "param-key"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(model="gemini-2.5-pro")
            assert provider.model == "gemini-2.5-pro"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthenticationError, match="GOOGLE_API_KEY"):
                GeminiProvider()

    def test_init_vertexai_without_project(self):
        """Test initialization fails for Vertex AI without project."""
        with pytest.raises(ProviderAuthenticationError, match="Vertex AI requires"):
            GeminiProvider(vertexai=True)

    def test_get_provider_name(self, gemini_provider):
        """Test provider name."""
        assert gemini_provider.get_provider_name() == "gemini"


class TestGeminiProviderSendRequest:
    """Test GeminiProvider request sending."""

    @pytest.mark.asyncio
    async def test_send_request_success(
        self, gemini_provider, sample_request, mock_gemini_response
    ):
        """Test successful request."""
        # Mock the client
        gemini_provider.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_response
        )

        response = await gemini_provider.send_request(sample_request)

        assert isinstance(response, ProviderResponse)
        assert response.content == "Python is a high-level programming language."
        assert response.model == "gemini-2.5-flash"
        assert response.provider == "gemini"
        assert response.tokens_input == 50
        assert response.tokens_output == 30
        assert response.cost > 0

    @pytest.mark.asyncio
    async def test_send_request_with_custom_model(
        self, gemini_provider, sample_request, mock_gemini_response
    ):
        """Test request with custom model in request."""
        sample_request.model = "gemini-2.5-pro"

        gemini_provider.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_response
        )

        response = await gemini_provider.send_request(sample_request)

        # Verify correct model was used in call
        call_args = gemini_provider.client.aio.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_send_request_empty_response(self, gemini_provider, sample_request):
        """Test handling of empty response."""
        mock_response = Mock()
        mock_response.text = None

        gemini_provider.client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(ProviderError, match="Empty response"):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_auth_error(self, gemini_provider, sample_request):
        """Test authentication error handling."""
        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API key is invalid")
        )

        with pytest.raises(ProviderAuthenticationError, match="Authentication failed"):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_rate_limit_error(self, gemini_provider, sample_request):
        """Test rate limit error handling."""
        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        with pytest.raises(ProviderRateLimitError, match="Rate limit exceeded"):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_quota_error(self, gemini_provider, sample_request):
        """Test quota exceeded error handling."""
        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Quota exhausted")
        )

        with pytest.raises(ProviderQuotaExceededError, match="Quota exceeded"):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self, gemini_provider, sample_request):
        """Test timeout error handling."""
        import asyncio

        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        with pytest.raises(ProviderTimeoutError, match="Request timed out"):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_invalid_model(self, gemini_provider, sample_request):
        """Test invalid model error handling."""
        sample_request.model = "invalid-model"

        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Model not found: invalid-model")
        )

        with pytest.raises(ProviderModelError):
            await gemini_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_generic_error(self, gemini_provider, sample_request):
        """Test generic error handling."""
        gemini_provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Unknown error")
        )

        with pytest.raises(ProviderError, match="Unknown error"):
            await gemini_provider.send_request(sample_request)


class TestGeminiProviderTokenCounting:
    """Test GeminiProvider token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, gemini_provider):
        """Test token counting."""
        # Mock the Gemini client's count_tokens method
        mock_response = Mock(total_tokens=10)
        gemini_provider.client.aio.models.count_tokens = AsyncMock(
            return_value=mock_response
        )

        count = await gemini_provider.count_tokens("Hello world")

        assert count == 10
        gemini_provider.client.aio.models.count_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, gemini_provider):
        """Test token counting with empty string."""
        mock_response = Mock(total_tokens=0)
        gemini_provider.client.aio.models.count_tokens = AsyncMock(
            return_value=mock_response
        )

        count = await gemini_provider.count_tokens("")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self, gemini_provider):
        """Test token counting with long text."""
        mock_response = Mock(total_tokens=500)
        gemini_provider.client.aio.models.count_tokens = AsyncMock(
            return_value=mock_response
        )

        long_text = "word " * 1000
        count = await gemini_provider.count_tokens(long_text)

        assert count == 500

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self, gemini_provider):
        """Test token counting fallback on error."""
        gemini_provider.client.aio.models.count_tokens = AsyncMock(
            side_effect=Exception("API error")
        )

        # Should fallback to character estimation (~4 chars per token)
        count = await gemini_provider.count_tokens("Hello world")

        # "Hello world" is 11 chars, so roughly 11/4 = 2-3 tokens
        assert count > 0


class TestGeminiProviderCostCalculation:
    """Test GeminiProvider cost calculation."""

    def test_get_cost_flash_2_5(self, gemini_provider):
        """Test cost calculation for Gemini 2.5 Flash."""
        # Gemini 2.5 Flash: $0.15/1M input, $0.60/1M output
        cost = gemini_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 0.75  # $0.15 + $0.60

    def test_get_cost_pro_2_5(self, gemini_provider):
        """Test cost calculation for Gemini 2.5 Pro."""
        gemini_provider.model = "gemini-2.5-pro"
        # Gemini 2.5 Pro: $1.25/1M input, $10.00/1M output
        cost = gemini_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 11.25  # $1.25 + $10.00

    def test_get_cost_pro_1_5(self, gemini_provider):
        """Test cost calculation for Gemini 1.5 Pro."""
        gemini_provider.model = "gemini-1.5-pro"
        # Gemini 1.5 Pro: $1.25/1M input, $5.00/1M output
        cost = gemini_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 6.25  # $1.25 + $5.00

    def test_get_cost_flash_1_5(self, gemini_provider):
        """Test cost calculation for Gemini 1.5 Flash."""
        gemini_provider.model = "gemini-1.5-flash"
        # Gemini 1.5 Flash: $0.075/1M input, $0.30/1M output
        cost = gemini_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 0.375  # $0.075 + $0.30

    def test_get_cost_small_usage(self, gemini_provider):
        """Test cost calculation for small token usage."""
        # 100 input tokens, 50 output tokens
        cost = gemini_provider.get_cost(100, 50)
        assert cost == pytest.approx(0.000045, rel=1e-5)  # (100*0.15 + 50*0.60) / 1M

    def test_get_cost_zero_tokens(self, gemini_provider):
        """Test cost calculation for zero tokens."""
        cost = gemini_provider.get_cost(0, 0)
        assert cost == 0.0

    def test_get_cost_unknown_model(self, gemini_provider):
        """Test cost calculation for unknown model defaults to Flash pricing."""
        gemini_provider.model = "gemini-unknown-model"
        cost = gemini_provider.get_cost(1_000_000, 1_000_000)
        # Should use default (Flash 2.5) pricing
        assert cost == 0.75


class TestGeminiProviderModelInfo:
    """Test GeminiProvider model information."""

    def test_get_model_info(self, gemini_provider):
        """Test getting model information."""
        info = gemini_provider.get_model_info()

        assert info["provider"] == "gemini"
        assert info["model"] == "gemini-2.5-flash"
        assert info["context_window"] == 200000
        assert "pricing" in info
        assert info["pricing"]["input"] == 0.15
        assert info["pricing"]["output"] == 0.60
        assert info["vertexai"] is False

    def test_get_model_info_custom_model(self, gemini_provider):
        """Test getting model info for custom model."""
        gemini_provider.model = "gemini-1.5-pro"
        info = gemini_provider.get_model_info()

        assert info["model"] == "gemini-1.5-pro"
        assert info["context_window"] == 2000000  # 2M tokens
        assert info["pricing"]["input"] == 1.25
        assert info["pricing"]["output"] == 5.0

    def test_get_model_info_vertexai(self):
        """Test getting model info for Vertex AI provider."""
        with patch("google.genai.Client"):
            provider = GeminiProvider(vertexai=True, project="test", location="us-central1")
            info = provider.get_model_info()

            assert info["vertexai"] is True

    def test_list_available_models(self):
        """Test listing available models."""
        models = GeminiProvider.list_available_models()

        assert len(models) == 4
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-pro" in models
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models


class TestGeminiProviderCleanup:
    """Test GeminiProvider cleanup."""

    @pytest.mark.asyncio
    async def test_aclose(self, gemini_provider):
        """Test closing the async client."""
        gemini_provider.client.aio.aclose = AsyncMock()

        await gemini_provider.aclose()

        gemini_provider.client.aio.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_error_handling(self, gemini_provider):
        """Test aclose handles errors gracefully."""
        gemini_provider.client.aio.aclose = AsyncMock(side_effect=Exception("Close failed"))

        # Should not raise exception
        await gemini_provider.aclose()
