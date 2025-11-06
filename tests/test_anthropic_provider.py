"""
Unit tests for AnthropicProvider.

Tests cover:
- Provider initialization
- Request sending and response handling
- Token counting
- Cost calculation
- Error handling and mapping
- Model validation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from anthropic import AsyncAnthropic
from anthropic.types import Message, ContentBlock, Usage

from quorum_mcp.providers.anthropic_provider import AnthropicProvider
from quorum_mcp.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
    ProviderModelError,
)


@pytest.fixture
def anthropic_provider():
    """Create AnthropicProvider instance for testing."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        return AnthropicProvider()


@pytest.fixture
def sample_request():
    """Create sample provider request."""
    return ProviderRequest(
        prompt="What is Python?",
        context="Programming language",
        system_prompt="You are a helpful assistant.",
        max_tokens=1000,
        temperature=0.7,
    )


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response."""
    return Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            ContentBlock(type="text", text="Python is a high-level programming language.")
        ],
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=50, output_tokens=30),
    )


class TestAnthropicProviderInit:
    """Test AnthropicProvider initialization."""

    def test_init_with_api_key_env(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "test-key"
            assert provider.model == "claude-3-5-sonnet-20241022"
            assert isinstance(provider.client, AsyncAnthropic)

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        provider = AnthropicProvider(api_key="param-key")
        assert provider.api_key == "param-key"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(model="claude-3-opus-20240229")
            assert provider.model == "claude-3-opus-20240229"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthenticationError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()

    def test_get_provider_name(self, anthropic_provider):
        """Test provider name."""
        assert anthropic_provider.get_provider_name() == "anthropic"


class TestAnthropicProviderSendRequest:
    """Test AnthropicProvider request sending."""

    @pytest.mark.asyncio
    async def test_send_request_success(
        self, anthropic_provider, sample_request, mock_anthropic_response
    ):
        """Test successful request."""
        # Mock the client
        anthropic_provider.client.messages.create = AsyncMock(
            return_value=mock_anthropic_response
        )

        response = await anthropic_provider.send_request(sample_request)

        assert isinstance(response, ProviderResponse)
        assert response.content == "Python is a high-level programming language."
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.provider == "anthropic"
        assert response.tokens_input == 50
        assert response.tokens_output == 30
        assert response.cost > 0

    @pytest.mark.asyncio
    async def test_send_request_with_custom_model(
        self, anthropic_provider, sample_request, mock_anthropic_response
    ):
        """Test request with custom model in request."""
        sample_request.model = "claude-3-opus-20240229"
        mock_anthropic_response.model = "claude-3-opus-20240229"

        anthropic_provider.client.messages.create = AsyncMock(
            return_value=mock_anthropic_response
        )

        response = await anthropic_provider.send_request(sample_request)

        assert response.model == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_send_request_auth_error(self, anthropic_provider, sample_request):
        """Test authentication error handling."""
        from anthropic import AuthenticationError

        anthropic_provider.client.messages.create = AsyncMock(
            side_effect=AuthenticationError("Invalid API key")
        )

        with pytest.raises(ProviderAuthenticationError, match="Authentication failed"):
            await anthropic_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_rate_limit_error(
        self, anthropic_provider, sample_request
    ):
        """Test rate limit error handling."""
        from anthropic import RateLimitError

        anthropic_provider.client.messages.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded")
        )

        with pytest.raises(ProviderRateLimitError, match="Rate limit exceeded"):
            await anthropic_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self, anthropic_provider, sample_request):
        """Test timeout error handling."""
        import asyncio

        anthropic_provider.client.messages.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        with pytest.raises(ProviderTimeoutError, match="Request timed out"):
            await anthropic_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_invalid_model(self, anthropic_provider, sample_request):
        """Test invalid model error handling."""
        from anthropic import BadRequestError

        sample_request.model = "invalid-model"

        anthropic_provider.client.messages.create = AsyncMock(
            side_effect=BadRequestError(
                "Invalid model", response=Mock(status_code=400), body={}
            )
        )

        with pytest.raises(ProviderModelError):
            await anthropic_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_generic_error(self, anthropic_provider, sample_request):
        """Test generic error handling."""
        anthropic_provider.client.messages.create = AsyncMock(
            side_effect=Exception("Unknown error")
        )

        with pytest.raises(ProviderError, match="Unknown error"):
            await anthropic_provider.send_request(sample_request)


class TestAnthropicProviderTokenCounting:
    """Test AnthropicProvider token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, anthropic_provider):
        """Test token counting."""
        # Mock the Anthropic client's count_tokens method
        mock_response = Mock(input_tokens=10)
        anthropic_provider.client.messages.count_tokens = AsyncMock(
            return_value=mock_response
        )

        count = await anthropic_provider.count_tokens("Hello world")

        assert count == 10
        anthropic_provider.client.messages.count_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, anthropic_provider):
        """Test token counting with empty string."""
        mock_response = Mock(input_tokens=0)
        anthropic_provider.client.messages.count_tokens = AsyncMock(
            return_value=mock_response
        )

        count = await anthropic_provider.count_tokens("")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self, anthropic_provider):
        """Test token counting with long text."""
        mock_response = Mock(input_tokens=500)
        anthropic_provider.client.messages.count_tokens = AsyncMock(
            return_value=mock_response
        )

        long_text = "word " * 1000
        count = await anthropic_provider.count_tokens(long_text)

        assert count == 500


class TestAnthropicProviderCostCalculation:
    """Test AnthropicProvider cost calculation."""

    def test_get_cost_sonnet_3_5(self, anthropic_provider):
        """Test cost calculation for Claude 3.5 Sonnet."""
        # Claude 3.5 Sonnet: $3/1M input, $15/1M output
        cost = anthropic_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 18.0  # $3 + $15

    def test_get_cost_opus(self, anthropic_provider):
        """Test cost calculation for Claude 3 Opus."""
        anthropic_provider.model = "claude-3-opus-20240229"
        # Claude 3 Opus: $15/1M input, $75/1M output
        cost = anthropic_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 90.0  # $15 + $75

    def test_get_cost_haiku(self, anthropic_provider):
        """Test cost calculation for Claude 3 Haiku."""
        anthropic_provider.model = "claude-3-haiku-20240307"
        # Claude 3 Haiku: $0.25/1M input, $1.25/1M output
        cost = anthropic_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 1.5  # $0.25 + $1.25

    def test_get_cost_small_usage(self, anthropic_provider):
        """Test cost calculation for small token usage."""
        # 100 input tokens, 50 output tokens
        cost = anthropic_provider.get_cost(100, 50)
        assert cost == pytest.approx(0.00105, rel=1e-5)  # (100*3 + 50*15) / 1M

    def test_get_cost_zero_tokens(self, anthropic_provider):
        """Test cost calculation for zero tokens."""
        cost = anthropic_provider.get_cost(0, 0)
        assert cost == 0.0

    def test_get_cost_unknown_model(self, anthropic_provider):
        """Test cost calculation for unknown model defaults to Sonnet pricing."""
        anthropic_provider.model = "claude-unknown-model"
        cost = anthropic_provider.get_cost(1_000_000, 1_000_000)
        # Should use default (Sonnet) pricing
        assert cost == 18.0


class TestAnthropicProviderModelInfo:
    """Test AnthropicProvider model information."""

    def test_get_model_info(self, anthropic_provider):
        """Test getting model information."""
        info = anthropic_provider.get_model_info()

        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-3-5-sonnet-20241022"
        assert info["context_window"] == 200000
        assert "pricing" in info
        assert info["pricing"]["input"] == 3.0
        assert info["pricing"]["output"] == 15.0

    def test_get_model_info_custom_model(self, anthropic_provider):
        """Test getting model info for custom model."""
        anthropic_provider.model = "claude-3-opus-20240229"
        info = anthropic_provider.get_model_info()

        assert info["model"] == "claude-3-opus-20240229"
        assert info["pricing"]["input"] == 15.0
        assert info["pricing"]["output"] == 75.0

    def test_list_available_models(self):
        """Test listing available models."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            models = AnthropicProvider.list_available_models()

            assert len(models) >= 4
            assert "claude-3-5-sonnet-20241022" in models
            assert "claude-3-opus-20240229" in models
            assert "claude-3-sonnet-20240229" in models
            assert "claude-3-haiku-20240307" in models


class TestAnthropicProviderValidation:
    """Test AnthropicProvider validation."""

    def test_validate_model_valid(self, anthropic_provider):
        """Test model validation with valid model."""
        # Should not raise
        anthropic_provider._validate_model("claude-3-5-sonnet-20241022")

    def test_validate_model_invalid(self, anthropic_provider):
        """Test model validation with invalid model."""
        with pytest.raises(ProviderModelError, match="Unsupported model"):
            anthropic_provider._validate_model("invalid-model-name")

    def test_validate_model_none(self, anthropic_provider):
        """Test model validation with None uses default."""
        # Should not raise, uses default model
        anthropic_provider._validate_model(None)
