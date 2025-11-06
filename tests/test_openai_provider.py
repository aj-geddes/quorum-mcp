"""
Unit tests for OpenAIProvider.

Tests cover:
- Provider initialization
- Request sending and response handling
- Token counting with tiktoken
- Cost calculation
- Error handling and mapping
- Model validation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from quorum_mcp.providers.openai_provider import OpenAIProvider
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
def openai_provider():
    """Create OpenAIProvider instance for testing."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return OpenAIProvider()


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
def mock_openai_response():
    """Create mock OpenAI API response."""
    return ChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Python is a high-level programming language.",
                ),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
        ),
    )


class TestOpenAIProviderInit:
    """Test OpenAIProvider initialization."""

    def test_init_with_api_key_env(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "test-key"
            assert provider.model == "gpt-4o"
            assert isinstance(provider.client, AsyncOpenAI)

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        provider = OpenAIProvider(api_key="param-key")
        assert provider.api_key == "param-key"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(model="gpt-4o-mini")
            assert provider.model == "gpt-4o-mini"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthenticationError, match="OPENAI_API_KEY"):
                OpenAIProvider()

    def test_get_provider_name(self, openai_provider):
        """Test provider name."""
        assert openai_provider.get_provider_name() == "openai"


class TestOpenAIProviderSendRequest:
    """Test OpenAIProvider request sending."""

    @pytest.mark.asyncio
    async def test_send_request_success(
        self, openai_provider, sample_request, mock_openai_response
    ):
        """Test successful request."""
        # Mock the client
        openai_provider.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        response = await openai_provider.send_request(sample_request)

        assert isinstance(response, ProviderResponse)
        assert response.content == "Python is a high-level programming language."
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.tokens_input == 50
        assert response.tokens_output == 30
        assert response.cost > 0

    @pytest.mark.asyncio
    async def test_send_request_with_custom_model(
        self, openai_provider, sample_request, mock_openai_response
    ):
        """Test request with custom model in request."""
        sample_request.model = "gpt-4o-mini"
        mock_openai_response.model = "gpt-4o-mini"

        openai_provider.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        response = await openai_provider.send_request(sample_request)

        assert response.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_send_request_empty_response(
        self, openai_provider, sample_request, mock_openai_response
    ):
        """Test handling of empty response content."""
        mock_openai_response.choices[0].message.content = None

        openai_provider.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        with pytest.raises(ProviderError, match="Empty response"):
            await openai_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_auth_error(self, openai_provider, sample_request):
        """Test authentication error handling."""
        from openai import AuthenticationError

        openai_provider.client.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError(
                "Invalid API key", response=Mock(), body={}
            )
        )

        with pytest.raises(ProviderAuthenticationError, match="Authentication failed"):
            await openai_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_rate_limit_error(self, openai_provider, sample_request):
        """Test rate limit error handling."""
        from openai import RateLimitError

        openai_provider.client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded", response=Mock(), body={})
        )

        with pytest.raises(ProviderRateLimitError, match="Rate limit exceeded"):
            await openai_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self, openai_provider, sample_request):
        """Test timeout error handling."""
        import asyncio

        openai_provider.client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        with pytest.raises(ProviderTimeoutError, match="Request timed out"):
            await openai_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_invalid_model(self, openai_provider, sample_request):
        """Test invalid model error handling."""
        from openai import BadRequestError

        sample_request.model = "invalid-model"

        openai_provider.client.chat.completions.create = AsyncMock(
            side_effect=BadRequestError(
                "Invalid model", response=Mock(status_code=400), body={}
            )
        )

        with pytest.raises(ProviderModelError):
            await openai_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_generic_error(self, openai_provider, sample_request):
        """Test generic error handling."""
        openai_provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Unknown error")
        )

        with pytest.raises(ProviderError, match="Unknown error"):
            await openai_provider.send_request(sample_request)


class TestOpenAIProviderTokenCounting:
    """Test OpenAIProvider token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, openai_provider):
        """Test token counting."""
        # tiktoken should encode this text
        count = await openai_provider.count_tokens("Hello world")

        # "Hello world" is typically 2-3 tokens depending on encoding
        assert count > 0
        assert count < 10

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, openai_provider):
        """Test token counting with empty string."""
        count = await openai_provider.count_tokens("")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self, openai_provider):
        """Test token counting with long text."""
        long_text = "word " * 1000
        count = await openai_provider.count_tokens(long_text)

        # Should be roughly 1000-2000 tokens
        assert count > 500
        assert count < 3000

    @pytest.mark.asyncio
    async def test_count_tokens_special_characters(self, openai_provider):
        """Test token counting with special characters."""
        text = "🚀 Python is amazing! 🐍"
        count = await openai_provider.count_tokens(text)

        assert count > 0


class TestOpenAIProviderCostCalculation:
    """Test OpenAIProvider cost calculation."""

    def test_get_cost_gpt4o(self, openai_provider):
        """Test cost calculation for GPT-4o."""
        # GPT-4o: $2.5/1M input, $10/1M output
        cost = openai_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 12.5  # $2.5 + $10

    def test_get_cost_gpt4o_mini(self, openai_provider):
        """Test cost calculation for GPT-4o-mini."""
        openai_provider.model = "gpt-4o-mini"
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        cost = openai_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 0.75  # $0.15 + $0.60

    def test_get_cost_gpt4_turbo(self, openai_provider):
        """Test cost calculation for GPT-4-turbo."""
        openai_provider.model = "gpt-4-turbo"
        # GPT-4-turbo: $10/1M input, $30/1M output
        cost = openai_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 40.0  # $10 + $30

    def test_get_cost_gpt4(self, openai_provider):
        """Test cost calculation for GPT-4."""
        openai_provider.model = "gpt-4"
        # GPT-4: $30/1M input, $60/1M output
        cost = openai_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 90.0  # $30 + $60

    def test_get_cost_small_usage(self, openai_provider):
        """Test cost calculation for small token usage."""
        # 100 input tokens, 50 output tokens
        cost = openai_provider.get_cost(100, 50)
        assert cost == pytest.approx(0.00075, rel=1e-5)  # (100*2.5 + 50*10) / 1M

    def test_get_cost_zero_tokens(self, openai_provider):
        """Test cost calculation for zero tokens."""
        cost = openai_provider.get_cost(0, 0)
        assert cost == 0.0

    def test_get_cost_unknown_model(self, openai_provider):
        """Test cost calculation for unknown model defaults to GPT-4o pricing."""
        openai_provider.model = "gpt-unknown-model"
        cost = openai_provider.get_cost(1_000_000, 1_000_000)
        # Should use default (GPT-4o) pricing
        assert cost == 12.5


class TestOpenAIProviderModelInfo:
    """Test OpenAIProvider model information."""

    def test_get_model_info(self, openai_provider):
        """Test getting model information."""
        info = openai_provider.get_model_info()

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o"
        assert info["context_window"] == 128000
        assert "pricing" in info
        assert info["pricing"]["input"] == 2.5
        assert info["pricing"]["output"] == 10.0

    def test_get_model_info_custom_model(self, openai_provider):
        """Test getting model info for custom model."""
        openai_provider.model = "gpt-4o-mini"
        info = openai_provider.get_model_info()

        assert info["model"] == "gpt-4o-mini"
        assert info["pricing"]["input"] == 0.15
        assert info["pricing"]["output"] == 0.60

    def test_list_available_models(self):
        """Test listing available models."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            models = OpenAIProvider.list_available_models()

            assert len(models) >= 5
            assert "gpt-4o" in models
            assert "gpt-4o-mini" in models
            assert "gpt-4-turbo" in models
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models


class TestOpenAIProviderValidation:
    """Test OpenAIProvider validation."""

    def test_validate_model_valid(self, openai_provider):
        """Test model validation with valid model."""
        # Should not raise
        openai_provider._validate_model("gpt-4o")

    def test_validate_model_invalid(self, openai_provider):
        """Test model validation with invalid model."""
        with pytest.raises(ProviderModelError, match="Unsupported model"):
            openai_provider._validate_model("invalid-model-name")

    def test_validate_model_none(self, openai_provider):
        """Test model validation with None uses default."""
        # Should not raise, uses default model
        openai_provider._validate_model(None)


class TestOpenAIProviderMessageFormatting:
    """Test OpenAI message formatting."""

    def test_format_messages_with_system_prompt(self, openai_provider, sample_request):
        """Test message formatting with system prompt."""
        messages = openai_provider._format_messages(sample_request)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert "What is Python?" in messages[1]["content"]

    def test_format_messages_without_system_prompt(self, openai_provider):
        """Test message formatting without system prompt."""
        request = ProviderRequest(
            prompt="What is Python?",
            context="",
            system_prompt=None,
            max_tokens=1000,
            temperature=0.7,
        )

        messages = openai_provider._format_messages(request)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Python?"

    def test_format_messages_with_context(self, openai_provider):
        """Test message formatting with context."""
        request = ProviderRequest(
            prompt="What is it?",
            context="Python is a programming language.",
            system_prompt=None,
            max_tokens=1000,
            temperature=0.7,
        )

        messages = openai_provider._format_messages(request)

        assert len(messages) == 1
        assert "Python is a programming language" in messages[0]["content"]
        assert "What is it?" in messages[0]["content"]
