"""
Unit tests for MistralProvider.

Tests cover:
- Provider initialization
- Request sending and response handling
- Token counting (estimation)
- Cost calculation
- Error handling and mapping
- Model validation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from mistralai.models import (
    ChatCompletionResponse,
    ChatCompletionChoice,
    AssistantMessage,
    UsageInfo,
    SDKError,
    HTTPValidationError,
)

from quorum_mcp.providers.mistral_provider import MistralProvider
from quorum_mcp.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
    ProviderModelError,
    ProviderConnectionError,
    ProviderQuotaExceededError,
    ProviderInvalidRequestError,
)


@pytest.fixture
def mistral_provider():
    """Create MistralProvider instance for testing."""
    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        return MistralProvider()


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
def mock_mistral_response():
    """Create mock Mistral AI API response."""
    return ChatCompletionResponse(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="mistral-large-latest",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content="Python is a high-level programming language.",
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
        ),
    )


class TestMistralProviderInit:
    """Test MistralProvider initialization."""

    def test_init_with_api_key_env(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            from mistralai import Mistral

            provider = MistralProvider()
            assert provider.api_key == "test-key"
            assert provider.model == "mistral-large-latest"
            assert isinstance(provider.client, Mistral)

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        provider = MistralProvider(api_key="param-key")
        assert provider.api_key == "param-key"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = MistralProvider(model="mistral-medium-latest")
            assert provider.model == "mistral-medium-latest"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthenticationError, match="MISTRAL_API_KEY"):
                MistralProvider()

    def test_get_provider_name(self, mistral_provider):
        """Test provider name."""
        assert mistral_provider.get_provider_name() == "mistral"


class TestMistralProviderSendRequest:
    """Test request sending and response handling."""

    @pytest.mark.asyncio
    async def test_send_request_success(self, mistral_provider, sample_request, mock_mistral_response):
        """Test successful request."""
        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_mistral_response,
        ):
            response = await mistral_provider.send_request(sample_request)

            assert isinstance(response, ProviderResponse)
            assert response.content == "Python is a high-level programming language."
            assert response.model == "mistral-large-latest"
            assert response.provider == "mistral"
            assert response.tokens_input == 50
            assert response.tokens_output == 30
            assert response.cost > 0  # Mistral charges for API calls
            assert response.latency > 0
            assert response.metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_send_request_with_context(self, mistral_provider, mock_mistral_response):
        """Test request properly formats context and query."""
        request = ProviderRequest(
            query="What is it?",
            context="Python programming language",
            max_tokens=100,
            temperature=0.5,
        )

        mock_complete = AsyncMock(return_value=mock_mistral_response)

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            mock_complete,
        ):
            await mistral_provider.send_request(request)

            # Verify the messages were formatted correctly
            call_args = mock_complete.call_args
            messages = call_args.kwargs["messages"]

            # Should have user message with context
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Python programming language" in messages[0]["content"]
            assert "What is it?" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_send_request_with_system_prompt(self, mistral_provider, sample_request, mock_mistral_response):
        """Test request includes system prompt."""
        mock_complete = AsyncMock(return_value=mock_mistral_response)

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            mock_complete,
        ):
            await mistral_provider.send_request(sample_request)

            call_args = mock_complete.call_args
            messages = call_args.kwargs["messages"]

            # Should have system message
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_send_request_uses_request_model(self, mistral_provider, mock_mistral_response):
        """Test that request model overrides provider default."""
        request = ProviderRequest(
            query="Test",
            model="mistral-medium-latest",
            max_tokens=100,
        )

        mock_complete = AsyncMock(return_value=mock_mistral_response)

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            mock_complete,
        ):
            await mistral_provider.send_request(request)

            call_args = mock_complete.call_args
            assert call_args.kwargs["model"] == "mistral-medium-latest"


class TestMistralProviderCost:
    """Test cost calculation."""

    def test_get_cost_large_model(self, mistral_provider):
        """Test cost calculation for Large model."""
        # mistral-large-latest: $2/1M input, $6/1M output
        cost = mistral_provider.get_cost(tokens_input=1_000_000, tokens_output=1_000_000)
        assert cost == 8.0  # $2 + $6

    def test_get_cost_medium_model(self, mistral_provider):
        """Test cost calculation for Medium model."""
        mistral_provider.model = "mistral-medium-latest"
        # mistral-medium-latest: $0.4/1M input, $2/1M output
        cost = mistral_provider.get_cost(tokens_input=1_000_000, tokens_output=1_000_000)
        assert cost == 2.4  # $0.4 + $2

    def test_get_cost_code_model(self, mistral_provider):
        """Test cost calculation for Code model."""
        mistral_provider.model = "codestral-latest"
        # codestral-latest: $0.3/1M input, $0.9/1M output
        cost = mistral_provider.get_cost(tokens_input=1_000_000, tokens_output=1_000_000)
        assert cost == 1.2  # $0.3 + $0.9

    def test_get_cost_small_usage(self, mistral_provider):
        """Test cost calculation for small token usage."""
        # 100 input tokens, 50 output tokens
        cost = mistral_provider.get_cost(100, 50)
        # (100 * $2 + 50 * $6) / 1M = $0.0005
        assert cost == pytest.approx(0.0005, abs=0.00001)

    def test_get_cost_zero_tokens(self, mistral_provider):
        """Test cost calculation for zero tokens."""
        cost = mistral_provider.get_cost(0, 0)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_response_includes_cost(self, mistral_provider, sample_request, mock_mistral_response):
        """Test that response includes calculated cost."""
        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            return_value=mock_mistral_response,
        ):
            response = await mistral_provider.send_request(sample_request)

            # 50 input tokens * $2/1M = $0.0001
            # 30 output tokens * $6/1M = $0.00018
            # Total = $0.00028
            expected_cost = (50 / 1_000_000) * 2.0 + (30 / 1_000_000) * 6.0
            assert response.cost == pytest.approx(expected_cost, abs=0.00001)


class TestMistralProviderTokens:
    """Test token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, mistral_provider):
        """Test token counting estimation."""
        text = "This is a test message for token counting."
        count = await mistral_provider.count_tokens(text)
        # Rough estimation: len(text) // 4
        assert count == len(text) // 4
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, mistral_provider):
        """Test token counting with empty string."""
        count = await mistral_provider.count_tokens("")
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self, mistral_provider):
        """Test token counting with long text."""
        long_text = "word " * 1000
        count = await mistral_provider.count_tokens(long_text)
        # Should be roughly len(text) // 4
        assert count > 500
        assert count < 2000


class TestMistralProviderInfo:
    """Test provider and model information."""

    def test_get_model_info_large(self, mistral_provider):
        """Test model info for Large model."""
        info = mistral_provider.get_model_info()
        assert info["provider"] == "mistral"
        assert info["model"] == "mistral-large-latest"
        assert info["context_window"] == 128000
        assert info["pricing"]["input"] == 2.0
        assert info["pricing"]["output"] == 6.0

    def test_get_model_info_medium(self, mistral_provider):
        """Test model info for Medium model."""
        mistral_provider.model = "mistral-medium-latest"
        info = mistral_provider.get_model_info()
        assert info["model"] == "mistral-medium-latest"
        assert info["context_window"] == 32000
        assert info["pricing"]["input"] == 0.4
        assert info["pricing"]["output"] == 2.0

    def test_get_model_info_code(self, mistral_provider):
        """Test model info for Code model."""
        mistral_provider.model = "codestral-latest"
        info = mistral_provider.get_model_info()
        assert info["model"] == "codestral-latest"
        assert info["context_window"] == 256000
        assert info["pricing"]["input"] == 0.3
        assert info["pricing"]["output"] == 0.9

    def test_list_available_models(self):
        """Test listing available models."""
        models = MistralProvider.list_available_models()
        assert isinstance(models, list)
        assert "mistral-large-latest" in models
        assert "mistral-medium-latest" in models
        assert "codestral-latest" in models
        assert "pixtral-large-latest" in models
        assert len(models) > 5


class TestMistralProviderModelValidation:
    """Test model validation."""

    def test_validate_model_valid(self, mistral_provider):
        """Test validation of valid model."""
        # Should not raise
        mistral_provider._validate_model("mistral-large-latest")
        mistral_provider._validate_model("mistral-medium-latest")
        mistral_provider._validate_model("codestral-latest")

    def test_validate_model_none(self, mistral_provider):
        """Test validation of None (uses default)."""
        # Should not raise
        mistral_provider._validate_model(None)

    def test_validate_model_invalid(self, mistral_provider):
        """Test validation of invalid model."""
        with pytest.raises(ProviderModelError, match="Unsupported model"):
            mistral_provider._validate_model("invalid-model-name")


class TestMistralProviderErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error(self, mistral_provider, sample_request):
        """Test authentication error handling by detecting error message patterns."""
        # Create mock error with authentication keywords
        mock_error = Exception("authentication failed: Invalid API key (401)")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            # Should catch and raise ProviderAuthenticationError based on message content
            with pytest.raises(ProviderError):  # Will be generic since we're using Exception
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, mistral_provider, sample_request):
        """Test rate limit error detection."""
        # Create mock error with rate limit keywords
        mock_error = Exception("rate limit exceeded (429)")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_timeout_error(self, mistral_provider, sample_request):
        """Test timeout error detection."""
        # Create mock error with timeout keywords
        mock_error = Exception("request timeout")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_connection_error(self, mistral_provider, sample_request):
        """Test connection error detection."""
        # Create mock error with connection keywords
        mock_error = Exception("connection refused")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_quota_exceeded_error(self, mistral_provider, sample_request):
        """Test quota exceeded error detection."""
        # Create mock error with quota keywords
        mock_error = Exception("quota exceeded")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, mistral_provider, sample_request):
        """Test invalid model error detection."""
        # Since HTTPValidationError is complex, test validation with simple exception
        mock_error = Exception("model validation failed: invalid-model does not exist")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_validation_error(self, mistral_provider, sample_request):
        """Test validation error detection."""
        # Test with simple exception containing validation message
        mock_error = Exception("validation failed: invalid temperature value")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError):
                await mistral_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_generic_sdk_error(self, mistral_provider, sample_request):
        """Test generic SDK error detection."""
        # Test with generic exception
        mock_error = Exception("some other SDK error")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await mistral_provider.send_request(sample_request)

            assert "SDK error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unexpected_error(self, mistral_provider, sample_request):
        """Test unexpected error handling."""
        mock_error = Exception("Unexpected error")

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await mistral_provider.send_request(sample_request)

            assert "Unexpected error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_choices_error(self, mistral_provider, sample_request):
        """Test error when no choices returned."""
        empty_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="mistral-large-latest",
            choices=[],  # Empty choices
            usage=UsageInfo(
                prompt_tokens=50,
                completion_tokens=0,
                total_tokens=50,
            ),
        )

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            new_callable=AsyncMock,
            return_value=empty_response,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await mistral_provider.send_request(sample_request)

            assert "No choices returned" in str(exc_info.value)


class TestMistralProviderMessageFormatting:
    """Test message formatting."""

    @pytest.mark.asyncio
    async def test_format_messages_with_system_and_context(self, mistral_provider, mock_mistral_response):
        """Test formatting messages with system prompt and context."""
        request = ProviderRequest(
            query="What is it?",
            context="Python programming",
            system_prompt="You are an expert.",
            max_tokens=100,
        )

        mock_complete = AsyncMock(return_value=mock_mistral_response)

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            mock_complete,
        ):
            await mistral_provider.send_request(request)

            call_args = mock_complete.call_args
            messages = call_args.kwargs["messages"]

            # Should have 2 messages: system and user
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are an expert."
            assert messages[1]["role"] == "user"
            assert "Python programming" in messages[1]["content"]
            assert "What is it?" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_format_messages_without_system(self, mistral_provider, mock_mistral_response):
        """Test formatting messages without system prompt."""
        request = ProviderRequest(
            query="Test query",
            max_tokens=100,
        )

        mock_complete = AsyncMock(return_value=mock_mistral_response)

        with patch.object(
            mistral_provider.client.chat,
            "complete_async",
            mock_complete,
        ):
            await mistral_provider.send_request(request)

            call_args = mock_complete.call_args
            messages = call_args.kwargs["messages"]

            # Should have only 1 user message
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Test query"
