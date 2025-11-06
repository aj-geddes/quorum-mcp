"""
Unit tests for OpenAICompatibleProvider.

Tests cover:
- Provider initialization with custom endpoints
- Request sending to OpenAI-compatible APIs
- Token counting with tiktoken
- Cost calculation (free and paid)
- Error handling and connection errors
- Support for local and cloud providers
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from quorum_mcp.providers.openai_compatible_provider import OpenAICompatibleProvider
from quorum_mcp.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
    ProviderConnectionError,
)


@pytest.fixture
def local_provider():
    """Create OpenAICompatibleProvider for local LM Studio."""
    return OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1",
        model="local-model",
        provider_name="lm-studio",
        api_key="not-needed",  # Local endpoints don't need API key
        pricing=None,  # Free for local
    )


@pytest.fixture
def cloud_provider():
    """Create OpenAICompatibleProvider for OpenRouter."""
    return OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
        provider_name="openrouter",
        api_key="test-api-key",
        pricing={"input": 3.0, "output": 15.0},  # Paid cloud provider
    )


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
def mock_openai_response():
    """Create mock OpenAI-compatible API response."""
    return ChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="local-model",
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


class TestOpenAICompatibleProviderInit:
    """Test OpenAICompatibleProvider initialization."""

    def test_init_local_endpoint(self):
        """Test initialization for local endpoint."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
            provider_name="lm-studio",
        )
        assert provider.base_url == "http://localhost:1234/v1"
        assert provider.model == "local-model"
        assert provider.provider_name == "lm-studio"
        assert provider.api_key == "not-needed"  # Auto-set for local
        assert provider.pricing is None  # Free for local
        assert provider.context_window == 32000  # Default
        assert isinstance(provider.client, AsyncOpenAI)

    def test_init_cloud_endpoint_with_pricing(self):
        """Test initialization for cloud endpoint with pricing."""
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            provider_name="openrouter",
            api_key="test-key",
            pricing={"input": 3.0, "output": 15.0},
        )
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.api_key == "test-key"
        assert provider.pricing == {"input": 3.0, "output": 15.0}

    def test_init_custom_context_window(self):
        """Test initialization with custom context window."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:5000/v1",
            model="llama-2-70b",
            context_window=8192,
        )
        assert provider.context_window == 8192

    def test_init_without_api_key_uses_dummy(self):
        """Test initialization without API key uses dummy value."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="test-model",
        )
        assert provider.api_key == "not-needed"

    def test_client_uses_custom_base_url(self):
        """Test that client is configured with custom base URL."""
        provider = OpenAICompatibleProvider(
            base_url="http://custom:8080/v1",
            model="test-model",
        )
        # OpenAI client adds trailing slash
        assert str(provider.client.base_url).rstrip('/') == "http://custom:8080/v1"


class TestOpenAICompatibleProviderSendRequest:
    """Test request sending and response handling."""

    @pytest.mark.asyncio
    async def test_send_request_success(self, local_provider, sample_request, mock_openai_response):
        """Test successful request to local endpoint."""
        with patch.object(
            local_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await local_provider.send_request(sample_request)

            assert isinstance(response, ProviderResponse)
            assert response.content == "Python is a high-level programming language."
            assert response.model == "local-model"
            assert response.provider == "lm-studio"
            assert response.tokens_input == 50
            assert response.tokens_output == 30
            assert response.cost == 0.0  # Free for local
            assert response.latency > 0
            assert response.metadata["finish_reason"] == "stop"
            assert response.metadata["base_url"] == "http://localhost:1234/v1"

    @pytest.mark.asyncio
    async def test_send_request_with_context(self, local_provider, mock_openai_response):
        """Test request properly formats context and query."""
        request = ProviderRequest(
            query="What is it?",
            context="Python programming language",
            max_tokens=100,
            temperature=0.5,
        )

        mock_create = AsyncMock(return_value=mock_openai_response)

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            mock_create,
        ):
            await local_provider.send_request(request)

            # Verify the messages were formatted correctly
            call_args = mock_create.call_args
            messages = call_args.kwargs["messages"]

            # Should have user message with context
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Python programming language" in messages[0]["content"]
            assert "What is it?" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_send_request_with_system_prompt(self, local_provider, sample_request, mock_openai_response):
        """Test request includes system prompt."""
        mock_create = AsyncMock(return_value=mock_openai_response)

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            mock_create,
        ):
            await local_provider.send_request(sample_request)

            call_args = mock_create.call_args
            messages = call_args.kwargs["messages"]

            # Should have system message
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_send_request_uses_request_model(self, local_provider, mock_openai_response):
        """Test that request model overrides provider default."""
        request = ProviderRequest(
            query="Test",
            model="different-model",
            max_tokens=100,
        )

        mock_create = AsyncMock(return_value=mock_openai_response)

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            mock_create,
        ):
            await local_provider.send_request(request)

            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "different-model"


class TestOpenAICompatibleProviderCost:
    """Test cost calculation for free and paid providers."""

    def test_get_cost_free_local(self, local_provider):
        """Test cost calculation for free local provider."""
        cost = local_provider.get_cost(tokens_input=1000, tokens_output=500)
        assert cost == 0.0

    def test_get_cost_paid_cloud(self, cloud_provider):
        """Test cost calculation for paid cloud provider."""
        # OpenRouter: $3/1M input, $15/1M output
        cost = cloud_provider.get_cost(tokens_input=1_000_000, tokens_output=1_000_000)
        assert cost == 18.0  # $3 + $15

    def test_get_cost_paid_partial(self, cloud_provider):
        """Test cost calculation with partial token counts."""
        # 500K tokens each
        cost = cloud_provider.get_cost(tokens_input=500_000, tokens_output=500_000)
        assert cost == 9.0  # $1.5 + $7.5

    @pytest.mark.asyncio
    async def test_response_includes_correct_cost(self, cloud_provider, sample_request, mock_openai_response):
        """Test that response includes calculated cost."""
        with patch.object(
            cloud_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response,
        ):
            response = await cloud_provider.send_request(sample_request)

            # 50 input tokens * $3/1M = $0.00015
            # 30 output tokens * $15/1M = $0.00045
            # Total = $0.0006
            expected_cost = (50 / 1_000_000) * 3.0 + (30 / 1_000_000) * 15.0
            assert response.cost == pytest.approx(expected_cost, abs=0.0001)


class TestOpenAICompatibleProviderTokens:
    """Test token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens_with_tiktoken(self, local_provider):
        """Test token counting using tiktoken."""
        text = "This is a test message for token counting."
        count = await local_provider.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self, local_provider):
        """Test token counting falls back to estimation if tiktoken fails."""
        text = "Test message"

        # Mock encoding to be None (simulating tiktoken unavailable)
        local_provider.encoding = None

        count = await local_provider.count_tokens(text)
        # Fallback: len(text) // 4
        assert count == len(text) // 4


class TestOpenAICompatibleProviderInfo:
    """Test provider and model information."""

    def test_get_provider_name_local(self, local_provider):
        """Test getting provider name for local endpoint."""
        assert local_provider.get_provider_name() == "lm-studio"

    def test_get_provider_name_cloud(self, cloud_provider):
        """Test getting provider name for cloud endpoint."""
        assert cloud_provider.get_provider_name() == "openrouter"

    def test_get_model_info_local(self, local_provider):
        """Test model info for local provider."""
        info = local_provider.get_model_info()
        assert info["provider"] == "lm-studio"
        assert info["model"] == "local-model"
        assert info["context_window"] == 32000
        assert info["base_url"] == "http://localhost:1234/v1"
        assert info["pricing"] == {"input": 0.0, "output": 0.0}
        assert info["local"] is True  # Contains localhost

    def test_get_model_info_cloud(self, cloud_provider):
        """Test model info for cloud provider."""
        info = cloud_provider.get_model_info()
        assert info["provider"] == "openrouter"
        assert info["model"] == "anthropic/claude-3.5-sonnet"
        assert info["pricing"] == {"input": 3.0, "output": 15.0}
        assert info["local"] is False  # Not localhost

    def test_list_available_models(self):
        """Test listing available models."""
        models = OpenAICompatibleProvider.list_available_models()
        assert isinstance(models, list)
        assert "local-model" in models
        assert "lm-studio" in models


class TestOpenAICompatibleProviderErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error(self, cloud_provider, sample_request):
        """Test authentication error handling."""
        from openai import AuthenticationError

        mock_error = AuthenticationError(
            "Invalid API key",
            response=Mock(),
            body={}
        )

        with patch.object(
            cloud_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderAuthenticationError) as exc_info:
                await cloud_provider.send_request(sample_request)

            assert "Authentication failed" in str(exc_info.value)
            assert exc_info.value.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, cloud_provider, sample_request):
        """Test rate limit error handling."""
        from openai import RateLimitError

        mock_error = RateLimitError(
            "Rate limit exceeded",
            response=Mock(),
            body={}
        )

        with patch.object(
            cloud_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderRateLimitError) as exc_info:
                await cloud_provider.send_request(sample_request)

            assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error(self, local_provider, sample_request):
        """Test timeout error handling."""
        from openai import APITimeoutError

        mock_error = APITimeoutError("Request timed out")

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderTimeoutError) as exc_info:
                await local_provider.send_request(sample_request)

            assert "Request timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error(self, local_provider, sample_request):
        """Test connection error handling."""
        mock_error = Exception("Connection refused")

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderConnectionError) as exc_info:
                await local_provider.send_request(sample_request)

            assert "Cannot connect" in str(exc_info.value)
            assert "localhost:1234" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_error(self, local_provider, sample_request):
        """Test generic API error handling."""
        from openai import APIError

        mock_error = APIError(
            "Server error",
            request=Mock(),
            body={}
        )

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await local_provider.send_request(sample_request)

            assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_choices_error(self, local_provider, sample_request):
        """Test error when no choices returned."""
        empty_response = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="local-model",
            choices=[],  # Empty choices
            usage=CompletionUsage(
                prompt_tokens=50,
                completion_tokens=0,
                total_tokens=50,
            ),
        )

        with patch.object(
            local_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=empty_response,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await local_provider.send_request(sample_request)

            assert "No choices returned" in str(exc_info.value)


class TestOpenAICompatibleProviderExamples:
    """Test real-world usage examples from docstring."""

    def test_lm_studio_example(self):
        """Test LM Studio configuration example."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
            provider_name="lm-studio",
            api_key="not-needed",
        )
        assert provider.provider_name == "lm-studio"
        assert provider.pricing is None  # Free

    def test_textgen_webui_example(self):
        """Test text-generation-webui configuration example."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:5000/v1",
            model="TheBloke/Llama-2-13B-GGUF",
            provider_name="textgen-webui",
        )
        assert provider.provider_name == "textgen-webui"
        assert provider.api_key == "not-needed"  # Auto-set

    def test_openrouter_example(self):
        """Test OpenRouter configuration example."""
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            provider_name="openrouter",
            api_key="test-key",
            pricing={"input": 3.0, "output": 15.0},
        )
        assert provider.provider_name == "openrouter"
        assert provider.pricing is not None  # Paid

    def test_together_ai_example(self):
        """Test Together AI configuration example."""
        provider = OpenAICompatibleProvider(
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3-70b-chat-hf",
            provider_name="together-ai",
            api_key="test-key",
            pricing={"input": 0.9, "output": 0.9},
        )
        assert provider.provider_name == "together-ai"
        assert provider.base_url == "https://api.together.xyz/v1"
