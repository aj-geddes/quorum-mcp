"""
Unit tests for OllamaProvider.

Tests cover:
- Provider initialization
- Request sending and response handling
- Token counting estimation
- Cost calculation (always $0)
- Error handling and mapping
- Model availability checking
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from ollama import ResponseError

from quorum_mcp.providers.ollama_provider import OllamaProvider
from quorum_mcp.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderModelError,
    ProviderInvalidRequestError,
)


@pytest.fixture
def ollama_provider():
    """Create OllamaProvider instance for testing."""
    with patch("ollama.AsyncClient"):
        return OllamaProvider(model="llama3.2", host="http://localhost:11434")


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
def mock_ollama_response():
    """Create mock Ollama API response."""
    return {
        "message": {
            "role": "assistant",
            "content": "Python is a high-level programming language.",
        },
        "done": True,
        "eval_count": 30,
        "eval_duration": 123456789,
        "total_duration": 234567890,
    }


class TestOllamaProviderInit:
    """Test OllamaProvider initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        with patch("ollama.AsyncClient"):
            provider = OllamaProvider()
            assert provider.model == "llama3.2"
            assert provider.host == "http://localhost:11434"
            assert provider.timeout == 120.0

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch("ollama.AsyncClient"):
            provider = OllamaProvider(model="mistral")
            assert provider.model == "mistral"

    def test_init_with_custom_host(self):
        """Test initialization with custom host."""
        with patch("ollama.AsyncClient"):
            provider = OllamaProvider(host="http://custom-host:8080")
            assert provider.host == "http://custom-host:8080"

    def test_init_with_env_var(self):
        """Test initialization uses OLLAMA_HOST env var."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://env-host:9000"}):
            with patch("ollama.AsyncClient"):
                provider = OllamaProvider()
                assert provider.host == "http://env-host:9000"

    def test_get_provider_name(self, ollama_provider):
        """Test provider name."""
        assert ollama_provider.get_provider_name() == "ollama"


class TestOllamaProviderSendRequest:
    """Test OllamaProvider request sending."""

    @pytest.mark.asyncio
    async def test_send_request_success(
        self, ollama_provider, sample_request, mock_ollama_response
    ):
        """Test successful request."""
        # Mock the client
        ollama_provider.client.chat = AsyncMock(return_value=mock_ollama_response)

        response = await ollama_provider.send_request(sample_request)

        assert isinstance(response, ProviderResponse)
        assert response.content == "Python is a high-level programming language."
        assert response.model == "llama3.2"
        assert response.provider == "ollama"
        assert response.tokens_input > 0
        assert response.tokens_output > 0
        assert response.cost == 0.0  # Local inference is free!

    @pytest.mark.asyncio
    async def test_send_request_with_custom_model(
        self, ollama_provider, sample_request, mock_ollama_response
    ):
        """Test request with custom model in request."""
        sample_request.model = "mistral"

        ollama_provider.client.chat = AsyncMock(return_value=mock_ollama_response)

        response = await ollama_provider.send_request(sample_request)

        # Verify correct model was used in call
        call_args = ollama_provider.client.chat.call_args
        assert call_args.kwargs["model"] == "mistral"

    @pytest.mark.asyncio
    async def test_send_request_without_context(self, ollama_provider, mock_ollama_response):
        """Test request without context."""
        request = ProviderRequest(query="Hello")

        ollama_provider.client.chat = AsyncMock(return_value=mock_ollama_response)

        response = await ollama_provider.send_request(request)

        # Verify messages structure
        call_args = ollama_provider.client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_send_request_with_system_prompt(
        self, ollama_provider, sample_request, mock_ollama_response
    ):
        """Test request includes system prompt."""
        ollama_provider.client.chat = AsyncMock(return_value=mock_ollama_response)

        await ollama_provider.send_request(sample_request)

        # Verify system message was included
        call_args = ollama_provider.client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_send_request_model_not_found(self, ollama_provider, sample_request):
        """Test model not found error handling."""
        error = ResponseError(error="model not found", status_code=404)
        ollama_provider.client.chat = AsyncMock(side_effect=error)

        with pytest.raises(ProviderModelError, match="Model not found"):
            await ollama_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_connection_error(self, ollama_provider, sample_request):
        """Test connection error handling."""
        ollama_provider.client.chat = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with pytest.raises(ProviderConnectionError, match="Cannot connect to Ollama"):
            await ollama_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self, ollama_provider, sample_request):
        """Test timeout error handling."""
        ollama_provider.client.chat = AsyncMock(side_effect=TimeoutError("Timed out"))

        with pytest.raises(ProviderTimeoutError, match="Request timed out"):
            await ollama_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_invalid_request(self, ollama_provider, sample_request):
        """Test invalid request error handling."""
        error = ResponseError(error="invalid request", status_code=400)
        ollama_provider.client.chat = AsyncMock(side_effect=error)

        with pytest.raises(ProviderInvalidRequestError, match="Invalid request"):
            await ollama_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_generic_error(self, ollama_provider, sample_request):
        """Test generic error handling."""
        ollama_provider.client.chat = AsyncMock(side_effect=Exception("Unknown error"))

        with pytest.raises(ProviderError, match="Unknown error"):
            await ollama_provider.send_request(sample_request)

    @pytest.mark.asyncio
    async def test_send_request_options_passed(
        self, ollama_provider, sample_request, mock_ollama_response
    ):
        """Test that options are correctly passed to API."""
        ollama_provider.client.chat = AsyncMock(return_value=mock_ollama_response)

        await ollama_provider.send_request(sample_request)

        # Verify options were passed
        call_args = ollama_provider.client.chat.call_args
        options = call_args.kwargs["options"]
        assert options["temperature"] == 0.7
        assert options["num_predict"] == 1000


class TestOllamaProviderTokenCounting:
    """Test OllamaProvider token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, ollama_provider):
        """Test token counting estimation."""
        count = await ollama_provider.count_tokens("Hello world")

        # Rough estimation: ~4 chars per token
        # "Hello world" is 11 chars, so roughly 11/4 = 2-3 tokens
        assert count > 0
        assert count == 11 // 4

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, ollama_provider):
        """Test token counting with empty string."""
        count = await ollama_provider.count_tokens("")

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self, ollama_provider):
        """Test token counting with long text."""
        long_text = "word " * 1000
        count = await ollama_provider.count_tokens(long_text)

        # Should be roughly len(text) // 4
        assert count > 0
        assert count == len(long_text) // 4


class TestOllamaProviderCostCalculation:
    """Test OllamaProvider cost calculation."""

    def test_get_cost_always_zero(self, ollama_provider):
        """Test that cost is always $0 for local inference."""
        cost = ollama_provider.get_cost(1_000_000, 1_000_000)
        assert cost == 0.0

    def test_get_cost_any_tokens(self, ollama_provider):
        """Test cost is $0 regardless of token count."""
        assert ollama_provider.get_cost(0, 0) == 0.0
        assert ollama_provider.get_cost(100, 50) == 0.0
        assert ollama_provider.get_cost(10_000, 5_000) == 0.0


class TestOllamaProviderModelInfo:
    """Test OllamaProvider model information."""

    def test_get_model_info(self, ollama_provider):
        """Test getting model information."""
        info = ollama_provider.get_model_info()

        assert info["provider"] == "ollama"
        assert info["model"] == "llama3.2"
        assert info["context_window"] == 128000
        assert info["pricing"]["input"] == 0.0
        assert info["pricing"]["output"] == 0.0
        assert info["host"] == "http://localhost:11434"
        assert info["local"] is True

    def test_get_model_info_custom_model(self, ollama_provider):
        """Test getting model info for custom model."""
        ollama_provider.model = "mistral"
        info = ollama_provider.get_model_info()

        assert info["model"] == "mistral"
        assert info["context_window"] == 32000

    def test_get_model_info_unknown_model(self, ollama_provider):
        """Test getting model info for unknown model defaults to 32K context."""
        ollama_provider.model = "unknown-model"
        info = ollama_provider.get_model_info()

        assert info["model"] == "unknown-model"
        assert info["context_window"] == 32000  # Default

    def test_list_available_models(self):
        """Test listing available models."""
        models = OllamaProvider.list_available_models()

        assert len(models) >= 7
        assert "llama3.2" in models
        assert "mistral" in models
        assert "mixtral" in models
        assert "qwen3" in models
        assert "deepseek-r1" in models
        assert "gemma3" in models


class TestOllamaProviderAvailability:
    """Test OllamaProvider availability checking."""

    @pytest.mark.asyncio
    async def test_check_availability_success(self, ollama_provider):
        """Test availability check when server is running."""
        mock_list_response = {
            "models": [
                {"model": "llama3.2"},
                {"model": "mistral"},
            ]
        }
        ollama_provider.client.list = AsyncMock(return_value=mock_list_response)

        availability = await ollama_provider.check_availability()

        assert availability["server_running"] is True
        assert availability["model_available"] is True
        assert "llama3.2" in availability["available_models"]

    @pytest.mark.asyncio
    async def test_check_availability_model_not_available(self, ollama_provider):
        """Test availability check when model is not pulled."""
        mock_list_response = {
            "models": [
                {"model": "mistral"},
            ]
        }
        ollama_provider.client.list = AsyncMock(return_value=mock_list_response)

        availability = await ollama_provider.check_availability()

        assert availability["server_running"] is True
        assert availability["model_available"] is False

    @pytest.mark.asyncio
    async def test_check_availability_server_not_running(self, ollama_provider):
        """Test availability check when server is not running."""
        ollama_provider.client.list = AsyncMock(side_effect=ConnectionError("Refused"))

        availability = await ollama_provider.check_availability()

        assert availability["server_running"] is False
        assert availability["model_available"] is False
        assert "error" in availability


class TestOllamaProviderCleanup:
    """Test OllamaProvider cleanup."""

    @pytest.mark.asyncio
    async def test_aclose(self, ollama_provider):
        """Test closing the async client."""
        ollama_provider.client.aclose = AsyncMock()

        await ollama_provider.aclose()

        ollama_provider.client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_error_handling(self, ollama_provider):
        """Test aclose handles errors gracefully."""
        ollama_provider.client.aclose = AsyncMock(side_effect=Exception("Close failed"))

        # Should not raise exception
        await ollama_provider.aclose()
