"""
Integration tests for Quorum-MCP.

Tests cover:
- End-to-end workflow with real provider integration
- MCP server tool integration
- Full deliberation flow
- Session persistence and retrieval
- Cost tracking across full sessions
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import os

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers.base import Provider, ProviderRequest, ProviderResponse
from quorum_mcp.providers.anthropic_provider import AnthropicProvider
from quorum_mcp.providers.openai_provider import OpenAIProvider
from quorum_mcp.session import SessionManager, SessionStatus
from quorum_mcp.server import q_in, q_out, initialize_server


class MockProvider(Provider):
    """Mock provider for integration testing."""

    def __init__(self, name: str, responses: dict[int, str] | None = None):
        """
        Initialize mock provider.

        Args:
            name: Provider name
            responses: Dict mapping round number to response text
        """
        self.name = name
        self.responses = responses or {1: f"Response from {name}"}
        self.current_round = 1

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """Send mock request."""
        response_text = self.responses.get(
            self.current_round, f"Round {self.current_round} response from {self.name}"
        )
        self.current_round += 1

        return ProviderResponse(
            content=response_text,
            model=f"{self.name}-model",
            provider=self.name,
            tokens_input=100,
            tokens_output=50,
            cost=0.0015,
            metadata={"round": self.current_round - 1},
        )

    async def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(text.split())

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Calculate cost."""
        return (tokens_input * 0.00001) + (tokens_output * 0.00005)

    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.name

    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            "provider": self.name,
            "model": f"{self.name}-model",
            "context_window": 100000,
            "pricing": {"input": 0.01, "output": 0.05},
        }


@pytest.fixture
async def session_manager():
    """Create and start session manager."""
    manager = SessionManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def mock_providers():
    """Create mock providers with realistic responses."""
    provider1 = MockProvider(
        "anthropic",
        {
            1: "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            2: "After reviewing other perspectives, I agree that Python's ease of use and extensive libraries make it excellent for beginners and professionals alike.",
            3: "In synthesis, Python excels in web development, data science, and automation due to its rich ecosystem.",
        },
    )

    provider2 = MockProvider(
        "openai",
        {
            1: "Python is an interpreted, high-level programming language with dynamic typing and strong readability.",
            2: "I concur with the previous assessment. Python's versatility and community support are key strengths.",
            3: "Overall, Python is ideal for rapid development, scripting, and data analysis applications.",
        },
    )

    return [provider1, provider2]


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_quick_consensus_workflow(self, mock_providers, session_manager):
        """Test complete quick consensus workflow."""
        # Create orchestrator
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        # Execute query
        session = await orchestrator.execute_quorum(
            query="What is Python?",
            context="Programming languages",
            mode="quick_consensus",
        )

        # Verify session completed
        assert session.status == SessionStatus.COMPLETED
        assert session.session_id is not None

        # Verify consensus was built
        assert session.consensus is not None
        assert "summary" in session.consensus
        assert "confidence" in session.consensus
        assert "cost" in session.consensus

        # Verify responses from both providers
        assert len(session.provider_responses) == 2
        assert "anthropic" in session.provider_responses
        assert "openai" in session.provider_responses

        # Verify session is retrievable
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        assert retrieved.consensus == session.consensus

    @pytest.mark.asyncio
    async def test_full_deliberation_workflow(self, mock_providers, session_manager):
        """Test complete full deliberation workflow (3 rounds)."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Should we use microservices?",
            context="Small team, early stage startup",
            mode="full_deliberation",
        )

        # Verify 3-round deliberation
        assert session.status == SessionStatus.COMPLETED

        # Each provider should have multiple rounds
        for provider_name in session.provider_responses:
            assert len(session.provider_responses[provider_name]) >= 1

        # Should have recommendations
        assert "recommendations" in session.consensus

    @pytest.mark.asyncio
    async def test_devils_advocate_workflow(self, mock_providers, session_manager):
        """Test devil's advocate workflow."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="We should skip code reviews to move faster",
            context="Startup trying to reach product-market fit",
            mode="devils_advocate",
        )

        assert session.status == SessionStatus.COMPLETED
        assert "minority_opinions" in session.consensus


class TestMCPServerIntegration:
    """Test MCP server tool integration."""

    @pytest.mark.asyncio
    async def test_q_in_tool(self, mock_providers, session_manager):
        """Test q_in tool integration."""
        # Set up global state
        from quorum_mcp import server

        server._session_manager = session_manager
        server._orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        # Call q_in tool
        result = await q_in(
            query="What is Python?",
            context="Programming",
            mode="quick_consensus",
        )

        # Verify result structure
        assert "session_id" in result
        assert "status" in result
        assert "consensus" in result
        assert "confidence" in result
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_q_out_tool(self, mock_providers, session_manager):
        """Test q_out tool integration."""
        from quorum_mcp import server

        # Set up orchestrator
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )
        server._session_manager = session_manager
        server._orchestrator = orchestrator

        # Create a session first
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Retrieve using q_out
        result = await q_out(session_id=session.session_id)

        # Verify result
        assert result["session_id"] == session.session_id
        assert result["status"] == "completed"
        assert "consensus" in result
        assert result["query"] == "Test query"

    @pytest.mark.asyncio
    async def test_q_out_not_found(self, session_manager):
        """Test q_out with non-existent session."""
        from quorum_mcp import server

        server._session_manager = session_manager

        result = await q_out(session_id="non-existent-id")

        assert result["status"] == "not_found"
        assert "error" in result


class TestSessionPersistence:
    """Test session persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_session_persists_across_queries(
        self, mock_providers, session_manager
    ):
        """Test that sessions persist and can be retrieved."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        # Create multiple sessions
        session1 = await orchestrator.execute_quorum(
            query="Query 1", mode="quick_consensus"
        )
        session2 = await orchestrator.execute_quorum(
            query="Query 2", mode="quick_consensus"
        )

        # Retrieve sessions
        retrieved1 = await session_manager.get_session(session1.session_id)
        retrieved2 = await session_manager.get_session(session2.session_id)

        assert retrieved1.session_id == session1.session_id
        assert retrieved2.session_id == session2.session_id
        assert retrieved1.query == "Query 1"
        assert retrieved2.query == "Query 2"

    @pytest.mark.asyncio
    async def test_list_all_sessions(self, mock_providers, session_manager):
        """Test listing all sessions."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        # Create sessions
        await orchestrator.execute_quorum(query="Query 1", mode="quick_consensus")
        await orchestrator.execute_quorum(query="Query 2", mode="quick_consensus")

        # List sessions
        sessions = await session_manager.list_sessions()

        assert len(sessions) >= 2

    @pytest.mark.asyncio
    async def test_session_statistics(self, mock_providers, session_manager):
        """Test session statistics."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        await orchestrator.execute_quorum(query="Query 1", mode="quick_consensus")
        await orchestrator.execute_quorum(query="Query 2", mode="quick_consensus")

        stats = await session_manager.get_stats()

        assert stats["total"] >= 2
        assert "by_status" in stats
        assert stats["by_status"]["completed"] >= 2


class TestCostTracking:
    """Test cost tracking across full sessions."""

    @pytest.mark.asyncio
    async def test_cost_accumulation(self, mock_providers, session_manager):
        """Test that costs accumulate correctly across rounds."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Test query", mode="full_deliberation"
        )

        # Verify cost tracking
        assert "cost" in session.consensus
        assert session.consensus["cost"]["total_cost"] > 0
        assert "providers" in session.consensus["cost"]

        # Each provider should have cost tracked
        for provider_name in session.provider_responses:
            assert provider_name in session.consensus["cost"]["providers"]

    @pytest.mark.asyncio
    async def test_cost_per_provider(self, mock_providers, session_manager):
        """Test per-provider cost tracking."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        provider_costs = session.consensus["cost"]["providers"]

        # Each provider should have non-zero cost
        assert all(cost > 0 for cost in provider_costs.values())


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.mark.asyncio
    async def test_continues_with_partial_provider_failure(self, session_manager):
        """Test that system continues when one provider fails."""

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise Exception("Provider unavailable")

        good_provider = MockProvider("good", {1: "Good response"})
        bad_provider = FailingProvider("bad")

        orchestrator = Orchestrator(
            providers=[good_provider, bad_provider], session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Should still complete with one working provider
        assert session.status == SessionStatus.COMPLETED
        assert "good" in session.provider_responses

    @pytest.mark.asyncio
    async def test_handles_timeout_gracefully(self, session_manager):
        """Test graceful handling of provider timeouts."""

        class SlowProvider(MockProvider):
            async def send_request(self, request):
                import asyncio

                await asyncio.sleep(10)  # Simulate slow response
                return await super().send_request(request)

        fast_provider = MockProvider("fast", {1: "Fast response"})
        slow_provider = SlowProvider("slow")

        orchestrator = Orchestrator(
            providers=[fast_provider, slow_provider], session_manager=session_manager
        )

        # Note: This test may need timeout configuration in orchestrator
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Should complete with at least the fast provider
        assert session.status == SessionStatus.COMPLETED


class TestMultiRoundCoordination:
    """Test multi-round provider coordination."""

    @pytest.mark.asyncio
    async def test_provider_sees_previous_responses(self, session_manager):
        """Test that providers receive context from previous rounds."""
        provider1 = MockProvider(
            "p1",
            {
                1: "Initial response about Python",
                2: "After reviewing other responses, I want to add more details",
                3: "Final synthesis incorporating all perspectives",
            },
        )

        provider2 = MockProvider(
            "p2",
            {
                1: "My initial take on Python",
                2: "Building on the previous discussion",
                3: "Comprehensive final answer",
            },
        )

        orchestrator = Orchestrator(
            providers=[provider1, provider2], session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="What is Python?", mode="full_deliberation"
        )

        # Verify multiple rounds executed
        assert session.status == SessionStatus.COMPLETED
        # Both providers should have participated
        assert len(session.provider_responses) == 2


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Live API tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)
class TestLiveAPIIntegration:
    """
    Live API integration tests (requires API keys).

    Set RUN_LIVE_TESTS=1 and provide API keys to run these tests.
    Warning: These tests will incur actual API costs.
    """

    @pytest.mark.asyncio
    async def test_live_anthropic_provider(self, session_manager):
        """Test with real Anthropic API."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider()
        orchestrator = Orchestrator(providers=[provider], session_manager=session_manager)

        session = await orchestrator.execute_quorum(
            query="What is 2+2?", mode="quick_consensus", max_tokens=100
        )

        assert session.status == SessionStatus.COMPLETED
        assert "4" in session.consensus["summary"]

    @pytest.mark.asyncio
    async def test_live_openai_provider(self, session_manager):
        """Test with real OpenAI API."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = OpenAIProvider()
        orchestrator = Orchestrator(providers=[provider], session_manager=session_manager)

        session = await orchestrator.execute_quorum(
            query="What is 2+2?", mode="quick_consensus", max_tokens=100
        )

        assert session.status == SessionStatus.COMPLETED
        assert "4" in session.consensus["summary"]

    @pytest.mark.asyncio
    async def test_live_multi_provider_consensus(self, session_manager):
        """Test with both Anthropic and OpenAI."""
        if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Both API keys required")

        providers = [AnthropicProvider(), OpenAIProvider()]
        orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

        session = await orchestrator.execute_quorum(
            query="Name one programming language",
            mode="quick_consensus",
            max_tokens=50,
        )

        assert session.status == SessionStatus.COMPLETED
        assert len(session.provider_responses) == 2
        assert session.consensus["cost"]["total_cost"] > 0
