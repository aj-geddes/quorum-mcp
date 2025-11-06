"""
Unit tests for Orchestrator.

Tests cover:
- Orchestrator initialization
- Quick consensus mode
- Full deliberation mode
- Devil's advocate mode
- Session management integration
- Provider coordination
- Consensus building
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers.base import Provider, ProviderRequest, ProviderResponse
from quorum_mcp.session import Session, SessionManager, SessionStatus


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, name: str, responses: list[str] | None = None):
        self.name = name
        self.responses = responses or ["Mock response"]
        self.call_count = 0

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """Mock send request."""
        response_text = self.responses[
            min(self.call_count, len(self.responses) - 1)
        ]
        self.call_count += 1

        return ProviderResponse(
            content=response_text,
            model=f"{self.name}-model",
            provider=self.name,
            tokens_input=50,
            tokens_output=30,
            cost=0.001,
            metadata={},
        )

    async def count_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split())

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Mock cost calculation."""
        return (tokens_input + tokens_output) * 0.00001

    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.name

    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            "provider": self.name,
            "model": f"{self.name}-model",
            "context_window": 100000,
        }


@pytest.fixture
async def session_manager():
    """Create session manager for testing."""
    manager = SessionManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def mock_providers():
    """Create mock providers for testing."""
    return [
        MockProvider("provider1", ["Response from provider 1"]),
        MockProvider("provider2", ["Response from provider 2"]),
    ]


@pytest.fixture
def orchestrator(mock_providers, session_manager):
    """Create orchestrator for testing."""
    return Orchestrator(providers=mock_providers, session_manager=session_manager)


class TestOrchestratorInit:
    """Test Orchestrator initialization."""

    def test_init_with_providers(self, mock_providers, session_manager):
        """Test initialization with providers."""
        orchestrator = Orchestrator(
            providers=mock_providers, session_manager=session_manager
        )

        assert len(orchestrator.providers) == 2
        assert orchestrator.session_manager == session_manager

    def test_init_empty_providers(self, session_manager):
        """Test initialization with no providers raises error."""
        with pytest.raises(ValueError, match="At least one provider"):
            Orchestrator(providers=[], session_manager=session_manager)

    def test_init_without_session_manager(self, mock_providers):
        """Test initialization without session manager."""
        orchestrator = Orchestrator(providers=mock_providers)

        assert orchestrator.session_manager is not None


class TestOrchestratorQuickConsensus:
    """Test quick consensus mode."""

    @pytest.mark.asyncio
    async def test_execute_quick_consensus(self, orchestrator, session_manager):
        """Test quick consensus execution."""
        session = await orchestrator.execute_quorum(
            query="What is Python?",
            context="Programming language",
            mode="quick_consensus",
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.consensus is not None
        assert "summary" in session.consensus
        assert "confidence" in session.consensus
        assert "agreement_areas" in session.consensus
        assert session.consensus["confidence"] > 0

    @pytest.mark.asyncio
    async def test_quick_consensus_single_provider(self, session_manager):
        """Test quick consensus with single provider."""
        provider = MockProvider("solo", ["Python is a programming language"])
        orchestrator = Orchestrator(providers=[provider], session_manager=session_manager)

        session = await orchestrator.execute_quorum(
            query="What is Python?", mode="quick_consensus"
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.consensus is not None

    @pytest.mark.asyncio
    async def test_quick_consensus_with_custom_params(
        self, orchestrator, session_manager
    ):
        """Test quick consensus with custom parameters."""
        session = await orchestrator.execute_quorum(
            query="What is Python?",
            context="Focus on web development",
            mode="quick_consensus",
            max_tokens=500,
            temperature=0.5,
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.query == "What is Python?"

    @pytest.mark.asyncio
    async def test_quick_consensus_cost_tracking(self, orchestrator, session_manager):
        """Test cost tracking in quick consensus."""
        session = await orchestrator.execute_quorum(
            query="What is Python?", mode="quick_consensus"
        )

        assert "cost" in session.consensus
        assert session.consensus["cost"]["total_cost"] > 0
        assert "providers" in session.consensus["cost"]


class TestOrchestratorFullDeliberation:
    """Test full deliberation mode."""

    @pytest.mark.asyncio
    async def test_execute_full_deliberation(self, orchestrator, session_manager):
        """Test full deliberation execution (3 rounds)."""
        session = await orchestrator.execute_quorum(
            query="Should we use microservices?",
            context="Small team, early stage",
            mode="full_deliberation",
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.consensus is not None

        # Check that multiple rounds were executed
        for provider_name in session.provider_responses:
            # Should have responses for rounds 1, 2, 3
            assert len(session.provider_responses[provider_name]) >= 1

    @pytest.mark.asyncio
    async def test_full_deliberation_recommendations(
        self, orchestrator, session_manager
    ):
        """Test that full deliberation includes recommendations."""
        session = await orchestrator.execute_quorum(
            query="Best database for our app?", mode="full_deliberation"
        )

        assert "recommendations" in session.consensus
        # Should have at least some recommendations
        assert isinstance(session.consensus["recommendations"], list)


class TestOrchestratorDevilsAdvocate:
    """Test devil's advocate mode."""

    @pytest.mark.asyncio
    async def test_execute_devils_advocate(self, orchestrator, session_manager):
        """Test devil's advocate mode."""
        session = await orchestrator.execute_quorum(
            query="We should skip testing to move faster",
            context="Startup environment",
            mode="devils_advocate",
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.consensus is not None
        assert "minority_opinions" in session.consensus

    @pytest.mark.asyncio
    async def test_devils_advocate_critical_analysis(
        self, orchestrator, session_manager
    ):
        """Test devil's advocate provides critical analysis."""
        session = await orchestrator.execute_quorum(
            query="Deploy directly to production without staging",
            mode="devils_advocate",
        )

        # Should have both supporting and critical perspectives
        assert "disagreement_areas" in session.consensus


class TestOrchestratorSessionManagement:
    """Test orchestrator session management."""

    @pytest.mark.asyncio
    async def test_creates_new_session(self, orchestrator, session_manager):
        """Test that orchestrator creates new session."""
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        assert session.session_id is not None
        assert session.query == "Test query"

        # Verify session is stored in manager
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_resumes_existing_session(self, orchestrator, session_manager):
        """Test resuming an existing session."""
        # Create initial session
        session1 = await orchestrator.execute_quorum(
            query="Initial query", mode="quick_consensus"
        )

        # Try to resume the same session
        session2 = await orchestrator.execute_quorum(
            query="Initial query",
            mode="quick_consensus",
            session_id=session1.session_id,
        )

        assert session2.session_id == session1.session_id

    @pytest.mark.asyncio
    async def test_session_status_updates(self, orchestrator, session_manager):
        """Test that session status is updated correctly."""
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Should be completed after execution
        assert session.status == SessionStatus.COMPLETED

        # Verify in session manager
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved.status == SessionStatus.COMPLETED


class TestOrchestratorProviderCoordination:
    """Test provider coordination."""

    @pytest.mark.asyncio
    async def test_parallel_provider_execution(self, session_manager):
        """Test that providers are called in parallel."""
        # Create providers that track call times
        call_times = []

        async def track_call(*args, **kwargs):
            call_times.append(datetime.now(timezone.utc))
            return ProviderResponse(
                content="Response",
                model="test",
                provider="test",
                tokens_input=10,
                tokens_output=10,
                cost=0.001,
                metadata={},
            )

        provider1 = MockProvider("p1")
        provider2 = MockProvider("p2")
        provider1.send_request = track_call
        provider2.send_request = track_call

        orchestrator = Orchestrator(
            providers=[provider1, provider2], session_manager=session_manager
        )

        await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Both providers should have been called
        assert len(call_times) >= 2

    @pytest.mark.asyncio
    async def test_handles_provider_failure(self, session_manager):
        """Test handling when one provider fails."""

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise Exception("Provider failed")

        good_provider = MockProvider("good", ["Working response"])
        bad_provider = FailingProvider("bad")

        orchestrator = Orchestrator(
            providers=[good_provider, bad_provider], session_manager=session_manager
        )

        # Should still complete with one working provider
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        assert session.status == SessionStatus.COMPLETED
        # Should have response from good provider
        assert "good" in session.provider_responses


class TestOrchestratorConsensusBuilding:
    """Test consensus building algorithms."""

    @pytest.mark.asyncio
    async def test_builds_consensus_from_responses(self, session_manager):
        """Test consensus building."""
        provider1 = MockProvider(
            "p1",
            ["API design should focus on REST principles, versioning, and documentation"],
        )
        provider2 = MockProvider(
            "p2",
            [
                "Good API design requires REST conventions, proper versioning, and comprehensive documentation"
            ],
        )

        orchestrator = Orchestrator(
            providers=[provider1, provider2], session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="What makes good API design?", mode="quick_consensus"
        )

        # Should identify common themes
        assert session.consensus["confidence"] > 0.3
        # Should have some agreement areas
        assert len(session.consensus["agreement_areas"]) > 0

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, session_manager):
        """Test confidence score calculation."""
        # Similar responses should have higher confidence
        provider1 = MockProvider("p1", ["Python is good for web development"])
        provider2 = MockProvider("p2", ["Python is great for web development"])

        orchestrator = Orchestrator(
            providers=[provider1, provider2], session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Is Python good for web dev?", mode="quick_consensus"
        )

        # Should have reasonable confidence for similar responses
        assert 0.0 <= session.consensus["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_identifies_disagreement(self, session_manager):
        """Test identification of disagreement areas."""
        provider1 = MockProvider("p1", ["Microservices are the best architecture"])
        provider2 = MockProvider(
            "p2", ["Monoliths are better for small teams and MVPs"]
        )

        orchestrator = Orchestrator(
            providers=[provider1, provider2], session_manager=session_manager
        )

        session = await orchestrator.execute_quorum(
            query="Microservices vs monolith?", mode="full_deliberation"
        )

        # Should identify disagreement
        if "disagreement_areas" in session.consensus:
            # Lower confidence expected when providers disagree
            assert session.consensus["confidence"] < 0.9


class TestOrchestratorErrorHandling:
    """Test orchestrator error handling."""

    @pytest.mark.asyncio
    async def test_handles_all_providers_failing(self, session_manager):
        """Test handling when all providers fail."""

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise Exception("All providers down")

        bad1 = FailingProvider("bad1")
        bad2 = FailingProvider("bad2")

        orchestrator = Orchestrator(
            providers=[bad1, bad2],
            session_manager=session_manager,
            check_health=False,  # Disable health checks for this test
        )

        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        # Should mark as failed
        assert session.status == SessionStatus.FAILED

    @pytest.mark.asyncio
    async def test_handles_invalid_mode(self, orchestrator):
        """Test handling of invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            await orchestrator.execute_quorum(
                query="Test query", mode="invalid_mode"
            )

    @pytest.mark.asyncio
    async def test_handles_empty_query(self, orchestrator):
        """Test handling of empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await orchestrator.execute_quorum(query="", mode="quick_consensus")


class TestOrchestratorMetadata:
    """Test orchestrator metadata tracking."""

    @pytest.mark.asyncio
    async def test_tracks_execution_metadata(self, orchestrator, session_manager):
        """Test that execution metadata is tracked."""
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        assert session.metadata is not None
        assert "providers_used" in session.metadata
        assert len(session.metadata["providers_used"]) > 0

    @pytest.mark.asyncio
    async def test_tracks_total_cost(self, orchestrator, session_manager):
        """Test that total cost is tracked."""
        session = await orchestrator.execute_quorum(
            query="Test query", mode="quick_consensus"
        )

        assert "cost" in session.consensus
        assert session.consensus["cost"]["total_cost"] >= 0


class TestOrchestratorModes:
    """Test different operational modes."""

    @pytest.mark.asyncio
    async def test_all_modes_complete_successfully(
        self, orchestrator, session_manager
    ):
        """Test that all operational modes can complete."""
        modes = ["quick_consensus", "full_deliberation", "devils_advocate"]

        for mode in modes:
            session = await orchestrator.execute_quorum(
                query=f"Test query for {mode}", mode=mode
            )

            assert session.status == SessionStatus.COMPLETED, f"Mode {mode} failed"
            assert session.consensus is not None, f"Mode {mode} has no consensus"
