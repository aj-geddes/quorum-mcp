"""
Unit tests for session management module
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from quorum_mcp.session import Session, SessionManager, SessionStatus, get_session_manager


class TestSession:
    """Tests for Session model"""

    def test_session_creation_defaults(self):
        """Test session creation with default values"""
        session = Session(query="Test query")

        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.status == SessionStatus.PENDING
        assert session.query == "Test query"
        assert session.mode == "full_deliberation"
        assert session.provider_responses == {}
        assert session.consensus is None
        assert session.metadata == {}
        assert session.error is None

    def test_session_creation_with_custom_values(self):
        """Test session creation with custom values"""
        session = Session(query="Custom query", mode="quick_consensus")

        assert session.query == "Custom query"
        assert session.mode == "quick_consensus"

    def test_update_timestamp(self):
        """Test timestamp update"""
        session = Session(query="Test")
        original_time = session.updated_at

        # Wait a tiny bit to ensure time difference
        import time

        time.sleep(0.01)

        session.update_timestamp()
        assert session.updated_at > original_time

    def test_add_provider_response(self):
        """Test adding provider responses"""
        session = Session(query="Test")

        session.add_provider_response("claude", 1, {"text": "Response 1"})
        assert "claude" in session.provider_responses
        assert 1 in session.provider_responses["claude"]
        assert session.provider_responses["claude"][1] == {"text": "Response 1"}

        session.add_provider_response("claude", 2, {"text": "Response 2"})
        assert 2 in session.provider_responses["claude"]

        session.add_provider_response("gpt4", 1, {"text": "GPT Response"})
        assert "gpt4" in session.provider_responses

    def test_set_consensus(self):
        """Test setting consensus result"""
        session = Session(query="Test")
        consensus_data = {"summary": "Consensus reached", "confidence": 0.95}

        session.set_consensus(consensus_data)

        assert session.consensus == consensus_data
        assert session.status == SessionStatus.COMPLETED

    def test_mark_failed(self):
        """Test marking session as failed"""
        session = Session(query="Test")

        session.mark_failed("Provider timeout")

        assert session.status == SessionStatus.FAILED
        assert session.error == "Provider timeout"

    def test_is_expired(self):
        """Test session expiration check"""
        # Create session with mocked old timestamp
        with patch("quorum_mcp.session.datetime") as mock_datetime:
            old_time = datetime.utcnow() - timedelta(hours=25)
            mock_datetime.utcnow.return_value = old_time

            session = Session(query="Test")

        # Check expiration (should be expired with 24h TTL)
        assert session.is_expired(ttl_hours=24) is True
        assert session.is_expired(ttl_hours=48) is False


@pytest.mark.asyncio
class TestSessionManager:
    """Tests for SessionManager class"""

    async def test_create_session(self):
        """Test session creation"""
        manager = SessionManager()

        session = await manager.create_session(query="Test query", mode="full_deliberation")

        assert session.query == "Test query"
        assert session.mode == "full_deliberation"
        assert session.status == SessionStatus.PENDING

    async def test_get_session(self):
        """Test retrieving session by ID"""
        manager = SessionManager()

        # Create session
        created_session = await manager.create_session("Test query")
        session_id = created_session.session_id

        # Retrieve session
        retrieved_session = await manager.get_session(session_id)

        assert retrieved_session.session_id == session_id
        assert retrieved_session.query == "Test query"

    async def test_get_nonexistent_session(self):
        """Test retrieving non-existent session raises error"""
        manager = SessionManager()

        with pytest.raises(KeyError, match="Session not found"):
            await manager.get_session("nonexistent-id")

    async def test_update_session(self):
        """Test updating session data"""
        manager = SessionManager()

        # Create session
        session = await manager.create_session("Test query")
        session_id = session.session_id

        # Update session
        updates = {"status": SessionStatus.IN_PROGRESS, "metadata": {"round": 1}}
        updated_session = await manager.update_session(session_id, updates)

        assert updated_session.status == SessionStatus.IN_PROGRESS
        assert updated_session.metadata == {"round": 1}

    async def test_update_invalid_field(self):
        """Test updating invalid field raises error"""
        manager = SessionManager()

        session = await manager.create_session("Test query")
        session_id = session.session_id

        with pytest.raises(ValueError, match="Invalid session field"):
            await manager.update_session(session_id, {"invalid_field": "value"})

    async def test_list_sessions(self):
        """Test listing all sessions"""
        manager = SessionManager()

        # Create multiple sessions
        _session1 = await manager.create_session("Query 1")
        session2 = await manager.create_session("Query 2")
        _session3 = await manager.create_session("Query 3")

        # Update one to completed status
        await manager.update_session(session2.session_id, {"status": SessionStatus.COMPLETED})

        # List all sessions
        all_sessions = await manager.list_sessions()
        assert len(all_sessions) == 3

        # List only completed sessions
        completed_sessions = await manager.list_sessions(status=SessionStatus.COMPLETED)
        assert len(completed_sessions) == 1
        assert completed_sessions[0].session_id == session2.session_id

    async def test_delete_session(self):
        """Test deleting a session"""
        manager = SessionManager()

        session = await manager.create_session("Test query")
        session_id = session.session_id

        # Delete session
        await manager.delete_session(session_id)

        # Verify deletion
        with pytest.raises(KeyError, match="Session not found"):
            await manager.get_session(session_id)

    async def test_get_stats(self):
        """Test getting session statistics"""
        manager = SessionManager()

        # Create sessions with different statuses
        _session1 = await manager.create_session("Query 1")
        session2 = await manager.create_session("Query 2")
        session3 = await manager.create_session("Query 3")

        await manager.update_session(session2.session_id, {"status": SessionStatus.COMPLETED})
        await manager.update_session(session3.session_id, {"status": SessionStatus.FAILED})

        # Get stats
        stats = await manager.get_stats()

        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 3
        assert stats["expired_sessions"] == 0
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1

    async def test_cleanup_expired_sessions(self):
        """Test automatic cleanup of expired sessions"""
        manager = SessionManager(ttl_hours=1)

        # Create session with old timestamp
        with patch("quorum_mcp.session.datetime") as mock_datetime:
            old_time = datetime.utcnow() - timedelta(hours=2)
            mock_datetime.utcnow.return_value = old_time

            old_session = await manager.create_session("Old query")
            old_session_id = old_session.session_id

        # Create fresh session
        new_session = await manager.create_session("New query")
        new_session_id = new_session.session_id

        # Run cleanup
        await manager._cleanup_expired_sessions()

        # Old session should be removed
        with pytest.raises(KeyError):
            await manager.get_session(old_session_id)

        # New session should remain
        retrieved = await manager.get_session(new_session_id)
        assert retrieved.session_id == new_session_id

    async def test_thread_safety_concurrent_operations(self):
        """Test thread safety with concurrent operations"""
        manager = SessionManager()

        async def create_and_update():
            session = await manager.create_session("Concurrent query")
            await manager.update_session(session.session_id, {"status": SessionStatus.COMPLETED})
            return session.session_id

        # Run multiple concurrent operations
        session_ids = await asyncio.gather(*[create_and_update() for _ in range(10)])

        # Verify all sessions were created
        assert len(session_ids) == 10
        assert len(set(session_ids)) == 10  # All unique

        # Verify all can be retrieved
        for session_id in session_ids:
            session = await manager.get_session(session_id)
            assert session.status == SessionStatus.COMPLETED

    async def test_background_cleanup_lifecycle(self):
        """Test starting and stopping background cleanup task"""
        manager = SessionManager(cleanup_interval=1)

        # Start cleanup task
        await manager.start()
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()

        # Stop cleanup task
        await manager.stop()
        assert manager._cleanup_task is not None
        assert manager._cleanup_task.done()


def test_get_session_manager_singleton():
    """Test singleton pattern for session manager"""
    manager1 = get_session_manager()
    manager2 = get_session_manager()

    assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
