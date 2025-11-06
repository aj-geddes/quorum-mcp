"""
Session State Management for Quorum-MCP

This module provides session tracking for multi-round AI deliberation across
multiple provider consultations. Each session maintains its state through the
entire quorum consensus process.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session lifecycle states"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def _get_utcnow() -> datetime:
    """Get current UTC time. Separate function to enable mocking in tests."""
    return datetime.utcnow()


class Session(BaseModel):
    """
    Session model representing a single quorum consultation.

    Tracks the complete lifecycle from initial query through multiple rounds
    of deliberation to final consensus.
    """

    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique session identifier (UUID)"
    )
    created_at: datetime = Field(
        default_factory=_get_utcnow, description="Timestamp when session was created"
    )
    updated_at: datetime = Field(
        default_factory=_get_utcnow, description="Timestamp of last session update"
    )
    status: SessionStatus = Field(
        default=SessionStatus.PENDING, description="Current session status"
    )
    query: str = Field(..., description="Original user query submitted to quorum")
    mode: str = Field(
        default="full_deliberation",
        description="Operational mode (full_deliberation, quick_consensus, devils_advocate)",
    )
    provider_responses: dict[str, dict[int, Any]] = Field(
        default_factory=dict,
        description="Provider responses organized by provider name and round number",
    )
    consensus: dict[str, Any] | None = Field(
        default=None, description="Final consensus result after deliberation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (costs, timing, provider info, etc.)"
    )
    error: str | None = Field(default=None, description="Error message if session failed")

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()

    def add_provider_response(self, provider: str, round_num: int, response: Any) -> None:
        """
        Add a provider response for a specific round.

        Args:
            provider: Provider name (e.g., 'claude', 'gpt4')
            round_num: Round number (1-based)
            response: Provider response data
        """
        if provider not in self.provider_responses:
            self.provider_responses[provider] = {}
        self.provider_responses[provider][round_num] = response
        self.update_timestamp()

    def set_consensus(self, consensus_data: dict[str, Any]) -> None:
        """
        Set the final consensus result.

        Args:
            consensus_data: Consensus result data
        """
        self.consensus = consensus_data
        self.status = SessionStatus.COMPLETED
        self.update_timestamp()

    def mark_failed(self, error_message: str) -> None:
        """
        Mark session as failed with error message.

        Args:
            error_message: Description of the failure
        """
        self.status = SessionStatus.FAILED
        self.error = error_message
        self.update_timestamp()

    def is_expired(self, ttl_hours: int = 24) -> bool:
        """
        Check if session has expired based on TTL.

        Args:
            ttl_hours: Time-to-live in hours (default 24)

        Returns:
            True if session has exceeded TTL
        """
        expiration_time = self.created_at + timedelta(hours=ttl_hours)
        return datetime.utcnow() > expiration_time


class SessionManager:
    """
    Thread-safe session manager for tracking quorum consultations.

    Provides in-memory storage with async locks for concurrent access,
    automatic cleanup of expired sessions, and complete lifecycle management.
    """

    def __init__(self, ttl_hours: int = 24, cleanup_interval: int = 3600):
        """
        Initialize session manager.

        Args:
            ttl_hours: Session time-to-live in hours (default 24)
            cleanup_interval: Background cleanup interval in seconds (default 3600)
        """
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._ttl_hours = ttl_hours
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task | None = None

        logger.info(
            f"SessionManager initialized (TTL: {ttl_hours}h, " f"Cleanup: {cleanup_interval}s)"
        )

    async def start(self) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Background cleanup task started")

    async def stop(self) -> None:
        """Stop background cleanup task"""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Background cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Background task that periodically removes expired sessions"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    async def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from storage"""
        async with self._lock:
            expired_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_expired(self._ttl_hours)
            ]

            for session_id in expired_ids:
                del self._sessions[session_id]
                logger.info(f"Removed expired session: {session_id}")

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired session(s)")

    async def create_session(self, query: str, mode: str = "full_deliberation") -> Session:
        """
        Create a new session.

        Args:
            query: User query to submit to quorum
            mode: Operational mode (default: 'full_deliberation')

        Returns:
            Newly created Session object
        """
        session = Session(query=query, mode=mode)

        async with self._lock:
            self._sessions[session.session_id] = session

        logger.info(
            f"Created session {session.session_id} " f"(mode: {mode}, query length: {len(query)})"
        )

        return session

    async def get_session(self, session_id: str) -> Session:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object

        Raises:
            KeyError: If session ID not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Session not found: {session_id}")
                raise KeyError(f"Session not found: {session_id}")

            session = self._sessions[session_id]

        # Check if expired
        if session.is_expired(self._ttl_hours):
            logger.warning(f"Accessed expired session: {session_id}")
            raise KeyError(f"Session expired: {session_id}")

        return session

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> Session:
        """
        Update session with new data.

        Args:
            session_id: Session identifier
            updates: Dictionary of field updates

        Returns:
            Updated Session object

        Raises:
            KeyError: If session ID not found
            ValueError: If invalid field update attempted
        """
        async with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Cannot update - session not found: {session_id}")
                raise KeyError(f"Session not found: {session_id}")

            session = self._sessions[session_id]

            # Check if expired
            if session.is_expired(self._ttl_hours):
                logger.warning(f"Cannot update expired session: {session_id}")
                raise KeyError(f"Session expired: {session_id}")

            # Apply updates
            for field, value in updates.items():
                if not hasattr(session, field):
                    logger.error(f"Invalid field update attempted: {field}")
                    raise ValueError(f"Invalid session field: {field}")
                setattr(session, field, value)

            session.update_timestamp()

        logger.info(f"Updated session {session_id} " f"(fields: {', '.join(updates.keys())})")

        return session

    async def list_sessions(
        self, status: SessionStatus | None = None, include_expired: bool = False
    ) -> list[Session]:
        """
        List all sessions, optionally filtered by status.

        Args:
            status: Filter by session status (optional)
            include_expired: Include expired sessions (default: False)

        Returns:
            List of Session objects
        """
        async with self._lock:
            sessions = list(self._sessions.values())

        # Filter out expired sessions unless explicitly requested
        if not include_expired:
            sessions = [s for s in sessions if not s.is_expired(self._ttl_hours)]

        # Filter by status if specified
        if status is not None:
            sessions = [s for s in sessions if s.status == status]

        return sessions

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Raises:
            KeyError: If session ID not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                logger.warning(f"Cannot delete - session not found: {session_id}")
                raise KeyError(f"Session not found: {session_id}")

            del self._sessions[session_id]

        logger.info(f"Deleted session: {session_id}")

    async def get_stats(self) -> dict[str, Any]:
        """
        Get session manager statistics.

        Returns:
            Dictionary with stats (total, by status, expired count)
        """
        async with self._lock:
            sessions = list(self._sessions.values())

        total = len(sessions)
        expired = sum(1 for s in sessions if s.is_expired(self._ttl_hours))
        active = total - expired

        status_counts = {}
        for status in SessionStatus:
            count = sum(
                1 for s in sessions if s.status == status and not s.is_expired(self._ttl_hours)
            )
            status_counts[status.value] = count

        return {
            "total_sessions": total,
            "active_sessions": active,
            "expired_sessions": expired,
            "by_status": status_counts,
            "ttl_hours": self._ttl_hours,
        }


# Singleton instance for application-wide use
_session_manager_instance: SessionManager | None = None


def get_session_manager(ttl_hours: int = 24, cleanup_interval: int = 3600) -> SessionManager:
    """
    Get or create the singleton SessionManager instance.

    Args:
        ttl_hours: Session TTL in hours (default 24)
        cleanup_interval: Cleanup interval in seconds (default 3600)

    Returns:
        SessionManager instance
    """
    global _session_manager_instance

    if _session_manager_instance is None:
        _session_manager_instance = SessionManager(
            ttl_hours=ttl_hours, cleanup_interval=cleanup_interval
        )

    return _session_manager_instance
