"""
Quorum-MCP: Multi-AI Consensus System MCP Server

This package implements an MCP server that orchestrates multiple AI providers
(Claude, GPT-4, etc.) to create consensus-based responses through multi-round
deliberation and synthesis.

Core Components:
- server: FastMCP server implementation with q_in and q_out tools
- session: Session state management for tracking quorum consultations
- providers: Abstraction layer for AI provider APIs
- orchestration: Multi-round deliberation engine
- synthesis: Consensus aggregation and response generation
- config: Configuration management
"""

__version__ = "0.1.0"
__author__ = "Quorum-MCP Contributors"

# Export session management components
from .session import (
    Session,
    SessionManager,
    SessionStatus,
    get_session_manager,
)

__all__ = [
    "__version__",
    "__author__",
    "Session",
    "SessionStatus",
    "SessionManager",
    "get_session_manager",
]
