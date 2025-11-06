"""
Quorum-MCP Server Implementation

This module implements the FastMCP server with two core tools:
- q_in: Submit a query to the quorum for consensus-based response
- q_out: Retrieve the consensus results from a quorum session

The server uses stdio transport for integration with Claude Desktop and other MCP clients.
"""

import asyncio
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from quorum_mcp.session import SessionManager, get_session_manager

# Initialize FastMCP server
mcp = FastMCP("Quorum-MCP")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances (initialized in main)
_session_manager: SessionManager | None = None
_orchestrator: Orchestrator | None = None


@mcp.tool()
async def q_in(
    query: str,
    context: str | None = None,
    mode: str = "quick_consensus",
) -> dict[str, Any]:
    """
    Submit a query to the quorum for consensus-based response.

    This tool initiates a multi-round deliberation process across multiple AI providers
    (Claude, GPT-4, etc.) to generate a consensus-based response.

    Args:
        query: The user's question or prompt to be processed by the quorum
        context: Optional additional context or constraints for the query
        mode: Operational mode - "quick_consensus", "full_deliberation", or "devils_advocate"

    Returns:
        A dictionary containing:
        - session_id: Unique identifier for this quorum session
        - status: Current status of the deliberation (completed, failed, etc.)
        - message: Human-readable status message
        - consensus: The consensus result (if completed)
        - confidence: Confidence score (0.0-1.0)
        - cost: Total cost in USD

    Example:
        >>> result = await q_in(
        ...     query="What are the best practices for API design?",
        ...     context="Focus on REST APIs and modern standards",
        ...     mode="full_deliberation"
        ... )
        >>> print(result["consensus"]["summary"])
        "Based on consensus across multiple AI models..."
    """
    global _orchestrator, _session_manager

    logger.info(f"q_in called with query: {query[:100]}... (mode: {mode})")

    if _orchestrator is None or _session_manager is None:
        return {
            "error": "Server not initialized. Please check API keys are configured.",
            "status": "error",
        }

    try:
        # Execute quorum consensus
        session = await _orchestrator.execute_quorum(query=query, context=context or "", mode=mode)

        # Return results
        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "message": f"Quorum consensus completed using {mode} mode",
            "consensus": session.consensus,
            "confidence": session.consensus.get("confidence", 0.0) if session.consensus else 0.0,
            "cost": session.metadata.get("total_cost", 0.0),
            "providers_used": session.metadata.get("providers_used", []),
        }

    except Exception as e:
        logger.error(f"Error in q_in: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed",
            "message": f"Quorum consensus failed: {e}",
        }


@mcp.tool()
async def q_out(session_id: str) -> dict[str, Any]:
    """
    Retrieve the consensus results from a quorum session.

    This tool fetches the results of a deliberation session by session ID.

    Args:
        session_id: The unique session identifier

    Returns:
        A dictionary containing:
        - session_id: The session identifier
        - status: Session status (completed, failed, etc.)
        - consensus: The final consensus result (if completed)
        - confidence: Confidence score (0.0-1.0) of the consensus
        - metadata: Additional information (rounds, tokens, cost, etc.)

    Example:
        >>> result = await q_out(session_id="abc-123")
        >>> print(result["consensus"]["summary"])
        "Based on consensus across multiple AI models..."
    """
    global _session_manager

    logger.info(f"q_out called for session: {session_id}")

    if _session_manager is None:
        return {"error": "Server not initialized", "status": "error"}

    try:
        # Retrieve session
        session = await _session_manager.get_session(session_id)

        if session is None:
            return {
                "error": f"Session {session_id} not found or expired",
                "status": "not_found",
            }

        # Return session data
        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "query": session.query,
            "mode": session.mode,
            "consensus": session.consensus,
            "confidence": session.consensus.get("confidence", 0.0) if session.consensus else 0.0,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
        }

    except Exception as e:
        logger.error(f"Error in q_out: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "error",
            "message": f"Failed to retrieve session: {e}",
        }


async def initialize_server() -> None:
    """Initialize server components (providers, session manager, orchestrator)."""
    global _session_manager, _orchestrator

    logger.info("Initializing Quorum-MCP server components...")

    # Initialize session manager
    _session_manager = get_session_manager()
    await _session_manager.start()
    logger.info("Session manager initialized")

    # Initialize providers
    providers = []

    # Try to initialize Anthropic provider
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude = AnthropicProvider()
            providers.append(claude)
            logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")

    # Try to initialize OpenAI provider
    if os.getenv("OPENAI_API_KEY"):
        try:
            gpt4 = OpenAIProvider()
            providers.append(gpt4)
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")

    # Try to initialize Gemini provider
    if os.getenv("GOOGLE_API_KEY"):
        try:
            gemini = GeminiProvider()
            providers.append(gemini)
            logger.info("Gemini provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini provider: {e}")

    if len(providers) == 0:
        logger.error(
            "No providers initialized. Please set at least one API key: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY"
        )
        return

    # Initialize orchestrator
    _orchestrator = Orchestrator(providers=providers, session_manager=_session_manager)
    logger.info(f"Orchestrator initialized with {len(providers)} provider(s)")


def main() -> None:
    """
    Main entry point for the Quorum-MCP server.

    This function initializes and runs the FastMCP server using stdio transport,
    making it compatible with Claude Desktop and other MCP clients.

    The server will:
    1. Initialize AI provider connections (Anthropic, OpenAI, Gemini)
    2. Initialize session management
    3. Start the MCP server on stdio transport
    4. Handle incoming tool requests (q_in, q_out)

    Environment Variables Required (at least one):
    - ANTHROPIC_API_KEY: API key for Claude
    - OPENAI_API_KEY: API key for GPT-4
    - GOOGLE_API_KEY: API key for Gemini
    """
    logger.info("Starting Quorum-MCP server...")

    try:
        # Initialize server components
        asyncio.run(initialize_server())

        # Run the FastMCP server
        # FastMCP automatically handles stdio transport by default
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
