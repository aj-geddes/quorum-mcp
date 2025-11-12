"""
Quorum-MCP Web Server

This module provides a FastAPI-based web interface for the Quorum-MCP system,
allowing users to interact with the consensus engine through a browser-based dashboard.

Features:
- Interactive query submission and monitoring
- Real-time consensus visualization
- Provider comparison tools
- Cost calculation and tracking
- Session management dashboard
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import (
    AnthropicProvider,
    CohereProvider,
    GeminiProvider,
    MistralProvider,
    NovitaProvider,
    OllamaProvider,
    OpenAIProvider,
)
from quorum_mcp.session import SessionManager, SessionStatus, get_session_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Quorum-MCP Web Dashboard",
    description="Interactive web interface for multi-AI consensus building",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
_session_manager: SessionManager | None = None
_orchestrator: Orchestrator | None = None
_active_websockets: list[WebSocket] = []

# Mount static files
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for submitting a consensus query."""

    query: str = Field(..., min_length=1, max_length=50000, description="User query")
    context: str | None = Field(None, max_length=100000, description="Optional context")
    mode: str = Field(
        "quick_consensus",
        description="Operational mode: quick_consensus, full_deliberation, or devils_advocate",
    )
    providers: list[str] | None = Field(
        None, description="Optional list of specific providers to use"
    )
    max_tokens: int | None = Field(None, ge=1, le=32000, description="Maximum tokens")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature")


class QueryResponse(BaseModel):
    """Response model for query submission."""

    session_id: str
    status: str
    message: str
    estimated_cost: float | None = None


class CostEstimate(BaseModel):
    """Request model for cost estimation."""

    query_length: int = Field(..., ge=0, description="Approximate query length in characters")
    queries_per_month: int = Field(..., ge=1, description="Number of queries per month")
    providers: list[str] = Field(..., description="List of provider names")
    mode: str = Field("quick_consensus", description="Operational mode")


class ProviderInfo(BaseModel):
    """Information about a provider."""

    name: str
    display_name: str
    status: str
    models: list[str]
    default_model: str
    pricing: dict[str, Any]
    features: list[str]
    available: bool


# Startup and Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global _session_manager, _orchestrator

    logger.info("Starting Quorum-MCP Web Server...")

    # Initialize session manager
    _session_manager = get_session_manager()
    await _session_manager.start()

    # Initialize providers based on available API keys
    providers = []

    # Check for Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            providers.append(AnthropicProvider())
            logger.info("✓ Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"✗ Anthropic provider failed: {e}")

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            providers.append(OpenAIProvider())
            logger.info("✓ OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"✗ OpenAI provider failed: {e}")

    # Check for Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            providers.append(GeminiProvider())
            logger.info("✓ Gemini provider initialized")
        except Exception as e:
            logger.warning(f"✗ Gemini provider failed: {e}")

    # Check for Cohere
    if os.getenv("COHERE_API_KEY"):
        try:
            providers.append(CohereProvider())
            logger.info("✓ Cohere provider initialized")
        except Exception as e:
            logger.warning(f"✗ Cohere provider failed: {e}")

    # Check for Mistral
    if os.getenv("MISTRAL_API_KEY"):
        try:
            providers.append(MistralProvider())
            logger.info("✓ Mistral provider initialized")
        except Exception as e:
            logger.warning(f"✗ Mistral provider failed: {e}")

    # Check for Novita
    if os.getenv("NOVITA_API_KEY"):
        try:
            providers.append(NovitaProvider())
            logger.info("✓ Novita provider initialized")
        except Exception as e:
            logger.warning(f"✗ Novita provider failed: {e}")

    # Always try Ollama (local, no API key needed)
    try:
        ollama = OllamaProvider()
        # Test if Ollama is actually running
        if await ollama.check_availability():
            providers.append(ollama)
            logger.info("✓ Ollama provider initialized (local)")
        else:
            logger.warning("✗ Ollama server not running (start with: ollama serve)")
    except Exception as e:
        logger.warning(f"✗ Ollama provider failed: {e}")

    if not providers:
        logger.error(
            "No providers available! Please set at least one API key or run Ollama."
        )
    else:
        logger.info(f"Initialized {len(providers)} provider(s)")

    # Initialize orchestrator
    _orchestrator = Orchestrator(
        providers=providers,
        session_manager=_session_manager,
    )

    logger.info("Quorum-MCP Web Server started successfully!")
    logger.info(f"Dashboard: http://localhost:8000")
    logger.info(f"API Docs: http://localhost:8000/api/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _session_manager

    logger.info("Shutting down Quorum-MCP Web Server...")

    if _session_manager:
        await _session_manager.stop()

    # Close all WebSocket connections
    for ws in _active_websockets:
        try:
            await ws.close()
        except Exception:
            pass

    logger.info("Shutdown complete")


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML."""
    dashboard_file = STATIC_DIR / "dashboard.html"
    if dashboard_file.exists():
        return HTMLResponse(content=dashboard_file.read_text())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quorum-MCP</title>
        </head>
        <body>
            <h1>Quorum-MCP Dashboard</h1>
            <p>Dashboard files not found. Please ensure static files are properly installed.</p>
            <p><a href="/api/docs">View API Documentation</a></p>
        </body>
        </html>
        """)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": len(_orchestrator.providers) if _orchestrator else 0,
    }


@app.get("/api/providers")
async def list_providers() -> list[ProviderInfo]:
    """List all available providers with their status."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    provider_info = []

    for provider in _orchestrator.providers:
        model_info = provider.get_model_info()
        available_models = (
            provider.list_available_models()
            if hasattr(provider, "list_available_models")
            else [provider.model]
        )

        features = []
        if hasattr(provider, "get_model_info"):
            info = provider.get_model_info()
            if "features" in info:
                features = [
                    f"{k}: {v}" for k, v in info["features"].items() if isinstance(v, bool) and v
                ]

        provider_info.append(
            ProviderInfo(
                name=provider.get_provider_name(),
                display_name=provider.get_provider_name().title(),
                status="available",
                models=available_models,
                default_model=provider.model,
                pricing=model_info.get("pricing", {}),
                features=features,
                available=True,
            )
        )

    return provider_info


@app.post("/api/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    """Submit a consensus query."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Execute quorum
        session = await _orchestrator.execute_quorum(
            query=request.query,
            context=request.context or "",
            mode=request.mode,
        )

        # Broadcast update to WebSocket clients
        await broadcast_session_update(session.session_id)

        return QueryResponse(
            session_id=session.session_id,
            status=session.status.value,
            message=f"Query processed with {request.mode} mode",
            estimated_cost=session.metadata.get("total_cost"),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get session details by ID."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    try:
        session = await _session_manager.get_session(session_id)

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "status": session.status.value,
            "query": session.query,
            "mode": session.mode,
            "provider_responses": session.provider_responses,
            "consensus": session.consensus,
            "metadata": session.metadata,
            "error": session.error,
        }

    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found or expired")


@app.get("/api/sessions")
async def list_sessions(
    status: Optional[str] = None, limit: int = 50, offset: int = 0
) -> dict[str, Any]:
    """List all sessions with pagination."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    # Convert status string to enum if provided
    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    sessions = await _session_manager.list_sessions(status=status_filter)

    # Apply pagination
    total = len(sessions)
    paginated = sessions[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "status": s.status.value,
                "query": s.query[:100] + "..." if len(s.query) > 100 else s.query,
                "mode": s.mode,
                "cost": s.metadata.get("total_cost", 0.0),
            }
            for s in paginated
        ],
    }


@app.get("/api/stats")
async def get_stats() -> dict[str, Any]:
    """Get system statistics."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    stats = await _session_manager.get_stats()

    # Add provider stats
    if _orchestrator:
        stats["providers"] = [
            {
                "name": p.get_provider_name(),
                "model": p.model,
            }
            for p in _orchestrator.providers
        ]

    return stats


@app.post("/api/estimate-cost")
async def estimate_cost(request: CostEstimate) -> dict[str, Any]:
    """Estimate costs for a usage scenario."""
    # Simplified cost estimation
    # In a real implementation, this would use actual provider pricing

    avg_input_tokens = request.query_length // 4  # Rough estimate
    avg_output_tokens = 500  # Assume 500 tokens average response

    costs_by_provider = {}
    total_monthly = 0.0

    # Simplified pricing (would be pulled from actual providers)
    pricing = {
        "anthropic": {"input": 3.0, "output": 15.0},
        "openai": {"input": 2.5, "output": 10.0},
        "gemini": {"input": 0.075, "output": 0.30},
        "cohere": {"input": 3.0, "output": 15.0},
        "mistral": {"input": 2.0, "output": 6.0},
        "novita": {"input": 0.04, "output": 0.04},
        "ollama": {"input": 0.0, "output": 0.0},
    }

    multiplier = 3 if request.mode == "full_deliberation" else 1

    for provider_name in request.providers:
        if provider_name in pricing:
            p = pricing[provider_name]
            cost_per_query = (
                (avg_input_tokens * p["input"] / 1_000_000) +
                (avg_output_tokens * p["output"] / 1_000_000)
            ) * multiplier
            monthly_cost = cost_per_query * request.queries_per_month

            costs_by_provider[provider_name] = {
                "per_query": round(cost_per_query, 4),
                "monthly": round(monthly_cost, 2),
            }
            total_monthly += monthly_cost

    return {
        "per_query_average": round(total_monthly / request.queries_per_month, 4),
        "monthly_total": round(total_monthly, 2),
        "by_provider": costs_by_provider,
        "assumptions": {
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "mode_multiplier": multiplier,
        },
    }


# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time session updates."""
    await websocket.accept()
    _active_websockets.append(websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        _active_websockets.remove(websocket)
        logger.info("WebSocket client disconnected")


async def broadcast_session_update(session_id: str):
    """Broadcast session update to all connected WebSocket clients."""
    if not _session_manager:
        return

    try:
        session = await _session_manager.get_session(session_id)

        update = {
            "type": "session_update",
            "session_id": session_id,
            "status": session.status.value,
            "consensus": session.consensus,
        }

        # Send to all connected clients
        disconnected = []
        for ws in _active_websockets:
            try:
                await ws.send_json(update)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            _active_websockets.remove(ws)

    except Exception as e:
        logger.error(f"Error broadcasting update: {e}")


# Run server
def main():
    """Main entry point for web server."""
    import uvicorn

    uvicorn.run(
        "quorum_mcp.web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
