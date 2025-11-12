"""
Quorum-MCP Web UI Application

FastAPI web application providing a modern UI for interacting with the
Quorum-MCP consensus system. Includes real-time monitoring, query submission,
result visualization, provider health dashboard, and performance analytics.

Features:
- Interactive query submission
- Real-time consensus building visualization
- Provider health monitoring
- Cost tracking and budget management
- Performance benchmarking dashboard
- Session management
- WebSocket for live updates
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quorum_mcp.benchmark import get_benchmark_tracker, BenchmarkMetric
from quorum_mcp.budget import get_budget_manager, BudgetConfig, BudgetPeriod
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import (
    AnthropicProvider,
    GeminiProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
)
from quorum_mcp.rate_limiter import get_rate_limiter_manager
from quorum_mcp.session import SessionManager, SessionStatus, get_session_manager

# Initialize FastAPI app
app = FastAPI(
    title="Quorum-MCP Web UI",
    description="Multi-AI Consensus System Web Interface",
    version="1.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
session_manager: SessionManager | None = None
orchestrator: Orchestrator | None = None
rate_limiter_manager = None
budget_manager = None
benchmark_tracker = None
websocket_connections: List[WebSocket] = []


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for submitting a query."""

    query: str
    context: str | None = None
    mode: str = "quick_consensus"
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class BudgetSetRequest(BaseModel):
    """Request model for setting a budget."""

    provider: str | None = None  # None for global
    limit: float
    period: str = "daily"
    warning_threshold: float = 0.80
    enforce: bool = True


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global session_manager, orchestrator, rate_limiter_manager, budget_manager, benchmark_tracker

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Initialize rate limiter, budget, and benchmarking systems
    from quorum_mcp.rate_limiter import get_rate_limiter_manager
    from quorum_mcp.budget import get_budget_manager
    from quorum_mcp.benchmark import get_benchmark_tracker

    rate_limiter_manager = get_rate_limiter_manager()
    budget_manager = get_budget_manager()
    benchmark_tracker = get_benchmark_tracker()

    # Initialize providers (with environment variable checks)
    providers = []
    try:
        provider = AnthropicProvider()
        # Attach advanced systems
        provider.rate_limiter = rate_limiter_manager.get_limiter("anthropic")
        provider.budget_manager = budget_manager
        provider.benchmark_tracker = benchmark_tracker
        providers.append(provider)
    except Exception:
        pass  # Skip if API key not set

    try:
        provider = OpenAIProvider()
        provider.rate_limiter = rate_limiter_manager.get_limiter("openai")
        provider.budget_manager = budget_manager
        provider.benchmark_tracker = benchmark_tracker
        providers.append(provider)
    except Exception:
        pass

    try:
        provider = GeminiProvider()
        provider.rate_limiter = rate_limiter_manager.get_limiter("google")
        provider.budget_manager = budget_manager
        provider.benchmark_tracker = benchmark_tracker
        providers.append(provider)
    except Exception:
        pass

    try:
        provider = MistralProvider()
        provider.rate_limiter = rate_limiter_manager.get_limiter("mistral")
        provider.budget_manager = budget_manager
        provider.benchmark_tracker = benchmark_tracker
        providers.append(provider)
    except Exception:
        pass

    try:
        provider = OllamaProvider()
        provider.rate_limiter = rate_limiter_manager.get_limiter("ollama")
        provider.budget_manager = budget_manager
        provider.benchmark_tracker = benchmark_tracker
        providers.append(provider)
    except Exception:
        pass  # Skip if Ollama not available

    if not providers:
        print("WARNING: No providers initialized. Set API keys or start Ollama.")

    # Initialize orchestrator
    orchestrator = Orchestrator(
        providers=providers,
        session_manager=session_manager,
        check_health=True,
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    if session_manager:
        await session_manager.stop()

    # Close all WebSocket connections
    for ws in websocket_connections:
        try:
            await ws.close()
        except:
            pass


# API Endpoints

@app.get("/")
async def root():
    """Serve the main web UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return {"message": "Quorum-MCP Web UI", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "providers": len(orchestrator.providers) if orchestrator else 0,
        "session_manager": "running" if session_manager else "stopped",
    }


@app.post("/api/query")
async def submit_query(request: QueryRequest):
    """
    Submit a new consensus query.

    Creates a new session and executes the quorum consensus process.
    """
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    try:
        session = await orchestrator.execute_quorum(
            query=request.query,
            context=request.context,
            mode=request.mode,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "consensus": session.consensus,
            "metadata": session.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details by ID."""
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")

    try:
        session = await session_manager.get_session(session_id)
        return {
            "session_id": session.session_id,
            "query": session.query,
            "status": session.status.value,
            "mode": session.mode,
            "consensus": session.consensus,
            "provider_responses": session.provider_responses,
            "metadata": session.metadata,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """List recent sessions."""
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")

    # Get all sessions
    all_sessions = []
    for session_id, session in session_manager._sessions.items():
        all_sessions.append({
            "session_id": session.session_id,
            "query": session.query[:100] + "..." if len(session.query) > 100 else session.query,
            "status": session.status.value,
            "mode": session.mode,
            "created_at": session.created_at.isoformat(),
        })

    # Sort by created_at descending
    all_sessions.sort(key=lambda x: x["created_at"], reverse=True)

    return {"sessions": all_sessions[:limit], "total": len(all_sessions)}


@app.get("/api/providers")
async def list_providers():
    """List available providers and their status."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    providers_info = []
    for provider in orchestrator.providers:
        # Get health status
        try:
            health = await provider.check_health()
            health_status = health.status.value
            health_details = health.details
        except Exception as e:
            health_status = "error"
            health_details = {"error": str(e)}

        providers_info.append({
            "name": provider.get_provider_name(),
            "model": getattr(provider, "model", "unknown"),
            "health": health_status,
            "health_details": health_details,
        })

    return {"providers": providers_info}


@app.get("/api/rate-limits")
async def get_rate_limits():
    """Get current rate limit status for all providers."""
    manager = get_rate_limiter_manager()
    status = await manager.get_all_status()
    return {"rate_limits": status}


@app.get("/api/budget")
async def get_budget_status():
    """Get current budget status."""
    manager = get_budget_manager()
    status = await manager.get_all_status()
    return {"budgets": status}


@app.post("/api/budget")
async def set_budget(request: BudgetSetRequest):
    """Set or update a budget."""
    manager = get_budget_manager()

    config = BudgetConfig(
        limit=request.limit,
        period=BudgetPeriod(request.period),
        warning_threshold=request.warning_threshold,
        enforce=request.enforce,
        provider=request.provider,
    )

    await manager.set_budget(config)

    return {"message": "Budget set successfully", "config": {
        "provider": request.provider or "global",
        "limit": request.limit,
        "period": request.period,
    }}


@app.get("/api/budget/alerts")
async def get_budget_alerts():
    """Get recent budget alerts."""
    manager = get_budget_manager()
    alerts = await manager.get_all_alerts()

    return {
        "alerts": [
            {
                "timestamp": alert.timestamp.isoformat(),
                "provider": alert.provider or "global",
                "type": alert.alert_type,
                "message": alert.message,
                "current_cost": alert.current_cost,
                "limit": alert.limit,
            }
            for alert in alerts[:50]  # Last 50 alerts
        ]
    }


@app.get("/api/benchmark/summary")
async def get_benchmark_summary():
    """Get performance benchmark summary."""
    tracker = get_benchmark_tracker()
    summary = await tracker.get_performance_summary(time_window=timedelta(hours=24))
    return {"summary": summary}


@app.get("/api/benchmark/providers")
async def get_provider_benchmarks():
    """Get performance benchmarks for all providers."""
    tracker = get_benchmark_tracker()

    if not orchestrator:
        return {"providers": []}

    provider_names = [p.get_provider_name() for p in orchestrator.providers]
    results = await tracker.compare_providers(provider_names, time_window=timedelta(hours=24))

    return {
        "providers": {
            name: {
                "provider": perf.provider,
                "model": perf.model,
                "avg_latency": perf.avg_latency,
                "p95_latency": perf.p95_latency,
                "avg_throughput": perf.avg_throughput,
                "avg_cost_per_1k_tokens": perf.avg_cost_per_1k_tokens,
                "success_rate": perf.success_rate,
                "total_requests": perf.total_requests,
                "total_cost": perf.total_cost,
            }
            for name, perf in results.items()
        }
    }


@app.get("/api/benchmark/leaderboard/{metric}")
async def get_leaderboard(metric: str):
    """Get provider leaderboard for a specific metric."""
    tracker = get_benchmark_tracker()

    try:
        metric_enum = BenchmarkMetric(metric)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")

    leaderboard = await tracker.get_leaderboard(
        metric=metric_enum,
        time_window=timedelta(hours=24),
        limit=10,
    )

    return {
        "metric": metric,
        "leaderboard": [
            {"provider": provider, "score": score}
            for provider, score in leaderboard
        ]
    }


# WebSocket for Real-time Updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)

            # Get current status
            status = {
                "type": "status_update",
                "timestamp": datetime.utcnow().isoformat(),
                "providers": len(orchestrator.providers) if orchestrator else 0,
                "active_sessions": len(session_manager._sessions) if session_manager else 0,
            }

            await websocket.send_json(status)
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


async def broadcast_update(data: Dict):
    """Broadcast an update to all connected WebSocket clients."""
    for ws in websocket_connections:
        try:
            await ws.send_json(data)
        except:
            pass  # Client disconnected


# Mount static files (will be created next)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
