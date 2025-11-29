---
layout: page
title: API Reference
description: Complete API documentation for Quorum-MCP MCP tools and Python interfaces
permalink: /api/
---

Quorum-MCP provides two interfaces: MCP tools for Claude Desktop integration and a Python API for programmatic use.

## MCP Tools

### q_in — Submit Query

Submit a query to the quorum for consensus-based response.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The question or prompt to process |
| `context` | string | No | `null` | Additional context or constraints |
| `mode` | string | No | `"quick_consensus"` | Operational mode |

#### Modes

| Mode | Rounds | Description |
|------|--------|-------------|
| `quick_consensus` | 1 | Fast, parallel independent responses |
| `full_deliberation` | 3 | Independent → Cross-review → Synthesis |
| `devils_advocate` | 2 | Critical analysis with counterarguments |

#### Request Example

```json
{
  "query": "What are the top 3 considerations for API design?",
  "context": "Building a REST API for a SaaS product",
  "mode": "quick_consensus"
}
```

#### Response

```json
{
  "session_id": "abc-123-def-456",
  "status": "completed",
  "message": "Quorum consensus completed using quick_consensus mode",
  "consensus": {
    "summary": "Consensus from 3 AI providers (anthropic, openai, gemini):\n\nAreas of Agreement:\n- Common themes: design, security, versioning...\n\nSynthesized Response:\n...",
    "confidence": 0.85,
    "agreement_areas": ["Common themes: design, security, versioning..."],
    "disagreement_areas": [],
    "key_points": ["Consider versioning strategy...", "Implement proper authentication..."],
    "provider_count": 3,
    "minority_opinions": [],
    "recommendations": [
      {
        "recommendation": "Common themes: design, security, versioning",
        "strength": 0.85,
        "consensus_level": "high"
      }
    ],
    "cost": {
      "total_cost": 0.0156,
      "total_tokens_input": 1250,
      "total_tokens_output": 890,
      "avg_cost_per_provider": 0.0052,
      "providers": {
        "anthropic": 0.0068,
        "openai": 0.0045,
        "gemini": 0.0043
      }
    }
  },
  "confidence": 0.85,
  "cost": 0.0156,
  "providers_used": ["anthropic", "openai", "gemini"]
}
```

### q_out — Retrieve Results

Retrieve the consensus results from a previous quorum session.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | The session ID returned by `q_in` |

#### Request Example

```json
{
  "session_id": "abc-123-def-456"
}
```

#### Response

```json
{
  "session_id": "abc-123-def-456",
  "status": "completed",
  "query": "What are the top 3 considerations for API design?",
  "mode": "quick_consensus",
  "consensus": { ... },
  "confidence": 0.85,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z",
  "metadata": {
    "providers_used": ["anthropic", "openai", "gemini"],
    "total_time": 15.2,
    "rounds": {
      "1": {
        "total_count": 3,
        "successful_count": 3,
        "failed_count": 0,
        "total_cost": 0.0156,
        "total_time": 14.8
      }
    }
  }
}
```

---

## Python API

### Orchestrator

The main class for executing quorum consensus.

```python
from quorum_mcp.orchestrator import Orchestrator
```

#### Constructor

```python
Orchestrator(
    providers: list[Provider],
    session_manager: SessionManager | None = None,
    min_providers: int = 1,
    provider_timeout: float = 60.0,
    max_retries: int = 1,
    check_health: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `providers` | `list[Provider]` | *required* | List of provider instances |
| `session_manager` | `SessionManager` | `None` | Session manager (created if None) |
| `min_providers` | `int` | `1` | Minimum providers required for consensus |
| `provider_timeout` | `float` | `60.0` | Timeout per provider in seconds |
| `max_retries` | `int` | `1` | Retry attempts per provider |
| `check_health` | `bool` | `True` | Check provider health before execution |

#### execute_quorum()

```python
async def execute_quorum(
    query: str,
    context: str | None = None,
    mode: str = "full_deliberation",
    session_id: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> Session
```

Execute a quorum consensus process.

**Returns:** `Session` object with complete results.

**Example:**

```python
session = await orchestrator.execute_quorum(
    query="What is the best database for a startup?",
    context="Small team, rapid iteration",
    mode="quick_consensus"
)

print(f"Confidence: {session.consensus['confidence']:.2%}")
print(f"Summary: {session.consensus['summary']}")
```

---

### Provider Base Class

All providers implement this interface.

```python
from quorum_mcp.providers.base import Provider
```

#### Abstract Methods

```python
class Provider(ABC):
    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request to the AI provider."""
        pass

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Calculate cost in USD."""
        pass

    def get_provider_name(self) -> str:
        """Return provider name."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        pass
```

#### check_health()

```python
async def check_health() -> HealthCheckResult
```

Check provider health status.

**Returns:** `HealthCheckResult` with status and details.

---

### ProviderRequest

Standardized request model.

```python
from quorum_mcp.providers.base import ProviderRequest
```

```python
ProviderRequest(
    query: str,                    # Required: The query/prompt
    system_prompt: str | None,     # System instructions
    context: str | None,           # Additional context
    model: str | None,             # Specific model override
    max_tokens: int = 4096,        # Max response tokens
    temperature: float = 0.7,      # Sampling temperature (0.0-2.0)
    top_p: float | None = None,    # Nucleus sampling
    timeout: float = 60.0,         # Request timeout in seconds
    metadata: dict = {}            # Additional metadata
)
```

---

### ProviderResponse

Standardized response model.

```python
from quorum_mcp.providers.base import ProviderResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Response text |
| `confidence` | `float | None` | Confidence score (0.0-1.0) |
| `model` | `str` | Model that generated response |
| `provider` | `str` | Provider name |
| `tokens_input` | `int` | Input token count |
| `tokens_output` | `int` | Output token count |
| `cost` | `float | None` | Cost in USD |
| `latency` | `float` | Response time in seconds |
| `timestamp` | `datetime` | When response was received |
| `metadata` | `dict` | Additional metadata |
| `error` | `str | None` | Error message if failed |

---

### Session

Session state model.

```python
from quorum_mcp.session import Session, SessionStatus
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Unique identifier |
| `status` | `SessionStatus` | PENDING, IN_PROGRESS, COMPLETED, FAILED |
| `query` | `str` | Original query |
| `mode` | `str` | Operational mode |
| `provider_responses` | `dict` | Responses by provider and round |
| `consensus` | `dict | None` | Final consensus result |
| `metadata` | `dict` | Additional metadata |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |
| `error` | `str | None` | Error message if failed |

---

### Error Types

```python
from quorum_mcp.providers.base import (
    ProviderError,                  # Base error
    ProviderAuthenticationError,    # Invalid API key
    ProviderRateLimitError,         # Rate limit exceeded
    ProviderTimeoutError,           # Request timeout
    ProviderConnectionError,        # Connection failed
    ProviderInvalidRequestError,    # Invalid request format
    ProviderModelError,             # Model unavailable
    ProviderQuotaExceededError,     # Budget/quota exceeded
)

from quorum_mcp.orchestrator import (
    OrchestratorError,              # General orchestration error
    InsufficientProvidersError,     # Too few providers available
)
```

---

### SessionManager

Manages session persistence and retrieval.

```python
from quorum_mcp.session import SessionManager, get_session_manager
```

```python
# Get singleton instance
session_manager = get_session_manager()

# Start the manager
await session_manager.start()

# Create a session
session = await session_manager.create_session(
    query="Your query",
    mode="quick_consensus"
)

# Get a session by ID
session = await session_manager.get_session(session_id)

# Update a session
await session_manager.update_session(session_id, {"status": SessionStatus.COMPLETED})

# Stop the manager
await session_manager.stop()
```
