---
layout: page
title: Architecture
description: Technical architecture and design of Quorum-MCP
permalink: /architecture/
---

Quorum-MCP follows a modular architecture designed for extensibility, reliability, and async-first operation.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Client                               │
│                     (Claude Desktop)                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ stdio
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastMCP Server                              │
│                    ┌──────────┬──────────┐                       │
│                    │  q_in    │  q_out   │                       │
│                    └────┬─────┴────┬─────┘                       │
└─────────────────────────┼──────────┼────────────────────────────┘
                          │          │
                          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Orchestrator                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Consensus Engine                         │    │
│  │  • Mode selection (quick/full/devils_advocate)           │    │
│  │  • Round management                                       │    │
│  │  • Agreement detection                                    │    │
│  │  • Confidence scoring                                     │    │
│  │  • Response synthesis                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Provider    │ │   Provider    │ │   Provider    │
│  (Anthropic)  │ │   (OpenAI)    │ │   (Gemini)    │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼
   Claude API        OpenAI API       Google API
```

## Core Components

### FastMCP Server (`server.py`)

The entry point for MCP clients. Exposes two tools:

- **`q_in`**: Submit queries for consensus processing
- **`q_out`**: Retrieve results by session ID

Uses stdio transport for Claude Desktop compatibility.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Quorum-MCP")

@mcp.tool()
async def q_in(query: str, context: str | None, mode: str) -> dict:
    # Validate inputs
    # Execute orchestrator
    # Return results
    pass

@mcp.tool()
async def q_out(session_id: str) -> dict:
    # Retrieve session
    # Return session data
    pass
```

### Orchestrator (`orchestrator.py`)

Coordinates multi-provider consensus through configurable modes.

#### Operational Modes

**Quick Consensus (1 round)**
```
All Providers ──parallel──► Responses ──► Consensus
```

**Full Deliberation (3 rounds)**
```
Round 1: Independent Analysis
    All Providers ──parallel──► Initial Responses

Round 2: Cross-Review
    Each Provider sees others' responses ──► Refined Responses

Round 3: Final Synthesis
    All Providers ──parallel──► Final Responses ──► Consensus
```

**Devil's Advocate (2 rounds)**
```
Round 1: Critical Analysis
    Provider 1 ──► Critical/opposing response

Round 2: Defense
    Other Providers see critique ──► Counter-responses ──► Consensus
```

#### Key Methods

```python
class Orchestrator:
    async def execute_quorum(query, context, mode, ...) -> Session:
        # Main entry point

    async def _execute_quick_consensus(...):
        # Single round execution

    async def _execute_full_deliberation(...):
        # Three-round execution

    async def _execute_devils_advocate(...):
        # Critical analysis execution

    async def _run_round(session_id, round_num, providers, prompt, ...):
        # Execute single round with all providers

    async def _build_consensus(session_id) -> dict:
        # Analyze responses and build consensus
```

### Provider System (`providers/`)

Abstract base class pattern for provider implementations.

```python
class Provider(ABC):
    @abstractmethod
    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        pass

    # Built-in methods
    async def check_health(self) -> HealthCheckResult:
        # Health check implementation

    async def validate_request(self, request: ProviderRequest):
        # Request validation

    async def check_rate_limits(self, estimated_tokens: int):
        # Rate limiting
```

#### Error Hierarchy

```
ProviderError (base)
├── ProviderAuthenticationError
├── ProviderRateLimitError
├── ProviderTimeoutError
├── ProviderConnectionError
├── ProviderInvalidRequestError
├── ProviderModelError
└── ProviderQuotaExceededError
```

### Session Management (`session.py`)

Pydantic-based session state tracking.

```python
class Session(BaseModel):
    session_id: str
    status: SessionStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    query: str
    mode: str
    provider_responses: dict[str, dict[int, Any]]  # provider -> round -> response
    consensus: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    error: str | None
```

#### Session Lifecycle

```
PENDING ──► IN_PROGRESS ──► COMPLETED
                │
                └──► FAILED
```

## Data Flow

### Request Processing

1. **MCP Client** sends `q_in` request
2. **Server** validates inputs, creates session
3. **Orchestrator** selects mode, filters healthy providers
4. **Providers** execute in parallel with retry logic
5. **Orchestrator** builds consensus from responses
6. **Server** returns results with session ID

### Response Structure

```json
{
  "consensus": {
    "summary": "Synthesized response...",
    "confidence": 0.85,
    "agreement_areas": [...],
    "disagreement_areas": [...],
    "key_points": [...],
    "provider_count": 3,
    "minority_opinions": [...],
    "recommendations": [...],
    "cost": {
      "total_cost": 0.0156,
      "providers": {...}
    }
  }
}
```

## Consensus Algorithm

### Confidence Scoring

```python
def _calculate_confidence(agreement_areas, disagreement_areas, provider_count):
    # Base confidence from provider count (max 0.8)
    base_confidence = min(provider_count / 5.0, 0.8)

    # Adjust for agreement/disagreement
    agreement_boost = min(len(agreement_areas) * 0.05, 0.15)
    disagreement_penalty = min(len(disagreement_areas) * 0.05, 0.15)

    confidence = base_confidence + agreement_boost - disagreement_penalty

    return max(0.0, min(1.0, confidence))
```

### Agreement Detection

- Word frequency analysis across responses
- Points mentioned by >50% of providers flagged as agreements
- Semantic similarity (future enhancement)

### Synthesis

- Aggregate key points from all responses
- Highlight consensus areas
- Preserve minority opinions
- Generate weighted recommendations

## Extension Points

### Adding a New Provider

1. Create `providers/<name>_provider.py`:

```python
from quorum_mcp.providers.base import Provider, ProviderRequest, ProviderResponse

class MyProvider(Provider):
    def __init__(self, api_key: str = None, model: str = "default-model"):
        self.api_key = api_key or os.getenv("MY_API_KEY")
        self.model = model
        # Initialize client

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        # Call API, return ProviderResponse
        pass

    async def count_tokens(self, text: str) -> int:
        # Implement tokenization
        pass

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        # Calculate cost
        pass

    def get_provider_name(self) -> str:
        return "my_provider"

    def get_model_info(self) -> dict:
        return {"name": self.model, "context_window": 8192}
```

2. Export from `providers/__init__.py`
3. Add initialization to `server.py`
4. Add tests in `tests/test_<name>_provider.py`

### Custom Consensus Algorithms

Override `_build_consensus` in Orchestrator subclass:

```python
class CustomOrchestrator(Orchestrator):
    async def _build_consensus(self, session_id: str) -> dict:
        # Custom consensus logic
        pass
```

## Performance Considerations

### Parallelization

- All providers queried concurrently within each round
- `asyncio.gather` for parallel execution
- Timeouts prevent slow providers from blocking

### Caching

- Session data stored in memory by default
- Extend SessionManager for persistent storage

### Rate Limiting

- Per-provider rate limiting support
- Configurable retry with exponential backoff
- Health checks filter unhealthy providers

## Security

### API Key Management

- Keys read from environment variables
- Never logged or stored in sessions
- Per-provider key isolation

### Input Validation

- Query length limits (50K characters)
- Context length limits (100K characters)
- Mode validation against allowed values

### Data Privacy

- Session data stored locally
- Ollama option for fully local processing
- No telemetry or external data sharing
