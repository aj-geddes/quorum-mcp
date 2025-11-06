# Session Management API Reference

## Overview

The Session Management module provides comprehensive state tracking for quorum consultation sessions across multiple rounds of AI deliberation. It implements thread-safe, async-first session management with automatic cleanup and expiration handling.

## Core Components

### Session Model

The `Session` class is a Pydantic model that represents a single quorum consultation.

```python
from quorum_mcp import Session, SessionStatus

session = Session(
    query="What is the best approach for implementing distributed consensus?",
    mode="full_deliberation"
)
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | UUID for unique identification (auto-generated) |
| `created_at` | `datetime` | Creation timestamp in UTC (auto-generated) |
| `updated_at` | `datetime` | Last modification timestamp (auto-updated) |
| `status` | `SessionStatus` | Current session state (PENDING, IN_PROGRESS, COMPLETED, FAILED) |
| `query` | `str` | Original user query submitted to quorum |
| `mode` | `str` | Operational mode (full_deliberation, quick_consensus, devils_advocate) |
| `provider_responses` | `Dict[str, Dict[int, Any]]` | Provider responses organized by provider name and round number |
| `consensus` | `Optional[Dict]` | Final consensus result after deliberation |
| `metadata` | `Dict` | Additional metadata (costs, timing, provider info, etc.) |
| `error` | `Optional[str]` | Error message if session failed |

#### Methods

##### `update_timestamp() -> None`
Update the `updated_at` timestamp to current UTC time.

##### `add_provider_response(provider: str, round_num: int, response: Any) -> None`
Add a provider response for a specific round.

```python
session.add_provider_response("claude", 1, {
    "text": "For distributed consensus, I recommend...",
    "confidence": 0.9
})
```

##### `set_consensus(consensus_data: Dict[str, Any]) -> None`
Set the final consensus result and mark session as COMPLETED.

```python
session.set_consensus({
    "summary": "Both AIs recommend evaluating requirements first",
    "agreement_level": 0.8,
    "recommendations": [...]
})
```

##### `mark_failed(error_message: str) -> None`
Mark session as FAILED with an error message.

```python
session.mark_failed("Provider timeout after 30 seconds")
```

##### `is_expired(ttl_hours: int = 24) -> bool`
Check if session has exceeded its time-to-live.

```python
if session.is_expired(ttl_hours=24):
    print("Session has expired")
```

### SessionStatus Enum

```python
class SessionStatus(str, Enum):
    PENDING = "pending"          # Session created, not yet started
    IN_PROGRESS = "in_progress"  # Active deliberation
    COMPLETED = "completed"      # Consensus reached
    FAILED = "failed"            # Processing error occurred
```

### SessionManager

The `SessionManager` class provides thread-safe session storage and lifecycle management.

#### Initialization

```python
from quorum_mcp import SessionManager

# Create manager with custom settings
manager = SessionManager(
    ttl_hours=24,          # Session time-to-live
    cleanup_interval=3600  # Cleanup runs every hour
)

# Or use the singleton instance
from quorum_mcp import get_session_manager
manager = get_session_manager()
```

#### Methods

##### `async create_session(query: str, mode: str = "full_deliberation") -> Session`
Create a new session with a unique ID.

```python
session = await manager.create_session(
    query="Should we use microservices or monolith?",
    mode="full_deliberation"
)
print(f"Created session: {session.session_id}")
```

##### `async get_session(session_id: str) -> Session`
Retrieve a session by ID. Raises `KeyError` if not found or expired.

```python
try:
    session = await manager.get_session(session_id)
except KeyError:
    print("Session not found or expired")
```

##### `async update_session(session_id: str, updates: Dict[str, Any]) -> Session`
Update session fields atomically. Raises `ValueError` for invalid fields.

```python
updated_session = await manager.update_session(
    session_id,
    {
        "status": SessionStatus.IN_PROGRESS,
        "metadata": {"round": 1, "providers": ["claude", "gpt4"]}
    }
)
```

##### `async list_sessions(status: Optional[SessionStatus] = None, include_expired: bool = False) -> List[Session]`
List all sessions, optionally filtered by status.

```python
# List all active sessions
all_sessions = await manager.list_sessions()

# List only completed sessions
completed = await manager.list_sessions(status=SessionStatus.COMPLETED)

# Include expired sessions
all_including_expired = await manager.list_sessions(include_expired=True)
```

##### `async delete_session(session_id: str) -> None`
Manually delete a session. Raises `KeyError` if not found.

```python
await manager.delete_session(session_id)
```

##### `async get_stats() -> Dict[str, Any]`
Get session manager statistics.

```python
stats = await manager.get_stats()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Active sessions: {stats['active_sessions']}")
print(f"By status: {stats['by_status']}")
```

Returns:
```python
{
    "total_sessions": 10,
    "active_sessions": 8,
    "expired_sessions": 2,
    "by_status": {
        "pending": 2,
        "in_progress": 3,
        "completed": 3,
        "failed": 0
    },
    "ttl_hours": 24
}
```

#### Lifecycle Management

##### `async start() -> None`
Start the background cleanup task.

```python
await manager.start()
```

##### `async stop() -> None`
Stop the background cleanup task gracefully.

```python
await manager.stop()
```

## Usage Examples

### Basic Session Lifecycle

```python
import asyncio
from quorum_mcp import get_session_manager, SessionStatus

async def basic_example():
    manager = get_session_manager()

    # 1. Create session
    session = await manager.create_session(
        query="How to optimize database queries?",
        mode="quick_consensus"
    )

    # 2. Start processing
    await manager.update_session(
        session.session_id,
        {"status": SessionStatus.IN_PROGRESS}
    )

    # 3. Add provider responses
    session = await manager.get_session(session.session_id)
    session.add_provider_response("claude", 1, {
        "analysis": "Use indexes, avoid N+1 queries"
    })
    session.add_provider_response("gpt4", 1, {
        "analysis": "Consider query patterns and data access"
    })

    # 4. Set consensus
    session.set_consensus({
        "recommendation": "Profile first, then optimize based on data"
    })

    # 5. Retrieve final result
    final_session = await manager.get_session(session.session_id)
    print(f"Status: {final_session.status}")
    print(f"Consensus: {final_session.consensus}")

asyncio.run(basic_example())
```

### Multi-Round Deliberation

```python
async def multi_round_example():
    manager = get_session_manager()

    session = await manager.create_session(
        query="Architecture decision: microservices vs monolith?",
        mode="full_deliberation"
    )

    # Round 1: Independent analysis
    session.add_provider_response("claude", 1, {
        "stance": "microservices",
        "reasoning": "Better scalability"
    })
    session.add_provider_response("gpt4", 1, {
        "stance": "it_depends",
        "reasoning": "Depends on team size"
    })

    # Round 2: Cross-review
    session.add_provider_response("claude", 2, {
        "comment": "I agree with GPT-4's nuanced view",
        "revised_stance": "it_depends"
    })
    session.add_provider_response("gpt4", 2, {
        "comment": "Both perspectives have merit"
    })

    # Round 3: Consensus
    session.set_consensus({
        "consensus_reached": True,
        "recommendation": "Start with modular monolith, migrate as needed",
        "agreement_level": 0.95
    })
```

### Concurrent Sessions

```python
async def concurrent_example():
    manager = get_session_manager()

    # Create multiple sessions in parallel
    queries = [
        "Best database for time-series data?",
        "How to implement rate limiting?",
        "Best practices for API versioning?"
    ]

    sessions = await asyncio.gather(*[
        manager.create_session(query)
        for query in queries
    ])

    print(f"Created {len(sessions)} sessions concurrently")
```

### Error Handling

```python
async def error_handling_example():
    manager = get_session_manager()

    # Handle missing session
    try:
        session = await manager.get_session("nonexistent-id")
    except KeyError as e:
        print(f"Session not found: {e}")

    # Handle invalid field update
    session = await manager.create_session("Test query")
    try:
        await manager.update_session(
            session.session_id,
            {"invalid_field": "value"}
        )
    except ValueError as e:
        print(f"Invalid field: {e}")

    # Mark session as failed
    session.mark_failed("Provider timeout")
    print(f"Session failed: {session.error}")
```

## Thread Safety

All `SessionManager` methods are thread-safe using `asyncio.Lock`. Multiple coroutines can safely access the manager concurrently:

```python
async def safe_concurrent_access():
    manager = get_session_manager()

    # All operations are atomic and thread-safe
    tasks = [
        manager.create_session(f"Query {i}")
        for i in range(100)
    ]

    sessions = await asyncio.gather(*tasks)
    print(f"Created {len(sessions)} sessions safely")
```

## Performance Considerations

- **Lookup**: O(1) by session ID
- **Filtering**: O(n) where n is total sessions
- **Lock contention**: Minimal due to fast operations
- **Memory**: Suitable for 100s-1000s of concurrent sessions
- **Cleanup**: Background task prevents memory leaks

## Integration with Orchestration

The SessionManager is designed to integrate seamlessly with the orchestration layer:

```python
async def orchestration_integration():
    manager = get_session_manager()

    # MCP q_in tool creates session
    session = await manager.create_session(query, mode)

    # Orchestrator updates status
    await manager.update_session(
        session.session_id,
        {"status": SessionStatus.IN_PROGRESS}
    )

    # Provider responses stored by round
    session = await manager.get_session(session.session_id)
    for provider in providers:
        response = await provider.send_request(request)
        session.add_provider_response(provider.name, round_num, response)

    # Consensus algorithm sets result
    consensus = calculate_consensus(session.provider_responses)
    session.set_consensus(consensus)

    # MCP q_out tool retrieves result
    final_session = await manager.get_session(session.session_id)
    return final_session.consensus
```

## Testing

Run the comprehensive test suite:

```bash
# Run all session tests
pytest tests/test_session.py -v

# Run with coverage
pytest tests/test_session.py --cov=quorum_mcp.session

# Run specific test
pytest tests/test_session.py::TestSessionManager::test_create_session -v
```

## Demo Script

Run the interactive demo to see session management in action:

```bash
python examples/session_demo.py
```

This demonstrates:
- Basic CRUD operations
- Multi-round deliberation
- Concurrent sessions
- Error handling
- Background cleanup

## Future Enhancements

Potential future additions to the session management system:

1. **Persistent Storage**: Redis or PostgreSQL backend
2. **Distributed Sessions**: Share across multiple server instances
3. **Advanced Querying**: Complex filters and search
4. **Audit Trail**: Complete session history
5. **Webhooks**: Notifications on status changes
6. **Recovery**: Session restoration after server restart
7. **Compression**: Efficient storage for large responses

## See Also

- [Provider Abstraction Layer](./provider_abstraction.md)
- [Orchestration Engine](./orchestration.md)
- [MCP Server Tools](./mcp_tools.md)
