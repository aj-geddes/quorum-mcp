"""
Demo script showing Session Management usage

This script demonstrates how to use the SessionManager for tracking
quorum consultation sessions.
"""

import asyncio
from datetime import datetime

from quorum_mcp.session import (
    SessionManager,
    SessionStatus,
)


async def basic_usage_demo():
    """Demonstrate basic session management operations"""
    print("\n=== Basic Session Management Demo ===\n")

    # Create a session manager
    manager = SessionManager()

    # Create a new session
    session = await manager.create_session(
        query="What is the best approach for implementing distributed consensus?",
        mode="full_deliberation",
    )
    print(f"Created session: {session.session_id}")
    print(f"Query: {session.query}")
    print(f"Status: {session.status}")
    print(f"Mode: {session.mode}")

    # Update session status
    await manager.update_session(session.session_id, {"status": SessionStatus.IN_PROGRESS})
    print(f"\nUpdated status to: {SessionStatus.IN_PROGRESS}")

    # Add provider responses
    session = await manager.get_session(session.session_id)
    session.add_provider_response(
        "claude", 1, {"text": "For distributed consensus, I recommend...", "confidence": 0.9}
    )
    session.add_provider_response(
        "gpt4", 1, {"text": "The best approach depends on your requirements...", "confidence": 0.85}
    )
    print("\nAdded responses from 2 providers")
    print(f"Provider responses: {list(session.provider_responses.keys())}")

    # Set consensus
    session.set_consensus(
        {
            "summary": "Both AIs recommend evaluating requirements first",
            "agreement_level": 0.8,
            "recommendations": [
                "Consider CAP theorem trade-offs",
                "Evaluate consistency requirements",
                "Assess network partition tolerance needs",
            ],
        }
    )
    print(f"\nSet consensus - Status: {session.status}")

    # List all sessions
    all_sessions = await manager.list_sessions()
    print(f"\nTotal sessions: {len(all_sessions)}")

    # Get statistics
    stats = await manager.get_stats()
    print("\nSession Statistics:")
    print(f"  Total: {stats['total_sessions']}")
    print(f"  Active: {stats['active_sessions']}")
    print(f"  By status: {stats['by_status']}")


async def multi_round_demo():
    """Demonstrate multi-round deliberation tracking"""
    print("\n\n=== Multi-Round Deliberation Demo ===\n")

    manager = SessionManager()

    # Create session
    session = await manager.create_session(
        query="Should we use microservices or monolith architecture?", mode="full_deliberation"
    )
    print(f"Session: {session.session_id}")

    # Simulate Round 1: Independent analysis
    print("\n--- Round 1: Independent Analysis ---")
    session.add_provider_response(
        "claude",
        1,
        {"stance": "microservices", "reasoning": "Better scalability and team autonomy"},
    )
    session.add_provider_response(
        "gpt4", 1, {"stance": "it_depends", "reasoning": "Depends on team size and complexity"}
    )
    session.add_provider_response(
        "gemini",
        1,
        {"stance": "monolith", "reasoning": "Simpler for small teams, easier debugging"},
    )
    print(f"Collected {len(session.provider_responses)} responses")

    # Simulate Round 2: Cross-review
    print("\n--- Round 2: Cross-Review ---")
    session.add_provider_response(
        "claude",
        2,
        {"comment": "I agree with GPT-4's nuanced view", "revised_stance": "it_depends"},
    )
    session.add_provider_response(
        "gpt4",
        2,
        {
            "comment": "Both perspectives have merit",
            "additional_factors": ["team_size", "deployment_complexity"],
        },
    )
    session.add_provider_response(
        "gemini",
        2,
        {
            "comment": "Valid points about scalability",
            "revised_stance": "monolith_first_then_migrate",
        },
    )

    # Simulate Round 3: Consensus
    print("\n--- Round 3: Consensus Building ---")
    session.set_consensus(
        {
            "consensus_reached": True,
            "recommendation": "Start with modular monolith, migrate to microservices as needed",
            "agreement_level": 0.95,
            "key_factors": [
                "Team size",
                "System complexity",
                "Deployment requirements",
                "Operational maturity",
            ],
            "dissenting_opinions": None,
        }
    )

    print(f"\nFinal Status: {session.status}")
    print(f"Consensus: {session.consensus['recommendation']}")


async def concurrent_sessions_demo():
    """Demonstrate handling multiple concurrent sessions"""
    print("\n\n=== Concurrent Sessions Demo ===\n")

    manager = SessionManager()

    # Create multiple sessions concurrently
    queries = [
        "What is the best database for time-series data?",
        "How to implement rate limiting?",
        "Best practices for API versioning?",
        "When to use GraphQL vs REST?",
        "How to handle authentication at scale?",
    ]

    print("Creating 5 concurrent sessions...")
    sessions = await asyncio.gather(
        *[manager.create_session(query, mode="quick_consensus") for query in queries]
    )

    print(f"Created {len(sessions)} sessions")
    for i, session in enumerate(sessions, 1):
        print(f"  {i}. {session.session_id[:8]}... - {session.query[:50]}...")

    # Update some sessions
    print("\nUpdating sessions in parallel...")
    await asyncio.gather(
        *[
            manager.update_session(sessions[i].session_id, {"status": SessionStatus.IN_PROGRESS})
            for i in range(3)
        ]
    )

    # Get statistics
    stats = await manager.get_stats()
    print("\nCurrent state:")
    print(f"  Pending: {stats['by_status']['pending']}")
    print(f"  In Progress: {stats['by_status']['in_progress']}")


async def session_lifecycle_demo():
    """Demonstrate complete session lifecycle"""
    print("\n\n=== Session Lifecycle Demo ===\n")

    manager = SessionManager()

    # 1. Create
    print("1. Creating session...")
    session = await manager.create_session(
        query="How to optimize database queries?", mode="devils_advocate"
    )
    print(f"   Created: {session.session_id}")
    print(f"   Status: {session.status}")

    # 2. Start processing
    print("\n2. Starting deliberation...")
    await manager.update_session(
        session.session_id,
        {
            "status": SessionStatus.IN_PROGRESS,
            "metadata": {
                "started_at": datetime.utcnow().isoformat(),
                "providers": ["claude", "gpt4"],
            },
        },
    )
    session = await manager.get_session(session.session_id)
    print(f"   Status: {session.status}")
    print(f"   Metadata: {session.metadata}")

    # 3. Collect responses
    print("\n3. Collecting provider responses...")
    session.add_provider_response(
        "claude",
        1,
        {
            "analysis": "Use indexes, avoid N+1 queries",
            "concerns": ["Over-indexing can slow writes"],
        },
    )
    session.add_provider_response(
        "gpt4",
        1,
        {
            "analysis": "Consider query patterns and data access",
            "concerns": ["Premature optimization"],
        },
    )
    print(f"   Responses: {len(session.provider_responses)}")

    # 4. Complete with consensus
    print("\n4. Finalizing consensus...")
    session.set_consensus(
        {
            "recommendation": "Profile first, then optimize based on data",
            "agreement": ["Use appropriate indexes", "Analyze query patterns"],
            "disagreement": ["Timing of optimization"],
            "final_confidence": 0.88,
        }
    )
    print(f"   Status: {session.status}")
    print(f"   Consensus: {session.consensus['recommendation']}")

    # 5. Retrieve final result
    print("\n5. Retrieving final result...")
    final_session = await manager.get_session(session.session_id)
    print(f"   Session ID: {final_session.session_id}")
    print(f"   Status: {final_session.status}")
    print(f"   Created: {final_session.created_at}")
    print(f"   Completed: {final_session.updated_at}")
    print(f"   Duration: {final_session.updated_at - final_session.created_at}")


async def error_handling_demo():
    """Demonstrate error handling"""
    print("\n\n=== Error Handling Demo ===\n")

    manager = SessionManager()

    # Test missing session
    print("1. Testing missing session...")
    try:
        await manager.get_session("nonexistent-id")
    except KeyError as e:
        print(f"   Caught expected error: {e}")

    # Test invalid field update
    print("\n2. Testing invalid field update...")
    session = await manager.create_session("Test query")
    try:
        await manager.update_session(session.session_id, {"invalid_field": "value"})
    except ValueError as e:
        print(f"   Caught expected error: {e}")

    # Test session failure
    print("\n3. Testing session failure...")
    session.mark_failed("Provider timeout after 30 seconds")
    print(f"   Status: {session.status}")
    print(f"   Error: {session.error}")


async def background_cleanup_demo():
    """Demonstrate background cleanup task"""
    print("\n\n=== Background Cleanup Demo ===\n")

    # Create manager with short TTL for demo
    manager = SessionManager(ttl_hours=0.001, cleanup_interval=2)

    print("Creating sessions...")
    for i in range(3):
        await manager.create_session(f"Query {i+1}")

    stats = await manager.get_stats()
    print(f"Active sessions: {stats['active_sessions']}")

    print("\nStarting background cleanup task...")
    await manager.start()

    print("Waiting 3 seconds for cleanup to run...")
    await asyncio.sleep(3)

    stats = await manager.get_stats()
    print("After cleanup:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Expired sessions: {stats['expired_sessions']}")

    await manager.stop()
    print("\nCleanup task stopped")


async def main():
    """Run all demos"""
    print("=" * 60)
    print("QUORUM-MCP SESSION MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    await basic_usage_demo()
    await multi_round_demo()
    await concurrent_sessions_demo()
    await session_lifecycle_demo()
    await error_handling_demo()
    await background_cleanup_demo()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
