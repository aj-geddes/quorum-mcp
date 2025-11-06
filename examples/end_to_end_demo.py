"""
End-to-End Quorum-MCP Demo

This script demonstrates the complete workflow of the Quorum-MCP system:
1. Provider initialization
2. Session management
3. Orchestration with multiple modes
4. Consensus building

Requirements:
- Set ANTHROPIC_API_KEY environment variable
- Set OPENAI_API_KEY environment variable
- Run: python examples/end_to_end_demo.py
"""

import asyncio
import os

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, OpenAIProvider
from quorum_mcp.session import get_session_manager


async def demo_quick_consensus():
    """Demonstrate quick consensus mode (single round)."""
    print("\n" + "=" * 80)
    print("DEMO 1: Quick Consensus Mode")
    print("=" * 80)

    # Initialize providers
    claude = AnthropicProvider()
    gpt4 = OpenAIProvider()

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=[claude, gpt4], session_manager=session_manager)

    # Execute quick consensus
    print("\nQuery: What are the top 3 considerations for API design?")
    print("Mode: quick_consensus (single round, fast)")
    print("\nExecuting...")

    session = await orchestrator.execute_quorum(
        query="What are the top 3 considerations for API design?",
        context="Building a modern RESTful API for a SaaS product",
        mode="quick_consensus",
    )

    print(f"\n✓ Consensus completed!")
    print(f"  Session ID: {session.session_id}")
    print(f"  Status: {session.status.value}")
    print(f"  Confidence: {session.consensus['confidence']:.2f}")
    print(f"  Cost: ${session.consensus['cost']['total_cost']:.4f}")
    print(f"\nConsensus Summary:")
    print(f"  {session.consensus['summary'][:300]}...")

    if session.consensus["agreement_areas"]:
        print(f"\nAreas of Agreement:")
        for i, area in enumerate(session.consensus["agreement_areas"][:3], 1):
            print(f"  {i}. {area}")

    await session_manager.stop()


async def demo_full_deliberation():
    """Demonstrate full deliberation mode (3 rounds)."""
    print("\n" + "=" * 80)
    print("DEMO 2: Full Deliberation Mode")
    print("=" * 80)

    # Initialize providers
    claude = AnthropicProvider()
    gpt4 = OpenAIProvider()

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=[claude, gpt4], session_manager=session_manager)

    # Execute full deliberation
    print("\nQuery: Should we use microservices or monolith architecture?")
    print("Context: Team of 8 developers, expecting 10K users in 18 months")
    print("Mode: full_deliberation (3 rounds with cross-review)")
    print("\nExecuting...")

    session = await orchestrator.execute_quorum(
        query="Should we use microservices or monolith architecture for our new project?",
        context="Team of 8 developers, limited DevOps, need MVP in 3 months, expecting growth to 10K users in 18 months",
        mode="full_deliberation",
    )

    print(f"\n✓ Full deliberation completed!")
    print(f"  Session ID: {session.session_id}")
    print(f"  Status: {session.status.value}")
    print(f"  Confidence: {session.consensus['confidence']:.2f}")
    print(f"  Providers: {len(session.consensus['cost'])} providers participated")
    print(f"  Cost: ${session.consensus['cost']['total_cost']:.4f}")
    print(f"  Tokens: {session.consensus['cost']['total_tokens_input']} in, {session.consensus['cost']['total_tokens_output']} out")

    print(f"\nConsensus Summary:")
    print(f"  {session.consensus['summary'][:400]}...")

    if session.consensus["agreement_areas"]:
        print(f"\nAreas of Strong Agreement:")
        for i, area in enumerate(session.consensus["agreement_areas"][:3], 1):
            print(f"  {i}. {area}")

    if session.consensus["disagreement_areas"]:
        print(f"\nAreas of Disagreement:")
        for i, area in enumerate(session.consensus["disagreement_areas"][:2], 1):
            print(f"  {i}. {area}")

    if session.consensus["recommendations"]:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(session.consensus["recommendations"][:3], 1):
            print(f"  {i}. {rec['recommendation']} (confidence: {rec['confidence_level']})")

    await session_manager.stop()


async def demo_devils_advocate():
    """Demonstrate devil's advocate mode."""
    print("\n" + "=" * 80)
    print("DEMO 3: Devil's Advocate Mode")
    print("=" * 80)

    # Initialize providers
    claude = AnthropicProvider()
    gpt4 = OpenAIProvider()

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=[claude, gpt4], session_manager=session_manager)

    # Execute devil's advocate
    print("\nQuery: We should deploy directly to production without staging")
    print("Mode: devils_advocate (critical analysis)")
    print("\nExecuting...")

    session = await orchestrator.execute_quorum(
        query="We should deploy directly to production without a staging environment to move faster",
        context="Small startup, need to ship features quickly, limited resources",
        mode="devils_advocate",
    )

    print(f"\n✓ Devil's advocate analysis completed!")
    print(f"  Session ID: {session.session_id}")
    print(f"  Status: {session.status.value}")
    print(f"  Confidence: {session.consensus['confidence']:.2f}")
    print(f"  Cost: ${session.consensus['cost']['total_cost']:.4f}")

    print(f"\nConsensus Summary:")
    print(f"  {session.consensus['summary'][:400]}...")

    if session.consensus["minority_opinions"]:
        print(f"\nDissenting/Critical Opinions:")
        for opinion in session.consensus["minority_opinions"][:2]:
            print(f"  • {opinion['provider']}: {opinion['opinion'][:200]}...")

    await session_manager.stop()


async def demo_session_retrieval():
    """Demonstrate session storage and retrieval."""
    print("\n" + "=" * 80)
    print("DEMO 4: Session Storage and Retrieval")
    print("=" * 80)

    # Initialize providers
    claude = AnthropicProvider()
    gpt4 = OpenAIProvider()

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=[claude, gpt4], session_manager=session_manager)

    # Execute a query
    print("\nExecuting query...")
    session = await orchestrator.execute_quorum(
        query="What is the best way to handle errors in async Python code?",
        context="Building an async web service with FastAPI",
        mode="quick_consensus",
    )

    print(f"✓ Session created: {session.session_id}")

    # Retrieve the session later
    print(f"\nRetrieving session {session.session_id}...")
    retrieved_session = await session_manager.get_session(session.session_id)

    if retrieved_session:
        print(f"✓ Session retrieved successfully!")
        print(f"  Query: {retrieved_session.query}")
        print(f"  Status: {retrieved_session.status.value}")
        print(f"  Confidence: {retrieved_session.consensus['confidence']:.2f}")
        print(f"  Created: {retrieved_session.created_at}")
        print(f"  Updated: {retrieved_session.updated_at}")

    # List all sessions
    print(f"\nListing all sessions...")
    all_sessions = await session_manager.list_sessions()
    print(f"✓ Found {len(all_sessions)} session(s)")

    # Get statistics
    stats = await session_manager.get_stats()
    print(f"\nSession Manager Statistics:")
    print(f"  Total sessions: {stats['total']}")
    print(f"  Active: {stats['by_status'].get('in_progress', 0)}")
    print(f"  Completed: {stats['by_status'].get('completed', 0)}")

    await session_manager.stop()


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("QUORUM-MCP END-TO-END DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo will showcase:")
    print("1. Quick Consensus Mode - Fast single-round consensus")
    print("2. Full Deliberation Mode - Multi-round deliberation with cross-review")
    print("3. Devil's Advocate Mode - Critical analysis and balanced viewpoints")
    print("4. Session Management - Storage and retrieval of consensus results")

    # Check environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: At least one API key must be set")
        print("   Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY environment variables")
        return

    try:
        # Run demos
        await demo_quick_consensus()
        await demo_full_deliberation()
        await demo_devils_advocate()
        await demo_session_retrieval()

        print("\n" + "=" * 80)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
