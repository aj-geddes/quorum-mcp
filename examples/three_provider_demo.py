"""
Three-Provider Quorum Demo

Demonstrates consensus building with Anthropic Claude, OpenAI GPT-4, and Google Gemini
working together to produce well-rounded responses.

Requirements:
- Set ANTHROPIC_API_KEY environment variable
- Set OPENAI_API_KEY environment variable
- Set GOOGLE_API_KEY environment variable
- Run: python examples/three_provider_demo.py
"""

import asyncio
import os

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from quorum_mcp.session import get_session_manager


async def demo_three_provider_consensus():
    """Demonstrate consensus with Claude, GPT-4, and Gemini."""
    print("\n" + "=" * 80)
    print("THREE-PROVIDER CONSENSUS DEMO")
    print("=" * 80)
    print("\nProviders: Claude (Anthropic), GPT-4 (OpenAI), Gemini (Google)")
    print("Mode: Quick Consensus (single round)")
    print("=" * 80)

    # Initialize all three providers
    providers = []

    if os.getenv("ANTHROPIC_API_KEY"):
        claude = AnthropicProvider()
        providers.append(claude)
        print("✓ Anthropic Claude initialized (claude-3-5-sonnet)")
    else:
        print("⚠ ANTHROPIC_API_KEY not set - skipping Anthropic")

    if os.getenv("OPENAI_API_KEY"):
        gpt4 = OpenAIProvider()
        providers.append(gpt4)
        print("✓ OpenAI GPT-4o initialized")
    else:
        print("⚠ OPENAI_API_KEY not set - skipping OpenAI")

    if os.getenv("GOOGLE_API_KEY"):
        gemini = GeminiProvider()
        providers.append(gemini)
        print("✓ Google Gemini initialized (gemini-2.5-flash)")
    else:
        print("⚠ GOOGLE_API_KEY not set - skipping Gemini")

    if len(providers) == 0:
        print("\n❌ Error: No providers initialized. Set at least one API key.")
        return

    print(f"\n{len(providers)} provider(s) ready for consensus building")

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    # Execute consensus query
    print("\n" + "-" * 80)
    print("QUERY: What are the top 3 best practices for modern Python development?")
    print("-" * 80)
    print("\nExecuting quorum consensus...\n")

    session = await orchestrator.execute_quorum(
        query="What are the top 3 best practices for modern Python development?",
        context="Focus on practices that improve code quality, maintainability, and team collaboration",
        mode="quick_consensus",
    )

    print("=" * 80)
    print("CONSENSUS RESULTS")
    print("=" * 80)
    print(f"\nSession ID: {session.session_id}")
    print(f"Status: {session.status.value}")
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Providers: {', '.join(session.provider_responses.keys())}")

    print(f"\n{'-'*80}")
    print("CONSENSUS SUMMARY")
    print(f"{'-'*80}")
    print(f"\n{session.consensus['summary']}")

    if session.consensus["agreement_areas"]:
        print(f"\n{'-'*80}")
        print("AREAS OF AGREEMENT")
        print(f"{'-'*80}")
        for i, area in enumerate(session.consensus["agreement_areas"], 1):
            print(f"\n{i}. {area}")

    if session.consensus.get("disagreement_areas"):
        print(f"\n{'-'*80}")
        print("AREAS OF DISAGREEMENT")
        print(f"{'-'*80}")
        for i, area in enumerate(session.consensus["disagreement_areas"], 1):
            print(f"\n{i}. {area}")

    # Cost breakdown
    print(f"\n{'-'*80}")
    print("COST ANALYSIS")
    print(f"{'-'*80}")
    cost_info = session.consensus["cost"]
    print(f"\nTotal Cost: ${cost_info['total_cost']:.4f}")
    print(f"Total Tokens: {cost_info['total_tokens_input']} in, {cost_info['total_tokens_output']} out")
    print("\nPer-Provider Breakdown:")
    for provider_name, provider_cost in cost_info["providers"].items():
        print(f"  • {provider_name}: ${provider_cost:.4f}")

    await session_manager.stop()
    print("\n" + "=" * 80)
    print("✓ DEMO COMPLETE")
    print("=" * 80)


async def demo_three_provider_full_deliberation():
    """Demonstrate full deliberation with all three providers."""
    print("\n" + "=" * 80)
    print("THREE-PROVIDER FULL DELIBERATION DEMO")
    print("=" * 80)
    print("\nProviders: Claude, GPT-4, Gemini")
    print("Mode: Full Deliberation (3 rounds with cross-review)")
    print("=" * 80)

    # Initialize providers
    providers = []

    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider())
        print("✓ Claude ready")

    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
        print("✓ GPT-4 ready")

    if os.getenv("GOOGLE_API_KEY"):
        providers.append(GeminiProvider())
        print("✓ Gemini ready")

    if len(providers) < 2:
        print("\n⚠ Full deliberation works best with at least 2 providers")
        if len(providers) == 0:
            print("❌ Error: No providers initialized")
            return

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    # Execute full deliberation
    print("\n" + "-" * 80)
    print("QUERY: Should a startup use microservices or monolith architecture?")
    print("CONTEXT: 8-person team, MVP in 3 months, expected 10K users in 18 months")
    print("-" * 80)
    print("\nExecuting full deliberation (this will take longer)...\n")

    session = await orchestrator.execute_quorum(
        query="Should a startup use microservices or monolith architecture?",
        context="Team of 8 developers, limited DevOps experience, need MVP in 3 months, expecting growth to 10K users in 18 months",
        mode="full_deliberation",
    )

    print("=" * 80)
    print("DELIBERATION RESULTS")
    print("=" * 80)
    print(f"\nSession ID: {session.session_id}")
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Rounds Completed: 3")
    print(f"Providers: {len(session.provider_responses)}")

    print(f"\n{'-'*80}")
    print("FINAL CONSENSUS")
    print(f"{'-'*80}")
    print(f"\n{session.consensus['summary']}")

    if session.consensus.get("recommendations"):
        print(f"\n{'-'*80}")
        print("RECOMMENDATIONS")
        print(f"{'-'*80}")
        for i, rec in enumerate(session.consensus["recommendations"][:5], 1):
            print(f"\n{i}. {rec['recommendation']}")
            print(f"   Confidence: {rec['confidence_level']}")

    # Cost summary
    cost_info = session.consensus["cost"]
    print(f"\n{'-'*80}")
    print("COST SUMMARY")
    print(f"{'-'*80}")
    print(f"Total Cost: ${cost_info['total_cost']:.4f}")
    print(f"Total Tokens: {cost_info['total_tokens_input'] + cost_info['total_tokens_output']}")

    await session_manager.stop()
    print("\n" + "=" * 80)
    print("✓ DELIBERATION COMPLETE")
    print("=" * 80)


async def main():
    """Run demos."""
    print("\n" + "=" * 80)
    print("QUORUM-MCP: THREE-PROVIDER DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases consensus building with:")
    print("  • Anthropic Claude (claude-3-5-sonnet-20241022)")
    print("  • OpenAI GPT-4o (gpt-4o)")
    print("  • Google Gemini (gemini-2.5-flash)")
    print("\nBy combining multiple AI providers, Quorum-MCP produces:")
    print("  ✓ More balanced and comprehensive answers")
    print("  ✓ Higher confidence through cross-validation")
    print("  ✓ Identification of consensus and divergent viewpoints")

    # Check which API keys are available
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))

    available_count = sum([has_anthropic, has_openai, has_gemini])

    print(f"\n{'-'*80}")
    print("API KEY STATUS")
    print(f"{'-'*80}")
    print(f"Anthropic: {'✓ Set' if has_anthropic else '✗ Not set'}")
    print(f"OpenAI:    {'✓ Set' if has_openai else '✗ Not set'}")
    print(f"Gemini:    {'✓ Set' if has_gemini else '✗ Not set'}")
    print(f"\nProviders available: {available_count}/3")

    if available_count == 0:
        print("\n❌ Error: No API keys configured")
        print("Please set at least one of:")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - GOOGLE_API_KEY")
        return

    try:
        # Run Demo 1: Quick Consensus
        await demo_three_provider_consensus()

        # Run Demo 2: Full Deliberation (if user wants)
        if available_count >= 2:
            print("\n")
            response = input("Run full deliberation demo? (this costs more) [y/N]: ")
            if response.lower() == 'y':
                await demo_three_provider_full_deliberation()

        print("\n" + "=" * 80)
        print("✓ ALL DEMOS COMPLETED")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
