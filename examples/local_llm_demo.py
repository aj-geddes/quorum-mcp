"""
Local LLM Demo with Ollama

Demonstrates zero-cost consensus building using Ollama with local LLMs.
This example shows how to run Quorum-MCP entirely offline with privacy-preserving
local inference.

Requirements:
- Ollama installed and running: https://ollama.com/download
- Models pulled: ollama pull llama3.2
- Run: python examples/local_llm_demo.py

Note: You can mix local (Ollama) with cloud providers for hybrid consensus!
"""

import asyncio
import os

from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import OllamaProvider
from quorum_mcp.session import get_session_manager


async def demo_local_consensus():
    """Demonstrate consensus with only local Ollama models."""
    print("\n" + "=" * 80)
    print("LOCAL LLM CONSENSUS DEMO (OLLAMA)")
    print("=" * 80)
    print("\nProvider: Ollama (Local LLM)")
    print("Cost: $0.00 (100% Local Inference)")
    print("Privacy: 100% Private (No Data Leaves Your Machine)")
    print("=" * 80)

    # Initialize Ollama provider
    print("\n🔍 Checking Ollama server...")
    try:
        ollama = OllamaProvider(model="llama3.2")

        # Check availability
        availability = await ollama.check_availability()

        if not availability["server_running"]:
            print("\n❌ Error: Ollama server is not running")
            print("\nTo start Ollama:")
            print("  1. Install: https://ollama.com/download")
            print("  2. Start server: ollama serve")
            print("  3. Pull model: ollama pull llama3.2")
            return

        if not availability["model_available"]:
            print(f"\n⚠ Warning: Model 'llama3.2' not found")
            print(f"\nPull the model with:")
            print(f"  ollama pull llama3.2")
            return

        print("✓ Ollama server is running")
        print(f"✓ Model 'llama3.2' is available")

    except Exception as e:
        print(f"\n❌ Failed to connect to Ollama: {e}")
        print("\nMake sure Ollama is installed and running:")
        print("  https://ollama.com/download")
        return

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator with Ollama
    orchestrator = Orchestrator(providers=[ollama], session_manager=session_manager)

    # Execute consensus query
    print("\n" + "-" * 80)
    print("QUERY: What are the top 3 benefits of using local LLMs?")
    print("-" * 80)
    print("\nExecuting local consensus...\n")

    session = await orchestrator.execute_quorum(
        query="What are the top 3 benefits of using local LLMs?",
        context="Focus on privacy, cost, and control aspects",
        mode="quick_consensus",
    )

    print("=" * 80)
    print("CONSENSUS RESULTS")
    print("=" * 80)
    print(f"\nSession ID: {session.session_id}")
    print(f"Status: {session.status.value}")
    print(f"Provider: Ollama (llama3.2)")

    print(f"\n{'-'*80}")
    print("CONSENSUS SUMMARY")
    print(f"{'-'*80}")
    print(f"\n{session.consensus['summary']}")

    if session.consensus.get("agreement_areas"):
        print(f"\n{'-'*80}")
        print("KEY POINTS")
        print(f"{'-'*80}")
        for i, area in enumerate(session.consensus["agreement_areas"], 1):
            print(f"\n{i}. {area}")

    # Cost breakdown
    print(f"\n{'-'*80}")
    print("COST ANALYSIS")
    print(f"{'-'*80}")
    cost_info = session.consensus["cost"]
    print(f"\n💰 Total Cost: ${cost_info['total_cost']:.4f}")
    print(f"📊 Total Tokens: {cost_info['total_tokens_input']} in, {cost_info['total_tokens_output']} out")
    print("\n✨ 100% FREE - Local inference costs nothing!")

    await session_manager.stop()
    print("\n" + "=" * 80)
    print("✓ DEMO COMPLETE")
    print("=" * 80)


async def demo_hybrid_consensus():
    """Demonstrate hybrid consensus with local (Ollama) and cloud providers."""
    print("\n" + "=" * 80)
    print("HYBRID CONSENSUS DEMO (LOCAL + CLOUD)")
    print("=" * 80)
    print("\nCombining local LLMs with cloud providers for best of both worlds:")
    print("  • Ollama: Free, private, offline")
    print("  • Cloud: More capable, diverse perspectives")
    print("=" * 80)

    # Initialize providers
    providers = []

    # Add Ollama (local)
    print("\n🔍 Checking Ollama...")
    try:
        ollama = OllamaProvider(model="llama3.2")
        availability = await ollama.check_availability()
        if availability["server_running"] and availability["model_available"]:
            providers.append(ollama)
            print("✓ Ollama (llama3.2) initialized - $0 cost")
        else:
            print("⚠ Ollama not available - skipping")
    except Exception as e:
        print(f"⚠ Ollama error: {e}")

    # Add cloud providers if keys are available
    if os.getenv("ANTHROPIC_API_KEY"):
        from quorum_mcp.providers import AnthropicProvider

        providers.append(AnthropicProvider())
        print("✓ Anthropic Claude initialized")

    if os.getenv("OPENAI_API_KEY"):
        from quorum_mcp.providers import OpenAIProvider

        providers.append(OpenAIProvider())
        print("✓ OpenAI initialized")

    if os.getenv("GOOGLE_API_KEY"):
        from quorum_mcp.providers import GeminiProvider

        providers.append(GeminiProvider())
        print("✓ Google Gemini initialized")

    if len(providers) == 0:
        print("\n❌ No providers available")
        return

    print(f"\n{len(providers)} provider(s) ready for hybrid consensus")

    # Initialize session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    # Execute consensus query
    print("\n" + "-" * 80)
    print("QUERY: Compare local vs cloud AI deployment strategies")
    print("-" * 80)
    print("\nExecuting hybrid consensus...\n")

    session = await orchestrator.execute_quorum(
        query="What are the key considerations when choosing between local and cloud AI deployment?",
        context="Focus on cost, privacy, performance, and scalability",
        mode="quick_consensus",
    )

    print("=" * 80)
    print("HYBRID CONSENSUS RESULTS")
    print("=" * 80)
    print(f"\nSession ID: {session.session_id}")
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Providers: {', '.join(session.provider_responses.keys())}")

    print(f"\n{'-'*80}")
    print("CONSENSUS SUMMARY")
    print(f"{'-'*80}")
    print(f"\n{session.consensus['summary']}")

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


async def main():
    """Run demos."""
    print("\n" + "=" * 80)
    print("QUORUM-MCP: LOCAL LLM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases consensus building with local LLMs:")
    print("  ✓ Zero cost - No API fees!")
    print("  ✓ 100% Private - Data never leaves your machine")
    print("  ✓ Offline capable - No internet required")
    print("  ✓ Fast - Local inference is quick")

    try:
        # Run Demo 1: Local-only consensus
        await demo_local_consensus()

        # Run Demo 2: Hybrid consensus (if user wants)
        print("\n")
        response = input("Run hybrid (local + cloud) demo? [y/N]: ")
        if response.lower() == "y":
            await demo_hybrid_consensus()

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
