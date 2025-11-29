---
layout: page
title: Examples
description: Real-world examples and use cases for Quorum-MCP
permalink: /examples/
---

Learn how to use Quorum-MCP effectively through practical examples.

## Basic Usage

### Simple Consensus Query

```python
import asyncio
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, OpenAIProvider, GeminiProvider
from quorum_mcp.session import get_session_manager

async def simple_query():
    # Initialize providers
    providers = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
    ]

    # Start session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator
    orchestrator = Orchestrator(
        providers=providers,
        session_manager=session_manager
    )

    # Execute consensus
    session = await orchestrator.execute_quorum(
        query="What are the key principles of good API design?",
        mode="quick_consensus"
    )

    # Print results
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Summary:\n{session.consensus['summary']}")
    print(f"\nTotal cost: ${session.consensus['cost']['total_cost']:.4f}")

    await session_manager.stop()

asyncio.run(simple_query())
```

## Operational Modes

### Full Deliberation for Complex Decisions

```python
async def architecture_decision():
    providers = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="Should we migrate from a monolithic architecture to microservices?",
        context="""
        Current situation:
        - 5-year-old Django monolith
        - 50,000 daily active users
        - Team of 8 developers
        - Some performance issues during peak hours
        - Deployment takes 30 minutes

        Constraints:
        - Limited budget for infrastructure changes
        - Need to maintain 99.9% uptime
        - Team has limited Kubernetes experience
        """,
        mode="full_deliberation",
        temperature=0.5
    )

    print("=== Architecture Decision Analysis ===\n")
    print(f"Confidence: {session.consensus['confidence']:.2%}\n")

    print("Key Points:")
    for point in session.consensus['key_points'][:5]:
        print(f"  - {point}")

    print("\nAreas of Agreement:")
    for area in session.consensus['agreement_areas']:
        print(f"  - {area}")

    if session.consensus['disagreement_areas']:
        print("\nAreas of Disagreement:")
        for area in session.consensus['disagreement_areas']:
            print(f"  - {area}")

    print("\nRecommendations:")
    for rec in session.consensus['recommendations']:
        print(f"  [{rec['consensus_level'].upper()}] {rec['recommendation']}")

    await session_manager.stop()

asyncio.run(architecture_decision())
```

### Devil's Advocate for Idea Validation

```python
async def validate_idea():
    providers = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="We should build our own authentication system instead of using Auth0 or Firebase Auth",
        context="""
        Our reasoning:
        - We want full control over user data
        - Auth0 pricing scales poorly for us
        - We have specific requirements for enterprise SSO
        """,
        mode="devils_advocate",
        temperature=0.7
    )

    print("=== Idea Validation (Devil's Advocate) ===\n")

    # The first provider's response is the critique
    print("Critical Analysis:")
    for provider, rounds in session.provider_responses.items():
        if 1 in rounds and 'content' in rounds[1]:
            print(f"\n[{provider}] Round 1 (Critique):")
            print(rounds[1]['content'][:500] + "...")
            break

    print(f"\n\nFinal Confidence: {session.consensus['confidence']:.2%}")
    print(f"\nSummary:\n{session.consensus['summary'][:1000]}")

    await session_manager.stop()

asyncio.run(validate_idea())
```

## Cost-Optimized Configurations

### Budget-Friendly Setup

```python
async def budget_friendly():
    # Use the cheapest providers
    providers = [
        OllamaProvider(model="llama3.2"),      # Free!
        NovitaProvider(),                       # ~$0.04/1M tokens
        GeminiProvider(model="gemini-2.5-flash"),  # ~$0.15/1M tokens
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="Explain the difference between SQL and NoSQL databases",
        mode="quick_consensus",
        max_tokens=512  # Limit response length
    )

    print(f"Total cost: ${session.consensus['cost']['total_cost']:.6f}")
    print("\nCost breakdown:")
    for provider, cost in session.consensus['cost']['providers'].items():
        print(f"  {provider}: ${cost:.6f}")

    await session_manager.stop()

asyncio.run(budget_friendly())
```

### Fully Private (Local Only)

```python
async def fully_private():
    # All local, all free, all private
    providers = [
        OllamaProvider(model="llama3.2"),
        OllamaProvider(model="mistral"),
        OllamaProvider(model="qwen3"),
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(
        providers=providers,
        session_manager=session_manager,
        check_health=True  # Verify models are available
    )

    session = await orchestrator.execute_quorum(
        query="Analyze this confidential business document for risks",
        context="[Your sensitive document content here]",
        mode="full_deliberation"
    )

    # Your data never left your machine!
    print(f"Processed 100% locally")
    print(f"Cost: ${session.consensus['cost']['total_cost']:.2f}")  # Always $0.00

    await session_manager.stop()

asyncio.run(fully_private())
```

## Specialized Use Cases

### Code Review

```python
async def code_review():
    providers = [
        AnthropicProvider(),  # Good at nuanced feedback
        OpenAIProvider(),     # Good at code patterns
        GeminiProvider(),     # Fast analysis
    ]

    code_to_review = '''
def process_user_data(data):
    result = eval(data['expression'])
    query = f"SELECT * FROM users WHERE id = {data['user_id']}"
    db.execute(query)
    return result
    '''

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="Review this Python code for security vulnerabilities and suggest improvements",
        context=f"```python\n{code_to_review}\n```",
        system_prompt="You are a senior security engineer. Be thorough and specific.",
        mode="full_deliberation",
        temperature=0.3
    )

    print("=== Security Code Review ===\n")
    print(session.consensus['summary'])

    await session_manager.stop()

asyncio.run(code_review())
```

### Technical Writing

```python
async def improve_documentation():
    providers = [
        AnthropicProvider(),  # Clear writing
        OpenAIProvider(),     # Structured output
        GeminiProvider(),     # Fast iterations
    ]

    draft = """
    The function takes a list and returns stuff. It uses a loop.
    Call it with your data. It might throw errors sometimes.
    """

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="Improve this technical documentation to be clear, complete, and professional",
        context=f"Original draft:\n{draft}",
        system_prompt="You are a technical writer. Maintain accuracy while improving clarity.",
        mode="full_deliberation"
    )

    print("=== Improved Documentation ===\n")
    print(session.consensus['summary'])

    await session_manager.stop()

asyncio.run(improve_documentation())
```

### Research Synthesis

```python
async def research_synthesis():
    providers = [
        AnthropicProvider(model="claude-3-opus-20240229"),  # Best reasoning
        OpenAIProvider(model="gpt-4o"),
        GeminiProvider(model="gemini-2.5-pro"),
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(providers=providers, session_manager=session_manager)

    session = await orchestrator.execute_quorum(
        query="What is the current scientific consensus on the health effects of intermittent fasting?",
        context="Focus on peer-reviewed research from the last 5 years",
        system_prompt="You are a research scientist. Cite specific studies when possible. Distinguish between established findings and emerging research.",
        mode="full_deliberation",
        temperature=0.3
    )

    print("=== Research Synthesis ===\n")
    print(f"Confidence: {session.consensus['confidence']:.2%}\n")
    print(session.consensus['summary'])

    await session_manager.stop()

asyncio.run(research_synthesis())
```

## Error Handling

### Graceful Degradation

```python
async def robust_query():
    providers = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
        OllamaProvider(),  # Fallback if cloud fails
    ]

    session_manager = get_session_manager()
    await session_manager.start()

    orchestrator = Orchestrator(
        providers=providers,
        session_manager=session_manager,
        min_providers=2,  # Need at least 2 to succeed
        max_retries=2,    # Retry failed requests
        check_health=True  # Skip unhealthy providers
    )

    try:
        session = await orchestrator.execute_quorum(
            query="What are best practices for error handling?",
            mode="quick_consensus"
        )

        print(f"Success! Used {session.consensus['provider_count']} providers")
        print(f"Providers: {session.metadata.get('providers_used', [])}")

    except InsufficientProvidersError as e:
        print(f"Not enough providers available: {e}")
        # Handle gracefully - maybe use cached response

    await session_manager.stop()

asyncio.run(robust_query())
```

## Running the Demo Scripts

The repository includes ready-to-run demos:

```bash
# Three-provider cloud demo
python examples/three_provider_demo.py

# Local LLM demo with Ollama
python examples/local_llm_demo.py

# End-to-end demo with all modes
python examples/end_to_end_demo.py

# Session management demo
python examples/session_demo.py
```
