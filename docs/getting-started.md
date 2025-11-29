---
layout: page
title: Getting Started
description: Install and configure Quorum-MCP to start getting consensus-based AI responses
permalink: /getting-started/
---

Get up and running with Quorum-MCP in just a few minutes. This guide covers installation, configuration, and your first query.

## Prerequisites

- **Python 3.10+** — Required for running the server
- **At least one AI provider** — API key for cloud providers, or Ollama for local inference

## Installation

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/aj-geddes/quorum-mcp.git
cd quorum-mcp

# Install with dependencies
pip install -e .

# Or install for development (includes test tools)
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check the CLI is available
quorum-mcp --help
```

## Configuration

### Cloud Providers

Set API keys as environment variables for the providers you want to use:

```bash
# Traditional providers (use any combination)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# Additional providers (optional)
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
export NOVITA_API_KEY="..."
```

You only need **at least one** provider configured. Mix and match based on your needs and budget.

### Local LLMs with Ollama (Free!)

For zero-cost, private local inference:

```bash
# 1. Install Ollama from https://ollama.com/download

# 2. Start the Ollama server
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2

# 4. (Optional) Configure host if not using defaults
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_ENABLE="true"
```

Ollama is enabled by default. Set `OLLAMA_ENABLE="false"` to disable it.

## Running the Server

### Option 1: MCP Server (for Claude Desktop)

```bash
# Start the MCP server
quorum-mcp

# Or run directly
python -m quorum_mcp.server
```

The server uses stdio transport for MCP communication.

### Option 2: Web Dashboard

```bash
# Start the web server
quorum-web

# Or run directly
python -m quorum_mcp.web_server
```

Open `http://localhost:8000` in your browser for an interactive UI.

## Claude Desktop Integration

Add Quorum-MCP to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "quorum-mcp": {
      "command": "quorum-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "your-key",
        "OPENAI_API_KEY": "your-key",
        "GOOGLE_API_KEY": "your-key"
      }
    }
  }
}
```

Restart Claude Desktop after making changes.

## Your First Query

### Using MCP Tools

Once connected, use the `q_in` tool to submit queries:

```json
{
  "query": "What is the best database for a startup?",
  "context": "Small team, rapid iteration, expecting growth",
  "mode": "quick_consensus"
}
```

### Using Python

```python
import asyncio
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, OpenAIProvider, GeminiProvider
from quorum_mcp.session import get_session_manager

async def main():
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
        query="What is the best database for a startup?",
        context="Small team, rapid iteration, expecting growth",
        mode="quick_consensus"
    )

    # Print results
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Summary: {session.consensus['summary']}")
    print(f"Cost: ${session.consensus['cost']['total_cost']:.4f}")

    await session_manager.stop()

asyncio.run(main())
```

## Understanding Results

The consensus response includes:

| Field | Description |
|-------|-------------|
| `summary` | Synthesized response from all providers |
| `confidence` | Score from 0.0 to 1.0 based on agreement |
| `agreement_areas` | Points where providers agree |
| `disagreement_areas` | Points where providers disagree |
| `key_points` | Important points from all responses |
| `cost` | Breakdown of API costs by provider |

## Operational Modes

### Quick Consensus

Single round, all providers respond independently. Fast but may have lower agreement.

```python
mode="quick_consensus"
```

Best for: Straightforward questions, when speed matters

### Full Deliberation

Three rounds of discussion:
1. **Round 1:** Independent analysis
2. **Round 2:** Cross-review (each sees others' responses)
3. **Round 3:** Final synthesis

```python
mode="full_deliberation"
```

Best for: Complex decisions, important questions, when quality matters more than speed

### Devil's Advocate

One provider takes a critical stance, others respond. Useful for challenging assumptions.

```python
mode="devils_advocate"
```

Best for: Validating ideas, finding weaknesses, exploring alternatives

## Next Steps

- **[Providers](/providers/)** — Learn about each AI provider's strengths and configuration
- **[API Reference](/api/)** — Complete documentation of MCP tools and Python API
- **[Architecture](/architecture/)** — Understand how Quorum-MCP works internally
- **[Examples](/examples/)** — See real-world usage patterns
