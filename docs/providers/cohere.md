---
layout: page
title: Cohere
description: Configure and use Cohere's Command models with Quorum-MCP
permalink: /providers/cohere/
---

Cohere provides enterprise-focused AI models with excellent RAG (Retrieval-Augmented Generation) capabilities. Production-ready with a generous free tier.

## Quick Setup

```bash
export COHERE_API_KEY="..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `command-r-plus` | 128K | $3.00/1M | $15.00/1M | Default, highest quality |
| `command-r` | 128K | $0.50/1M | $1.50/1M | Balanced |
| `command-light` | 128K | $0.30/1M | $0.60/1M | Speed, cost |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import CohereProvider

# Uses default model (command-r-plus)
provider = CohereProvider()

# Specify a different model
provider = CohereProvider(model="command-r")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | *required* | Your Cohere API key |

## Strengths

- **RAG Excellence:** Built for retrieval-augmented generation
- **Enterprise Ready:** SOC 2 compliance, enterprise support
- **Free Tier:** Generous free usage for development
- **Semantic Search:** Excellent embedding models
- **Multilingual:** Strong multilingual capabilities

## Considerations

- Smaller model selection compared to some providers
- Less consumer-focused documentation
- May require enterprise contact for high volume

## Example

```python
from quorum_mcp.providers import CohereProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_cohere():
    provider = CohereProvider()

    request = ProviderRequest(
        query="Based on the provided documents, what are the key financial metrics?",
        context="[Retrieved document chunks here...]",
        max_tokens=1024,
        temperature=0.3
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")
```

## RAG Integration

Cohere excels at RAG workloads:

```python
from quorum_mcp.providers import CohereProvider

provider = CohereProvider()

# Use with your retrieved documents
request = ProviderRequest(
    query="Answer the user's question using only the provided context",
    context="""
    Document 1: [Retrieved content...]
    Document 2: [Retrieved content...]
    Document 3: [Retrieved content...]

    User Question: What is the company's revenue growth?
    """,
    system_prompt="Only use information from the provided documents. Cite sources.",
    temperature=0.2
)
```

## Error Handling

```python
from quorum_mcp.providers.base import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderError
)

try:
    response = await provider.send_request(request)
except ProviderAuthenticationError:
    print("Invalid API key")
except ProviderRateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Best Practices

1. **Use for RAG applications** — Cohere's specialty
2. **Leverage Command R+ for complex reasoning** — Highest quality
3. **Use Command Light for embeddings** — Fast and cheap
4. **Take advantage of free tier** — Great for prototyping
5. **Consider enterprise tier** — For production workloads
