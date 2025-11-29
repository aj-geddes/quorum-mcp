---
layout: page
title: Novita AI
description: Configure and use Novita AI's ultra-low cost models with Quorum-MCP
permalink: /providers/novita/
---

Novita AI provides access to open-source models at ultra-low costs through an OpenAI-compatible API. Perfect for budget-constrained applications and high-volume workloads.

## Quick Setup

```bash
export NOVITA_API_KEY="..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `meta-llama/llama-3.3-70b-instruct` | 128K | $0.04/1M | $0.20/1M | Default, balanced |
| `deepseek/deepseek-r1` | 64K | $0.14/1M | $0.14/1M | Reasoning |
| `qwen/qwen-2.5-72b-instruct` | 128K | $0.04/1M | $0.20/1M | Multilingual |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import NovitaProvider

# Uses default model (llama-3.3-70b)
provider = NovitaProvider()

# Specify a different model
provider = NovitaProvider(model="deepseek/deepseek-r1")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NOVITA_API_KEY` | *required* | Your Novita API key |

## Strengths

- **Ultra-Low Cost:** Among the cheapest options available
- **OpenAI Compatible:** Drop-in replacement for OpenAI API
- **Model Variety:** Access to multiple open-source models
- **No Rate Limits:** More relaxed usage limits
- **Simple Pricing:** Straightforward per-token pricing

## Considerations

- Quality varies by model
- Newer provider with less track record
- Support may be limited
- Models are open-source (less refinement than proprietary)

## Example

```python
from quorum_mcp.providers import NovitaProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_novita():
    provider = NovitaProvider()

    request = ProviderRequest(
        query="Explain quantum computing in simple terms",
        system_prompt="You are a helpful science teacher.",
        max_tokens=1024,
        temperature=0.7
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.6f}")  # Note the 6 decimal places for very low costs
```

## High-Volume Use Case

Perfect for applications needing many API calls:

```python
from quorum_mcp.providers import NovitaProvider

provider = NovitaProvider()

# Process thousands of items cost-effectively
for item in large_dataset:
    request = ProviderRequest(
        query=f"Classify this text: {item['text']}",
        max_tokens=50,
        temperature=0
    )
    response = await provider.send_request(request)
    # Each call costs fractions of a cent
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

1. **Use for high-volume tasks** — Cost scales linearly
2. **Experiment with models** — Different models excel at different tasks
3. **Lower temperature for classification** — More consistent results
4. **Good for prototyping** — Cheap experimentation
5. **Consider DeepSeek for reasoning** — Strong at logical tasks
