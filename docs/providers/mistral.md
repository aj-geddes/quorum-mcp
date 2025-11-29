---
layout: page
title: Mistral AI
description: Configure and use Mistral AI models with Quorum-MCP
permalink: /providers/mistral/
---

Mistral AI is a European AI company offering excellent price-to-performance ratios. Their models are GDPR compliant and competitive with larger providers.

## Quick Setup

```bash
export MISTRAL_API_KEY="..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `mistral-large-latest` | 128K | $2.00/1M | $6.00/1M | Default, highest quality |
| `mistral-small-latest` | 128K | $0.20/1M | $0.60/1M | Cost-effective |
| `open-mixtral-8x22b` | 64K | $2.00/1M | $6.00/1M | Open-source, powerful |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import MistralProvider

# Uses default model (mistral-large-latest)
provider = MistralProvider()

# Specify a different model
provider = MistralProvider(model="mistral-small-latest")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | *required* | Your Mistral API key |

## Strengths

- **Price-Performance:** Excellent value for capability
- **EU Compliance:** GDPR compliant, EU-based
- **Multilingual:** Strong European language support
- **Open Models:** Some models available as open-source
- **Code Skills:** Excellent code generation

## Considerations

- Smaller context than some competitors
- Newer provider with less documentation
- Enterprise features still developing

## Example

```python
from quorum_mcp.providers import MistralProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_mistral():
    provider = MistralProvider()

    request = ProviderRequest(
        query="Write a Django REST API endpoint for user authentication",
        system_prompt="You are a senior backend developer. Write production-ready code.",
        max_tokens=2048,
        temperature=0.3
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")
```

## EU Compliance Use Case

For GDPR-sensitive applications:

```python
from quorum_mcp.providers import MistralProvider

# Mistral processes data in the EU
provider = MistralProvider()

# Personal data can be processed with GDPR compliance
request = ProviderRequest(
    query="Analyze this customer feedback and identify sentiment",
    context=customer_feedback_data,
    temperature=0.5
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

1. **Use Mistral Large for quality** — Competitive with GPT-4
2. **Use Mistral Small for volume** — Great price-performance
3. **Consider for EU workloads** — GDPR compliance built-in
4. **Good for code generation** — Particularly strong at code
5. **Check open-source options** — Mixtral can be self-hosted
