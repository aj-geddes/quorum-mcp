---
layout: page
title: OpenAI GPT-4
description: Configure and use OpenAI's GPT-4 models with Quorum-MCP
permalink: /providers/openai/
---

OpenAI's GPT-4 models offer broad knowledge and strong reasoning capabilities. They're the industry standard for many AI applications.

## Quick Setup

```bash
export OPENAI_API_KEY="sk-..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `gpt-4o` | 128K | $2.50/1M | $10.00/1M | Default, best quality |
| `gpt-4o-mini` | 128K | $0.15/1M | $0.60/1M | Cost-effective |
| `gpt-4-turbo` | 128K | $10.00/1M | $30.00/1M | Legacy, powerful |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import OpenAIProvider

# Uses default model (gpt-4o)
provider = OpenAIProvider()

# Specify a different model
provider = OpenAIProvider(model="gpt-4o-mini")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |

## Strengths

- **Broad Knowledge:** Extensive training data coverage
- **Strong Reasoning:** Good at logical analysis and math
- **Code Generation:** Excellent code completion and generation
- **Ecosystem:** Wide tool and library support
- **Multimodal:** Vision capabilities available

## Considerations

- Costs can add up for high-volume usage
- Token-based rate limiting
- May sometimes be verbose

## Example

```python
from quorum_mcp.providers import OpenAIProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_gpt4():
    provider = OpenAIProvider()

    request = ProviderRequest(
        query="Write a Python function to validate email addresses with regex",
        system_prompt="You are a senior Python developer. Write clean, well-documented code.",
        max_tokens=1024,
        temperature=0.3
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")
```

## Token Counting

OpenAI uses tiktoken for accurate token counting:

```python
# The provider automatically counts tokens
tokens = await provider.count_tokens("Your text here")
print(f"Token count: {tokens}")
```

## Error Handling

```python
from quorum_mcp.providers.base import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderQuotaExceededError,
    ProviderError
)

try:
    response = await provider.send_request(request)
except ProviderAuthenticationError:
    print("Invalid API key")
except ProviderRateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ProviderQuotaExceededError:
    print("Usage quota exceeded")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Best Practices

1. **Use GPT-4o for quality** — Best balance of capability and speed
2. **Use GPT-4o-mini for volume** — 15x cheaper for simpler tasks
3. **Set appropriate temperature** — Lower (0.1-0.3) for factual, higher (0.7-1.0) for creative
4. **Use system prompts** — They significantly improve response quality
5. **Monitor usage** — Set up billing alerts in the OpenAI dashboard
