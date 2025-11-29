---
layout: page
title: Anthropic Claude
description: Configure and use Anthropic's Claude models with Quorum-MCP
permalink: /providers/anthropic/
---

Anthropic's Claude models are known for thoughtful, nuanced reasoning and excellent instruction following. Claude excels at complex analysis, writing tasks, and safety-critical applications.

## Quick Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `claude-3-5-sonnet-20241022` | 200K | $3.00/1M | $15.00/1M | Default, balanced |
| `claude-3-opus-20240229` | 200K | $15.00/1M | $75.00/1M | Highest quality |
| `claude-3-haiku-20240307` | 200K | $0.25/1M | $1.25/1M | Speed, cost |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import AnthropicProvider

# Uses default model (claude-3-5-sonnet)
provider = AnthropicProvider()

# Specify a different model
provider = AnthropicProvider(model="claude-3-opus-20240229")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *required* | Your Anthropic API key |

## Strengths

- **Nuanced Reasoning:** Excellent at considering multiple perspectives
- **Instruction Following:** Precisely follows complex instructions
- **Safety:** Built with safety considerations as a priority
- **Long Context:** 200K token context window
- **Writing Quality:** Produces well-structured, clear responses

## Considerations

- Higher cost compared to some alternatives
- May be more conservative in certain responses
- Rate limits apply based on your API tier

## Example

```python
from quorum_mcp.providers import AnthropicProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_claude():
    provider = AnthropicProvider()

    request = ProviderRequest(
        query="Analyze the trade-offs between microservices and monolithic architecture",
        system_prompt="You are a senior software architect. Be specific and practical.",
        max_tokens=2048,
        temperature=0.7
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")
```

## Error Handling

The provider maps Anthropic-specific errors to standard `ProviderError` types:

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

1. **Use Sonnet for most tasks** — Best balance of quality and cost
2. **Reserve Opus for complex analysis** — When quality is paramount
3. **Use Haiku for simple queries** — Fast and cost-effective
4. **Leverage the large context** — Claude handles long documents well
5. **Be specific in prompts** — Claude follows instructions precisely
