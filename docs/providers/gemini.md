---
layout: page
title: Google Gemini
description: Configure and use Google's Gemini models with Quorum-MCP
permalink: /providers/gemini/
---

Google's Gemini models offer exceptional speed, cost-effectiveness, and the largest context windows available—up to 2 million tokens with Gemini 1.5 Pro.

## Quick Setup

```bash
export GOOGLE_API_KEY="..."
```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `gemini-2.5-flash` | 200K | $0.15/1M | $0.60/1M | Default, fast |
| `gemini-2.5-pro` | 200K | $1.25/1M | $10.00/1M | Higher quality |
| `gemini-1.5-pro` | 2M | $1.25/1M | $5.00/1M | Long context |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import GeminiProvider

# Uses default model (gemini-2.5-flash)
provider = GeminiProvider()

# Specify a different model
provider = GeminiProvider(model="gemini-1.5-pro")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *required* | Your Google AI API key |

## Strengths

- **Speed:** Fastest inference among major providers
- **Cost:** Very competitive pricing, especially Flash models
- **Context:** Up to 2M tokens (Gemini 1.5 Pro)
- **Multimodal:** Native image, audio, and video understanding
- **Free Tier:** Generous free usage limits

## Considerations

- Newer models may have less community documentation
- Some regional availability restrictions
- Safety filters may be more aggressive

## Example

```python
from quorum_mcp.providers import GeminiProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_gemini():
    provider = GeminiProvider()

    request = ProviderRequest(
        query="Summarize the key points of this 50-page document",
        context="[Your long document here...]",
        max_tokens=2048,
        temperature=0.5
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")
```

## Long Context Use Case

Gemini 1.5 Pro's 2M token context is perfect for:

```python
from quorum_mcp.providers import GeminiProvider

# Use Gemini 1.5 Pro for very long documents
provider = GeminiProvider(model="gemini-1.5-pro")

# Process entire codebases, books, or document collections
request = ProviderRequest(
    query="Analyze this entire codebase and identify potential security vulnerabilities",
    context=entire_codebase_content,  # Can be millions of characters
    max_tokens=4096
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

1. **Use Flash for most tasks** — Best speed and cost efficiency
2. **Use 1.5 Pro for long documents** — 2M context is unmatched
3. **Leverage multimodal capabilities** — Process images alongside text
4. **Take advantage of free tier** — Great for development and testing
5. **Monitor context usage** — Long contexts cost more
