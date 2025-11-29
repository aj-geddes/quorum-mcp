---
layout: page
title: Configuration
description: Configure Quorum-MCP providers, modes, and advanced settings
permalink: /configuration/
---

Quorum-MCP is configured primarily through environment variables and constructor parameters.

## Environment Variables

### Cloud Provider API Keys

```bash
# At least one provider is required
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
export NOVITA_API_KEY="..."
```

### Ollama (Local LLM)

```bash
# Ollama is enabled by default
export OLLAMA_ENABLE="true"           # Set to "false" to disable
export OLLAMA_HOST="http://localhost:11434"  # Custom server URL
```

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

### Basic Configuration

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

### With Python Path

If `quorum-mcp` isn't in your PATH:

```json
{
  "mcpServers": {
    "quorum-mcp": {
      "command": "/path/to/python",
      "args": ["-m", "quorum_mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key"
      }
    }
  }
}
```

### Ollama Only (Free, Private)

```json
{
  "mcpServers": {
    "quorum-mcp": {
      "command": "quorum-mcp",
      "env": {
        "OLLAMA_ENABLE": "true",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

## Python Configuration

### Orchestrator Options

```python
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import AnthropicProvider, OpenAIProvider, GeminiProvider

providers = [
    AnthropicProvider(),
    OpenAIProvider(),
    GeminiProvider(),
]

orchestrator = Orchestrator(
    providers=providers,
    min_providers=2,        # Require at least 2 successful responses
    provider_timeout=60.0,  # Timeout per provider (seconds)
    max_retries=1,          # Retry failed requests once
    check_health=True,      # Check provider health before execution
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `providers` | *required* | List of provider instances |
| `min_providers` | `1` | Minimum providers for valid consensus |
| `provider_timeout` | `60.0` | Seconds before provider timeout |
| `max_retries` | `1` | Retry attempts for failed requests |
| `check_health` | `True` | Run health checks before queries |

### Provider Configuration

Each provider accepts configuration parameters:

```python
from quorum_mcp.providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-...",           # Override env variable
    model="claude-3-opus-20240229", # Specific model
)
```

### Request Configuration

Configure individual requests:

```python
from quorum_mcp.providers.base import ProviderRequest

request = ProviderRequest(
    query="Your question here",
    system_prompt="You are a helpful assistant",
    context="Additional context",
    model="gpt-4o",          # Override provider default
    max_tokens=4096,         # Maximum response tokens
    temperature=0.7,         # Sampling temperature (0.0-2.0)
    top_p=0.9,               # Nucleus sampling
    timeout=120.0,           # Request timeout
)
```

## Mode Selection

### Quick Consensus

Best for straightforward queries where speed matters.

```python
session = await orchestrator.execute_quorum(
    query="What is the capital of France?",
    mode="quick_consensus"
)
```

**Characteristics:**
- Single round
- Parallel execution
- Fastest mode
- May have lower agreement

### Full Deliberation

Best for complex decisions requiring thorough analysis.

```python
session = await orchestrator.execute_quorum(
    query="Should we use microservices or monolith?",
    context="Building a new e-commerce platform",
    mode="full_deliberation"
)
```

**Characteristics:**
- Three rounds
- Cross-review between providers
- Higher quality consensus
- 3x the cost and time

### Devil's Advocate

Best for challenging assumptions and finding weaknesses.

```python
session = await orchestrator.execute_quorum(
    query="We should skip unit tests to ship faster",
    mode="devils_advocate"
)
```

**Characteristics:**
- Two rounds
- First provider critiques
- Others respond to criticism
- Good for validation

## Advanced Configuration

### Custom Session Manager

Implement custom storage:

```python
from quorum_mcp.session import SessionManager

class RedisSessionManager(SessionManager):
    def __init__(self, redis_url: str):
        super().__init__()
        self.redis = redis.from_url(redis_url)

    async def create_session(self, query: str, mode: str) -> Session:
        session = await super().create_session(query, mode)
        await self.redis.set(session.session_id, session.json())
        return session

    async def get_session(self, session_id: str) -> Session:
        data = await self.redis.get(session_id)
        return Session.parse_raw(data)
```

### Rate Limiting

Configure per-provider rate limits:

```python
from quorum_mcp.providers.base import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_minute=60,
    tokens_per_minute=100000,
    concurrent_requests=5,
)

provider = AnthropicProvider(rate_limit_config=rate_config)
```

### Retry Configuration

Configure retry behavior:

```python
from quorum_mcp.providers.base import RetryConfig

retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    retry_on_timeout=True,
    retry_on_rate_limit=True,
    retry_on_server_error=True,
)

provider = AnthropicProvider(retry_config=retry_config)
```

## Provider-Specific Settings

### Anthropic

```python
AnthropicProvider(
    api_key="...",
    model="claude-3-5-sonnet-20241022",  # default
)
```

### OpenAI

```python
OpenAIProvider(
    api_key="...",
    model="gpt-4o",  # default
)
```

### Google Gemini

```python
GeminiProvider(
    api_key="...",
    model="gemini-2.5-flash",  # default
)
```

### Ollama

```python
OllamaProvider(
    host="http://localhost:11434",  # default
    model="llama3.2",  # default
)
```

## Logging

Configure logging level:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# Or for more detail
logging.getLogger("quorum_mcp").setLevel(logging.DEBUG)
```

## Cost Optimization

### Use Cheaper Models

```python
providers = [
    GeminiProvider(model="gemini-2.5-flash"),  # $0.15/1M input
    NovitaProvider(),                           # $0.04/1M input
    OllamaProvider(),                           # Free!
]
```

### Limit Token Usage

```python
session = await orchestrator.execute_quorum(
    query="Brief answer: What is Python?",
    max_tokens=256,  # Shorter responses
)
```

### Use Quick Consensus

```python
# 1 round instead of 3
mode="quick_consensus"
```

## Security Best Practices

1. **Never commit API keys** — Use environment variables
2. **Use `.env` files** — With `.gitignore`
3. **Rotate keys regularly** — Especially if exposed
4. **Set billing alerts** — In each provider's dashboard
5. **Use Ollama for sensitive data** — 100% local processing
