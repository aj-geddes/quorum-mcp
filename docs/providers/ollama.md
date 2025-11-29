---
layout: page
title: Ollama (Local LLMs)
description: Configure and use Ollama for zero-cost, private local inference with Quorum-MCP
permalink: /providers/ollama/
---

Ollama enables you to run large language models locally on your own hardware. This means **zero cost** and **100% privacy**—your data never leaves your machine.

## Quick Setup

```bash
# 1. Install Ollama from https://ollama.com/download

# 2. Start the Ollama server
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2

# 4. Quorum-MCP will automatically detect Ollama
```

## Available Models

| Model | Parameters | RAM Required | Best For |
|-------|------------|--------------|----------|
| `llama3.2` | 3B | 4GB | Default, fast |
| `llama3.1` | 8B | 8GB | Balanced |
| `llama3.1:70b` | 70B | 64GB | Highest quality |
| `mistral` | 7B | 6GB | General purpose |
| `mixtral` | 8x7B | 32GB | Expert mix |
| `qwen3` | 7B | 6GB | Multilingual |
| `deepseek-r1` | Various | Varies | Reasoning |
| `gemma3` | 9B | 8GB | Google's open model |

## Configuration

### Basic Usage

```python
from quorum_mcp.providers import OllamaProvider

# Uses default model (llama3.2)
provider = OllamaProvider()

# Specify a different model
provider = OllamaProvider(model="mistral")

# Custom server URL
provider = OllamaProvider(host="http://192.168.1.100:11434")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_ENABLE` | `true` | Enable/disable Ollama |

## Checking Availability

```python
from quorum_mcp.providers import OllamaProvider

provider = OllamaProvider()
availability = await provider.check_availability()

print(f"Server running: {availability['server_running']}")
print(f"Model available: {availability['model_available']}")
print(f"Available models: {availability['available_models']}")
```

## Strengths

- **Zero Cost:** No API fees, ever
- **100% Privacy:** Data never leaves your machine
- **No Internet Required:** Works completely offline
- **No Rate Limits:** Use as much as your hardware allows
- **Model Control:** Choose exactly which model to run

## Considerations

- Requires decent hardware (especially for larger models)
- Quality may not match top commercial models
- Initial model download can be large
- Inference speed depends on your hardware

## Hardware Recommendations

| Use Case | RAM | GPU | Model Suggestion |
|----------|-----|-----|------------------|
| Light usage | 8GB | Optional | llama3.2 (3B) |
| General use | 16GB | 8GB VRAM | llama3.1 (8B) |
| Power user | 32GB | 16GB VRAM | mixtral (8x7B) |
| Best quality | 64GB+ | 24GB+ VRAM | llama3.1:70b |

## Example

```python
from quorum_mcp.providers import OllamaProvider
from quorum_mcp.providers.base import ProviderRequest

async def query_local():
    provider = OllamaProvider()

    # Check if model is available
    availability = await provider.check_availability()
    if not availability['model_available']:
        print(f"Model not found. Run: ollama pull {provider.model}")
        return

    request = ProviderRequest(
        query="Write a poem about open source software",
        max_tokens=512,
        temperature=0.8
    )

    response = await provider.send_request(request)

    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
    print(f"Cost: ${response.cost:.4f}")  # Always $0.00!
```

## Running Multiple Models

```bash
# Pull multiple models for variety
ollama pull llama3.2
ollama pull mistral
ollama pull qwen3

# Use them in Quorum-MCP
providers = [
    OllamaProvider(model="llama3.2"),
    OllamaProvider(model="mistral"),
    OllamaProvider(model="qwen3"),
]
```

## Private Consensus

For maximum privacy, use only Ollama providers:

```python
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import OllamaProvider

# All local, all private, all free
providers = [
    OllamaProvider(model="llama3.2"),
    OllamaProvider(model="mistral"),
    OllamaProvider(model="qwen3"),
]

orchestrator = Orchestrator(providers=providers)

# Your data never leaves your machine
session = await orchestrator.execute_quorum(
    query="Analyze this confidential document...",
    mode="full_deliberation"
)
```

## Troubleshooting

### Ollama server not running

```bash
# Start the server
ollama serve

# Or check if it's running
curl http://localhost:11434/api/tags
```

### Model not found

```bash
# List available models
ollama list

# Pull the model you need
ollama pull llama3.2
```

### Out of memory

- Try a smaller model (3B or 7B parameters)
- Close other applications
- Consider upgrading RAM or using swap

### Slow inference

- Use a smaller model
- If you have a GPU, ensure Ollama is using it
- Check `ollama ps` for current model status

## Best Practices

1. **Start with smaller models** — Ensure they work before trying larger ones
2. **Match model to hardware** — Don't try to run 70B on 8GB RAM
3. **Keep Ollama running** — Startup time is longer than inference
4. **Pre-pull models** — Download before you need them
5. **Mix with cloud** — Combine local and cloud providers for best results
