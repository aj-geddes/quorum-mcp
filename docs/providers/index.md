---
layout: page
title: Providers
description: Overview of all AI providers supported by Quorum-MCP
permalink: /providers/
---

Quorum-MCP supports 7 AI providers out of the box. Mix and match to create your ideal consensus panel based on your needs, budget, and privacy requirements.

## Provider Comparison

| Provider | Input Cost | Output Cost | Context | Privacy | Speed |
|----------|-----------|-------------|---------|---------|-------|
| **Ollama** | $0.00 | $0.00 | 128K | 100% Local | Fast |
| **Novita AI** | $0.04/1M | $0.20/1M | 128K | Cloud | Fast |
| **Mistral AI** | $0.04/1M | $0.12/1M | 128K | EU Cloud | Fast |
| **Google Gemini** | $0.15/1M | $0.60/1M | 2M | Cloud | Very Fast |
| **Cohere** | $0.30/1M | $1.50/1M | 128K | Cloud | Fast |
| **OpenAI** | $2.50/1M | $10.00/1M | 128K | Cloud | Fast |
| **Anthropic** | $3.00/1M | $15.00/1M | 200K | Cloud | Medium |

*Costs shown for default models. Premium models cost more.*

## Typical Consensus Costs

For a typical query (500 tokens in, 300 tokens out) with 3 providers:

- **Quick Consensus:** ~$0.01 - $0.02
- **Full Deliberation (3 rounds):** ~$0.03 - $0.06
- **With Ollama:** $0.00 (fully local)

## Cloud Providers

### [Anthropic Claude](/providers/anthropic/)
Thoughtful, nuanced reasoning with excellent instruction following. Known for safety and helpfulness.
- **Best for:** Complex reasoning, nuanced analysis, safety-critical applications
- **Models:** Claude 3.5 Sonnet (default), Claude 3 Opus, Claude 3 Haiku

### [OpenAI GPT-4](/providers/openai/)
Broad knowledge base with strong reasoning capabilities. Industry standard for many applications.
- **Best for:** General-purpose tasks, code generation, broad knowledge queries
- **Models:** GPT-4o (default), GPT-4o-mini, GPT-4 Turbo

### [Google Gemini](/providers/gemini/)
Fast and cost-effective with the largest context window available (2M tokens with Gemini 1.5 Pro).
- **Best for:** Long documents, cost-sensitive applications, speed-critical tasks
- **Models:** Gemini 2.5 Flash (default), Gemini 2.5 Pro, Gemini 1.5 Pro

### [Cohere](/providers/cohere/)
Enterprise-focused with excellent RAG capabilities. Production-ready with free tier available.
- **Best for:** Enterprise applications, RAG systems, semantic search
- **Models:** Command R+ (default), Command R, Command Light

### [Mistral AI](/providers/mistral/)
European AI with competitive pricing and GDPR compliance. Excellent price-to-performance ratio.
- **Best for:** EU compliance, cost optimization, multilingual tasks
- **Models:** Mistral Large (default), Mistral Small, Mixtral 8x22B

### [Novita AI](/providers/novita/)
Ultra-low cost provider with OpenAI-compatible API. Access to various open-source models.
- **Best for:** Budget-constrained applications, experimentation, high-volume queries
- **Models:** Llama 3.3 70B (default), DeepSeek R1, Qwen 2.5 72B

## Local Provider

### [Ollama](/providers/ollama/)
Run models locally for zero-cost, 100% private inference. Data never leaves your machine.
- **Best for:** Privacy-sensitive applications, offline use, cost elimination
- **Models:** Llama 3.2 (default), Mistral, Mixtral, Qwen3, DeepSeek R1, Gemma3

## Choosing Providers

### For Best Quality
Combine high-end models from different providers:
- Anthropic Claude 3.5 Sonnet
- OpenAI GPT-4o
- Google Gemini 2.5 Pro

### For Best Price
Use cost-effective options:
- Ollama (free, local)
- Novita AI (ultra-low cost)
- Mistral AI (great value)
- Google Gemini Flash (fast and cheap)

### For Maximum Privacy
Use only local inference:
- Ollama with Llama 3.2, Mistral, or Qwen3

### For Enterprise
Focus on reliability and compliance:
- Anthropic Claude (safety-focused)
- Cohere (enterprise RAG)
- Mistral AI (EU/GDPR compliant)

## Adding Custom Providers

Quorum-MCP's provider system is extensible. See the [Architecture](/architecture/) page for details on implementing custom providers.

```python
from quorum_mcp.providers.base import Provider

class MyCustomProvider(Provider):
    async def send_request(self, request):
        # Your implementation
        pass

    async def count_tokens(self, text):
        # Your implementation
        pass

    def get_cost(self, tokens_input, tokens_output):
        # Your implementation
        pass

    def get_provider_name(self):
        return "my_custom_provider"

    def get_model_info(self):
        return {"name": "custom-model", "context_window": 8192}
```
