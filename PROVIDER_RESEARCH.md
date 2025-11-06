# Provider Research: Additional AI Provider Support

**Status**: Research Complete
**Date**: 2025-11-06
**Purpose**: Research additional AI providers beyond Anthropic and OpenAI for Quorum-MCP

---

## Priority 1: Currently Implemented

### ✅ Anthropic Claude
- **SDK**: `anthropic` Python package
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Context Window**: 200K tokens
- **Pricing**: $0.25-$15/1M input, $1.25-$75/1M output
- **Status**: **Implemented** ✅

### ✅ OpenAI
- **SDK**: `openai` Python package (v1.0+)
- **Models**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo
- **Context Window**: 8K-128K tokens
- **Pricing**: $0.15-$30/1M input, $0.60-$60/1M output
- **Updated**: GPT-4o support added (Nov 2024) ✅
- **Status**: **Implemented** ✅

---

## Priority 2: Google Gemini (Next Sprint)

### Google Gemini API
- **SDK**: `google-generativeai` (Python 3.9+)
- **API Key**: From Google AI Studio
- **Free Tier**: Available via Google AI Studio
- **Paid Tier**: Vertex AI for enterprise

### Models Available

**Gemini 2.5 Pro** (Latest):
- Context Window: 200K+ tokens
- Pricing: $1.25/1M input, $10.00/1M output
- With reasoning: Higher cost for complex reasoning

**Gemini 2.5 Flash** (Efficient):
- Context Window: 200K+ tokens
- Pricing: $0.15/1M input, $0.60/1M output
- With reasoning: $0.015 input + $0.06 output per 100K

**Gemini 1.5 Pro** (Stable):
- Context Window: 2M tokens (largest available)
- Pricing: 64% reduction in 2024
- Prompts <128K: Significantly cheaper

**Gemini 1.5 Flash**:
- Faster, cheaper alternative
- Multimodal support (text, image, audio, video)

### Integration Plan
```python
from google import generativeai as genai

# SDK Usage
genai.configure(api_key="GOOGLE_API_KEY")
model = genai.GenerativeModel('gemini-2.5-pro')
response = model.generate_content("Prompt here")
```

### Implementation Notes
- API key from `GOOGLE_API_KEY` environment variable
- Async support via `google-generativeai`
- Multimodal capabilities (can extend for image/video input)
- Largest context window (2M tokens for 1.5 Pro)
- Competitive pricing especially for flash models

---

## Priority 3: Local LLM Support (Sprint 3)

### Ollama (Primary Local LLM Interface)

**What is Ollama?**
- Open-source platform for running LLMs locally
- REST API on localhost:11434
- No API costs, runs on local hardware
- Python SDK available

**Python Integration Options:**

1. **Official Ollama Python Library** (Recommended):
```python
import ollama
response = ollama.generate(
    model='llama2',
    prompt='What is a qubit?'
)
```

2. **REST API Direct**:
```python
import requests
response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'llama2',
    'prompt': 'Tell me about Python'
})
```

3. **LangChain Integration**:
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
response = llm.invoke("Tell me about Python")
```

**Available Models**:
- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Mistral (7B)
- Mixtral (8x7B)
- Phi-2, Phi-3
- Gemma
- CodeLlama
- Vicuna
- Custom models via Modelfile

**Features**:
- Streaming responses
- Model management (pull, push, list, delete)
- Embeddings support
- Chat and completion modes
- No API key required
- Zero cost (local compute only)

**Implementation Strategy**:
```python
class OllamaProvider(Provider):
    def __init__(self, base_url="http://localhost:11434", model="llama2"):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=base_url)

    async def send_request(self, request: ProviderRequest):
        # Use ollama.generate() or ollama.chat()
        # No API key needed
        # Cost = 0.0
        pass
```

**Advantages**:
- Privacy: Data never leaves local machine
- Cost: Zero API costs
- Speed: Low latency (local network)
- Availability: No internet dependency
- Customization: Can fine-tune models

**Challenges**:
- Requires local compute resources (GPU recommended)
- Model quality may be lower than hosted APIs
- Setup complexity for users
- Token counting needs different approach

---

### Open WebUI (Alternative Local Interface)

**What is Open WebUI?**
- Self-hosted web interface for LLMs
- Supports both Ollama and OpenAI-compatible APIs
- Docker/Kubernetes deployment
- Built-in RAG (Retrieval Augmented Generation)

**Integration Approach**:
- Open WebUI uses Ollama as backend
- Can be treated as Ollama provider
- Focus on Ollama integration covers Open WebUI

**Key Features**:
- Pipelines Framework for custom integrations
- Native Python function calling
- OpenAI API compatibility
- Easy Docker setup
- RAG support out of the box

**When to Use**:
- User wants local chat interface
- Need RAG capabilities
- Custom pipeline requirements
- OpenAI API compatibility needed

---

## Priority 4: Additional Providers (Future)

### Mistral AI
- **Models**: Mistral Large, Mistral Medium, Mistral Small
- **API**: Similar to OpenAI (chat completions)
- **SDK**: `mistralai` Python package
- **Pricing**: Competitive European alternative
- **Status**: Mentioned in plan, defer to Sprint 3+

### Cohere
- **Models**: Command, Command Light, Command R
- **SDK**: `cohere` Python package
- **Strengths**: Enterprise features, RAG
- **Status**: Future consideration

### Groq (Ultra-fast Inference)
- **Feature**: Lightning-fast inference on LPU hardware
- **Models**: Llama, Mistral models
- **SDK**: OpenAI-compatible API
- **Status**: Future consideration

### Together AI
- **Feature**: Access to 100+ open-source models
- **SDK**: OpenAI-compatible API
- **Status**: Future consideration

### Hugging Face Inference API
- **Feature**: Access to thousands of models
- **SDK**: `huggingface_hub` Python package
- **Status**: Future consideration

---

## Implementation Roadmap

### Sprint 2 (Current - MVP Complete)
- ✅ Anthropic Provider (Complete)
- ✅ OpenAI Provider (Complete + GPT-4o update)
- ✅ Basic Orchestration Engine
- ✅ Consensus Algorithm

### Sprint 3 (Enhanced Features)
- 🎯 **Google Gemini Provider** (Priority)
  - Implement GeminiProvider class
  - Support Gemini 2.5 Pro and Flash
  - Token counting and cost calculation
  - Error mapping

- 🎯 **Ollama Provider** (Local LLM Support)
  - Implement OllamaProvider class
  - Support local model discovery
  - Streaming support
  - Zero-cost tracking

### Sprint 4 (Scale)
- Mistral AI Provider
- Provider Factory/Registry
- Provider Health Monitoring
- Dynamic Provider Selection

### Future Sprints
- Additional providers as needed
- Custom provider plugin system
- Provider performance benchmarking

---

## Design Considerations

### Provider Abstraction Requirements

All providers must implement:
1. `send_request(request) -> response`
2. `count_tokens(text) -> int`
3. `get_cost(tokens_in, tokens_out) -> float`
4. `get_provider_name() -> str`
5. `get_model_info() -> dict`

### Challenges by Provider Type

**Hosted APIs (Anthropic, OpenAI, Gemini)**:
- ✅ Consistent API patterns
- ✅ Official SDKs available
- ✅ Clear pricing models
- ⚠️ Rate limiting required
- ⚠️ API key management
- ⚠️ Cost tracking essential

**Local LLMs (Ollama, Open WebUI)**:
- ✅ No API costs
- ✅ Privacy preserved
- ✅ Low latency
- ⚠️ Quality variation
- ⚠️ Setup complexity
- ⚠️ Resource requirements
- ⚠️ Different token counting approach
- ⚠️ Model availability checking

### Token Counting Strategies

**Hosted APIs**:
- Use official token counting APIs or libraries
- Anthropic: API-based counting
- OpenAI: tiktoken library
- Gemini: SDK token counting

**Local LLMs**:
- Use model-specific tokenizers
- Ollama: Estimate or use model tokenizer
- May need separate tokenizer libraries
- Less critical since no API costs

### Cost Tracking

**Hosted APIs**:
- Precise per-request cost calculation
- Track against budget limits
- Report cost per session
- Essential for user value proposition

**Local LLMs**:
- Cost = $0.00 (API)
- Could track compute costs if desired
- Focus on token counts for context limits
- Value proposition = privacy + cost savings

---

## Recommended Next Steps

1. **Immediate** (Current Sprint):
   - Complete orchestration engine ✅
   - Test with existing 2 providers ✅

2. **Sprint 3 Priorities**:
   - **Google Gemini**: 3rd major provider, fills gap
   - **Ollama**: Enable local/private deployments
   - Provider factory pattern

3. **Sprint 4+**:
   - Additional providers based on user demand
   - Provider performance monitoring
   - Health checks and failover

4. **Technical Debt**:
   - Provider plugin system (load external providers)
   - Provider benchmarking suite
   - Cost optimization algorithms

---

## Key Insights

1. **Gemini is high-priority**: Competitive pricing, huge context window (2M tokens), multimodal
2. **Ollama enables privacy**: Critical for users with sensitive data
3. **Cost varies dramatically**: $0 (Ollama) to $75/1M (Claude Opus output)
4. **Context windows matter**: 8K (GPT-4) to 2M (Gemini 1.5 Pro)
5. **API patterns converge**: Most use chat completion format
6. **Quality-cost tradeoffs**: Cheaper models may need more retries or multiple opinions

---

## Conclusion

**Current State**: 2 providers (Anthropic, OpenAI) fully implemented and production-ready.

**Next Priorities**:
1. Gemini (Sprint 3) - Major hosted provider gap
2. Ollama (Sprint 3) - Privacy and cost concerns
3. Additional providers (Sprint 4+) - As needed

The provider abstraction layer design supports easy addition of new providers. Each new provider requires ~400-600 lines of code following established patterns.
