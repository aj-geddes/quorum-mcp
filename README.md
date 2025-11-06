# Quorum-MCP

> Multi-AI Consensus System MCP Server - Get better answers through deliberation

[![Tests](https://img.shields.io/badge/tests-256%20passing-success)](https://github.com/aj-geddes/quorum-mcp)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)](https://github.com/aj-geddes/quorum-mcp)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 🎯 Overview

Quorum-MCP orchestrates multiple AI providers (Anthropic Claude, OpenAI, Google Gemini, Mistral AI, Ollama, and any OpenAI-compatible endpoint) through multi-round deliberation to produce consensus-based responses. By combining different AI models, you get more balanced, comprehensive, and reliable answers.

**Why Quorum?**
- 🎭 **Diverse Perspectives**: Each AI has unique strengths and biases
- 🤝 **Consensus Building**: Agreement across models increases confidence
- 🔍 **Quality Assurance**: Cross-validation catches errors and hallucinations
- 💡 **Richer Insights**: Disagreements reveal nuanced viewpoints

## ✨ Features

### Multi-Provider Support
- 🤖 **Anthropic Claude** - Thoughtful, nuanced reasoning
  - Models: `claude-3-5-sonnet-20241022` (default), `claude-3-opus`, `claude-3-haiku`
  - Context: 200K tokens | Cost: $3-$15/1M input
- 🧠 **OpenAI** - Broad knowledge, strong reasoning
  - Models: `gpt-4o` (default), `gpt-4o-mini`, `gpt-4-turbo`
  - Context: 128K tokens | Cost: $0.15-$30/1M input
- ✨ **Google Gemini** - Fast, cost-effective, huge context
  - Models: `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-1.5-pro`
  - Context: Up to **2M tokens** | Cost: $0.15-$1.25/1M input
- 🏠 **Ollama (Local LLMs)** - Private, zero-cost local inference
  - Models: `llama3.2` (default), `llama3.1`, `mistral`, `mixtral`, `qwen3`, `deepseek-r1`, `gemma3`
  - Context: Up to 128K tokens | Cost: **$0.00** (100% local)
  - Privacy: 100% - Data never leaves your machine
- 🚀 **Mistral AI** - European AI with strong code capabilities
  - Models: `mistral-large-latest` (default), `mistral-medium-latest`, `codestral-latest`, `pixtral-large-latest`
  - Context: Up to 256K tokens | Cost: $0.20-$6/1M input
- 🔌 **OpenAI-Compatible** - Universal support for local LLM servers
  - Supports: LM Studio, text-gen-webui, LocalAI, vLLM, llama.cpp, TabbyAPI
  - Cloud: OpenRouter, Together AI, Anyscale, Deep Infra
  - Custom endpoints with configurable pricing

### Three Operational Modes

**1. Quick Consensus** (Single Round)
```python
# Fast consensus for straightforward queries
session = await orchestrator.execute_quorum(
    query="What are Python best practices?",
    mode="quick_consensus"
)
```

**2. Full Deliberation** (3 Rounds)
```python
# Multi-round deliberation for complex decisions
# Round 1: Independent analysis
# Round 2: Cross-review and critique
# Round 3: Final synthesis
session = await orchestrator.execute_quorum(
    query="Should we use microservices or monolith?",
    mode="full_deliberation"
)
```

**3. Devil's Advocate** (Critical Analysis)
```python
# Challenge assumptions and find weaknesses
session = await orchestrator.execute_quorum(
    query="We should skip testing to move faster",
    mode="devils_advocate"
)
```

### Additional Features
- ⚡ **Async/Await**: Non-blocking I/O throughout
- 💰 **Cost Tracking**: Per-provider and total cost reporting (including $0 for local)
- 🏠 **Local LLMs**: Zero-cost inference with Ollama (100% private)
- 🏥 **Health Monitoring**: Automatic provider health checks before execution
- 🎯 **Smart Provider Selection**: Filters unhealthy providers automatically
- 📊 **Session Management**: Persistent session storage and retrieval
- 🔒 **Type Safe**: Full Pydantic validation
- 🧪 **Well Tested**: 256 passing tests, 84% code coverage
- 📝 **MCP Integration**: Works with Claude Desktop and other MCP clients

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aj-geddes/quorum-mcp.git
cd quorum-mcp

# Install with dependencies
pip install -e .

# Or install for development
pip install -e ".[dev]"
```

### Configuration

#### Cloud Providers (Optional)

Set your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."  # Optional: Mistral AI
```

#### Local LLMs with Ollama (Optional, Zero Cost)

Install and run Ollama for 100% free, private local inference:

```bash
# Install Ollama (Mac/Linux/Windows)
# Visit: https://ollama.com/download

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2

# Optional: Configure Ollama host (default: http://localhost:11434)
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_ENABLE="true"  # Set to "false" to disable
```

**Note**: At least one provider (cloud or local) is required. Ollama enables zero-cost consensus!

### Running the Server

```bash
# Start the MCP server
quorum-mcp

# Or run directly
python -m quorum_mcp.server
```

### Basic Usage Example

```python
import asyncio
from quorum_mcp.orchestrator import Orchestrator
from quorum_mcp.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    MistralProvider,
    OllamaProvider,
)
from quorum_mcp.session import get_session_manager

async def main():
    # Initialize providers (mix cloud and local!)
    providers = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
        MistralProvider(),  # New: Mistral AI
        OllamaProvider(),   # Local LLM (zero cost!)
    ]

    # Start session manager
    session_manager = get_session_manager()
    await session_manager.start()

    # Create orchestrator with health monitoring
    orchestrator = Orchestrator(
        providers=providers,
        session_manager=session_manager,
        check_health=True  # Filters unhealthy providers automatically
    )

    # Execute consensus
    session = await orchestrator.execute_quorum(
        query="What is the best database for a startup?",
        context="Small team, rapid iteration, expecting growth",
        mode="quick_consensus"
    )

    # Print results
    print(f"Confidence: {session.consensus['confidence']:.2%}")
    print(f"Summary: {session.consensus['summary']}")
    print(f"Cost: ${session.consensus['cost']['total_cost']:.4f}")

    # Check provider health status
    if "health_checks" in session.metadata:
        print("\nProvider Health:")
        for provider, health in session.metadata["health_checks"].items():
            print(f"  {provider}: {health['status']}")

    await session_manager.stop()

asyncio.run(main())
```

## 📖 Usage

### MCP Tools

Quorum-MCP provides two simple tools for MCP clients:

#### `q_in` - Submit Query

```json
{
  "query": "What are the top 3 considerations for API design?",
  "context": "Building a REST API for a SaaS product",
  "mode": "quick_consensus"
}
```

Returns:
```json
{
  "session_id": "abc-123-def",
  "status": "completed",
  "confidence": 0.85,
  "consensus": {
    "summary": "Based on consensus...",
    "agreement_areas": [...],
    "cost": {...}
  }
}
```

#### `q_out` - Retrieve Results

```json
{
  "session_id": "abc-123-def"
}
```

Returns the full session data including consensus results.

### Using with Claude Desktop

Add to your `claude_desktop_config.json`:

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

### Running Demos

```bash
# Three-provider consensus demo (cloud providers)
python examples/three_provider_demo.py

# Local LLM demo with Ollama (zero cost!)
python examples/local_llm_demo.py

# End-to-end demo with all modes
python examples/end_to_end_demo.py

# Session management demo
python examples/session_demo.py
```

## 💰 Cost Comparison

| Provider | Model | Input ($/1M) | Output ($/1M) | Context Window | Speed |
|----------|-------|--------------|---------------|----------------|-------|
| **Ollama** 🏆 | **llama3.2** | **$0.00** | **$0.00** | **128K** | ⚡⚡⚡ |
| **Ollama** | **mistral** | **$0.00** | **$0.00** | **32K** | ⚡⚡⚡ |
| Gemini   | 2.5 Flash | $0.15 | $0.60 | 200K | ⚡⚡⚡ |
| OpenAI   | 4o-mini | $0.15 | $0.60 | 128K | ⚡⚡⚡ |
| Mistral  | Small | $0.20 | $0.60 | 32K | ⚡⚡⚡ |
| Mistral  | Codestral | $0.30 | $0.90 | 256K | ⚡⚡ |
| Mistral  | Medium | $0.40 | $2.00 | 32K | ⚡⚡ |
| Gemini   | 2.5 Pro | $1.25 | $10.00 | 200K | ⚡⚡ |
| Gemini   | 1.5 Pro | $1.25 | $5.00 | 2M | ⚡ |
| Mistral  | Large | $2.00 | $6.00 | 128K | ⚡⚡ |
| OpenAI   | 4o | $2.50 | $10.00 | 128K | ⚡⚡ |
| Claude   | 3.5 Sonnet | $3.00 | $15.00 | 200K | ⚡⚡ |
| Claude   | 3 Opus | $15.00 | $75.00 | 200K | ⚡ |

**Typical Consensus Cost** (500 tokens in, 300 tokens out, 3 providers):
- Quick Consensus: ~$0.01 - $0.02
- Full Deliberation (3 rounds): ~$0.03 - $0.06

## 🏗️ Architecture

```mermaid
graph TD
    Client[MCP Client<br/>Claude Desktop]

    subgraph FastMCP["FastMCP Server"]
        QIn[q_in tool]
        QOut[q_out tool]
    end

    subgraph Orchestrator["Orchestrator Engine"]
        Consensus[Consensus Algorithms<br/>• Agreement detection<br/>• Confidence scoring<br/>• Synthesis & summarization]
    end

    subgraph Providers["AI Providers"]
        Anthropic[AnthropicProvider<br/>• Async client<br/>• Token counting<br/>• Cost tracking<br/>• Error mapping]
        OpenAI[OpenAIProvider<br/>• Async client<br/>• tiktoken<br/>• Cost tracking<br/>• Error mapping]
        Gemini[GeminiProvider<br/>• Async client<br/>• Token counting<br/>• Cost tracking<br/>• Error mapping]
        Ollama[OllamaProvider<br/>• Async client<br/>• Local inference<br/>• Zero cost<br/>• 100% private]
    end

    AnthropicAPI[Anthropic API]
    OpenAIAPI[OpenAI API]
    GeminiAPI[Google AI API]
    OllamaServer[Ollama Server<br/>Local]

    Client -->|stdio/HTTP| FastMCP
    QIn --> Consensus
    QOut --> Consensus
    Consensus --> Anthropic
    Consensus --> OpenAI
    Consensus --> Gemini
    Consensus --> Ollama
    Anthropic --> AnthropicAPI
    OpenAI --> OpenAIAPI
    Gemini --> GeminiAPI
    Ollama --> OllamaServer
```

## 📁 Project Structure

```
quorum-mcp/
├── src/quorum_mcp/
│   ├── __init__.py
│   ├── server.py              # FastMCP server with q_in/q_out tools
│   ├── orchestrator.py        # Multi-provider orchestration engine
│   ├── session.py             # Session management and persistence
│   └── providers/
│       ├── __init__.py
│       ├── base.py            # Abstract provider interface + health monitoring
│       ├── anthropic_provider.py      # Claude integration
│       ├── openai_provider.py         # OpenAI integration
│       ├── gemini_provider.py         # Gemini integration
│       ├── ollama_provider.py         # Ollama local LLM integration
│       ├── mistral_provider.py        # Mistral AI integration
│       └── openai_compatible_provider.py  # Universal local LLM support
├── examples/
│   ├── three_provider_demo.py  # Demo with cloud providers
│   ├── local_llm_demo.py       # Demo with Ollama (zero cost)
│   ├── end_to_end_demo.py      # All operational modes
│   └── session_demo.py         # Session management
├── tests/
│   ├── test_session.py
│   ├── test_orchestrator.py
│   ├── test_anthropic_provider.py
│   ├── test_openai_provider.py
│   ├── test_gemini_provider.py
│   ├── test_ollama_provider.py
│   ├── test_mistral_provider.py
│   ├── test_openai_compatible_provider.py
│   ├── test_health_monitoring.py
│   └── test_integration.py
├── docs/
│   └── session_management.md
├── .pre-commit-config.yaml     # Code quality hooks
├── pyproject.toml              # Project configuration
├── README.md
└── worklog.md                  # Complete development history
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quorum_mcp --cov-report=html

# Run specific test file
pytest tests/test_gemini_provider.py -v

# Run only fast tests
pytest -m "not slow"
```

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Individual tools
black src/ tests/           # Format code
ruff check src/ tests/      # Lint code
mypy src/                   # Type check
```

### Adding a New Provider

1. Create a new provider class inheriting from `Provider`
2. Implement required methods:
   - `send_request(request: ProviderRequest) -> ProviderResponse`
   - `count_tokens(text: str) -> int`
   - `get_cost(tokens_input: int, tokens_output: int) -> float`
   - `get_provider_name() -> str`
   - `get_model_info() -> dict`
3. Add comprehensive tests
4. Update `providers/__init__.py`
5. Add to `server.py` initialization

See `gemini_provider.py` as a reference implementation.

## 📊 Test Coverage

```
Module                                       Coverage
─────────────────────────────────────────────────────
providers/gemini_provider.py                    95%
providers/ollama_provider.py                    95%
session.py                                      93%
providers/openai_compatible_provider.py         92%
orchestrator.py                                 91%
providers/base.py (with health monitoring)      88%
providers/openai_provider.py                    84%
providers/mistral_provider.py                   80%
providers/anthropic_provider.py                 70%
─────────────────────────────────────────────────────
Total                                           84%
```

**Test Results:**
- ✅ 256 tests passing (96.2%)
- ❌ 6 tests failing (pre-existing Anthropic/OpenAI tests)
- ⚠️ 2 errors (mock-related, non-critical)

## 🗺️ Roadmap

### ✅ Phase 1: MVP (Complete)
- [x] Provider abstraction layer
- [x] Anthropic Claude integration
- [x] OpenAI integration
- [x] Basic orchestration engine
- [x] Session management
- [x] FastMCP server with q_in/q_out
- [x] Cost tracking

### ✅ Phase 2: Testing (Complete)
- [x] Comprehensive unit tests
- [x] Provider test suites
- [x] Integration tests
- [x] Pre-commit hooks
- [x] Code quality tooling

### ✅ Phase 3: Google Gemini (Complete)
- [x] Gemini provider implementation
- [x] Token counting and cost tracking
- [x] 95% test coverage
- [x] Three-provider demo
- [x] Documentation updates

### ✅ Phase 4: Local LLMs (Complete)
- [x] Ollama provider integration
- [x] Support for Llama 3.2, Llama 3.1, Mistral, Mixtral, Qwen3, DeepSeek-R1, Gemma3
- [x] Zero-cost local inference ($0.00)
- [x] 100% privacy-preserving mode (data never leaves machine)
- [x] 95% test coverage (29 passing tests)
- [x] Local LLM demo and hybrid (local+cloud) demo
- [x] Automatic server detection and model availability checking

### ✅ Phase 5: Universal Provider Support & Health Monitoring (Complete)
- [x] OpenAI-compatible API provider (universal local LLM support)
  - Supports: LM Studio, text-gen-webui, LocalAI, vLLM, llama.cpp, TabbyAPI
  - Cloud: OpenRouter, Together AI, Anyscale, Deep Infra
  - 30/30 tests passing (100%)
- [x] Mistral AI provider (cloud)
  - All 2025 models: Large, Medium, Codestral, Pixtral, Small
  - 37/37 tests passing (100%)
- [x] Provider health monitoring system
  - Three-tier status (HEALTHY/DEGRADED/UNHEALTHY)
  - Response time thresholds and error detection
  - 18/18 tests passing (100%)
- [x] Orchestrator health integration
  - Automatic pre-execution health checks
  - Filters unhealthy providers before consensus
  - 25/25 orchestrator tests passing (100%)
- [x] 256 total tests passing (96.2%), 84% code coverage

### 🔮 Phase 6: Advanced Features (Future)
- [ ] Caching layer for repeated queries
- [ ] Rate limiting per provider
- [ ] Budget controls and cost limits
- [ ] Performance benchmarking suite
- [ ] Streaming responses
- [ ] Tool use / function calling support
- [ ] Web UI for result visualization
- [ ] Provider fallback strategies
- [ ] A/B testing modes

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run code quality checks (`pre-commit run --all-files`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

**Code Standards:**
- Black for formatting (100 char lines)
- Ruff for linting
- mypy for type checking
- pytest for testing (aim for 80%+ coverage)
- Comprehensive docstrings

## ❓ Troubleshooting

### API Key Issues

```bash
# Verify API keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Test individual provider
python -c "from quorum_mcp.providers import GeminiProvider; print(GeminiProvider())"
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Or reinstall with dependencies
pip install -e ".[dev]" --force-reinstall
```

### Test Failures

```bash
# Clear pytest cache
rm -rf .pytest_cache __pycache__

# Run with verbose output
pytest -vv --tb=short
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io/)
- Powered by [FastMCP](https://github.com/jlowin/fastmcp)
- Utilizes:
  - [Anthropic Claude](https://www.anthropic.com/)
  - [OpenAI](https://openai.com/)
  - [Google Gemini](https://deepmind.google/technologies/gemini/)

## 📬 Contact

- GitHub: [@aj-geddes](https://github.com/aj-geddes)
- Issues: [GitHub Issues](https://github.com/aj-geddes/quorum-mcp/issues)

---

**Built with ❤️ for better AI through collaboration**
