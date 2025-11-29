# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quorum-MCP is a Multi-AI Consensus System that orchestrates multiple AI providers (Anthropic, OpenAI, Google Gemini, Cohere, Mistral, Novita, Ollama) through multi-round deliberation to produce consensus-based responses. It's implemented as an MCP (Model Context Protocol) server for integration with Claude Desktop and other MCP clients.

## Build and Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run the MCP server
quorum-mcp
# or: python -m quorum_mcp.server

# Run the web dashboard
quorum-web
# or: python -m quorum_mcp.web_server

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_gemini_provider.py -v

# Run fast tests only (exclude slow integration tests)
pytest -m "not slow"

# Code quality
pre-commit run --all-files

# Individual tools
black src/ tests/
ruff check src/ tests/ --fix
mypy src/
```

## Architecture

### Core Components

- **`server.py`**: FastMCP server exposing two tools (`q_in` for query submission, `q_out` for result retrieval) via stdio transport
- **`orchestrator.py`**: Coordinates multi-provider consensus through three operational modes:
  - `quick_consensus`: Single round, parallel independent responses
  - `full_deliberation`: Three rounds (independent analysis → cross-review → synthesis)
  - `devils_advocate`: One provider critiques, others respond
- **`session.py`**: Pydantic-based session state management tracking queries through deliberation lifecycle
- **`providers/base.py`**: Abstract `Provider` base class defining interface for all AI integrations (send_request, count_tokens, get_cost, check_health)

### Provider System

All providers inherit from `Provider` ABC in `providers/base.py`. Key abstractions:
- `ProviderRequest`/`ProviderResponse`: Standardized request/response models
- `ProviderError` hierarchy: Authentication, RateLimit, Timeout, Connection, InvalidRequest, Model, QuotaExceeded
- Built-in rate limiting, retry logic, health checks, and cost tracking

Providers: `AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`, `CohereProvider`, `MistralProvider`, `NovitaProvider`, `OllamaProvider`, `OpenAICompatibleProvider`

### Request Flow

1. MCP client calls `q_in(query, context, mode)`
2. Server creates/retrieves Session via SessionManager
3. Orchestrator filters healthy providers, executes rounds based on mode
4. Providers queried in parallel with retry logic
5. Consensus built from responses (agreement detection, confidence scoring, synthesis)
6. Results returned with session_id for later retrieval via `q_out`

## Code Patterns

- **Async-first**: All provider calls, session operations use async/await
- **Pydantic models**: Request/Response/Session validation with `ConfigDict`
- **Error mapping**: Each provider maps native errors to `ProviderError` subtypes
- **Token/cost tracking**: Providers implement `count_tokens()` and `get_cost()` for usage tracking

## Adding a New Provider

1. Create `providers/<name>_provider.py` inheriting from `Provider`
2. Implement: `send_request()`, `count_tokens()`, `get_cost()`, `get_provider_name()`, `get_model_info()`
3. Map provider-specific errors to `ProviderError` subclasses
4. Add tests in `tests/test_<name>_provider.py`
5. Export from `providers/__init__.py`
6. Add initialization in `server.py` `initialize_server()`

## Configuration

Environment variables (at least one provider required):
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- `COHERE_API_KEY`, `MISTRAL_API_KEY`, `NOVITA_API_KEY`
- `OLLAMA_ENABLE` (default: true), `OLLAMA_HOST` (default: http://localhost:11434)

## Testing

- pytest with asyncio_mode="auto" (async tests run automatically)
- Coverage enabled by default (--cov=quorum_mcp)
- Use `@pytest.mark.slow` for integration tests requiring live APIs
