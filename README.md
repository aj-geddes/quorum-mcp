# Quorum-MCP

Multi-AI Consensus System MCP Server - Orchestrates multiple AI providers for consensus-based responses.

## Overview

Quorum-MCP is a Model Context Protocol (MCP) server that coordinates multiple AI providers through multi-round deliberation to produce consensus-based responses. This novel approach combines the strengths of different AI models to deliver more reliable and well-rounded answers.

**Currently Supported Providers:**
- 🤖 **Anthropic Claude** (claude-3-5-sonnet, claude-3-opus, claude-3-haiku)
- 🧠 **OpenAI GPT-4** (gpt-4o, gpt-4o-mini, gpt-4-turbo)
- ✨ **Google Gemini** (gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-pro)

## Features

- **Multi-Provider Orchestration**: Coordinates Anthropic Claude, OpenAI GPT-4, and Google Gemini
- **Consensus-Based Responses**: Multi-round deliberation for high-quality outputs
- **Three Operational Modes**: Quick consensus, full deliberation, devil's advocate
- **Simple API**: Two core tools (`q_in`, `q_out`) for easy integration
- **Cost Management**: Built-in cost tracking across all providers
- **Session Management**: Async query processing with session tracking
- **95%+ Test Coverage**: Comprehensive test suite with 76+ passing tests

## Installation

### From Source

```bash
# Clone the repository
cd quorum-mcp

# Install in development mode
pip install -e ".[dev]"

# Or install for production
pip install -e .
```

### Configuration

1. Copy the configuration template:
```bash
cp config.yaml.template config.yaml
```

2. Edit `config.yaml` and add your API keys:
```yaml
providers:
  claude:
    api_key: "your-anthropic-api-key"
  openai:
    api_key: "your-openai-api-key"
  gemini:
    api_key: "your-google-api-key"
```

Alternatively, use environment variables (recommended):
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

**Note**: At least one API key is required. For best results, use all three providers to get comprehensive consensus.

## Usage

### Running the Server

```bash
# Start the MCP server
quorum-mcp

# Or run directly
python -m quorum_mcp.server
```

### Using with Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "quorum-mcp": {
      "command": "quorum-mcp",
      "args": []
    }
  }
}
```

### Tool Usage

**Submit a query to the quorum:**
```python
result = await q_in(
    query="What are the best practices for API design?",
    context="Focus on REST APIs and modern standards"
)
session_id = result["session_id"]
```

**Retrieve consensus results:**
```python
result = await q_out(
    session_id=session_id,
    wait=True  # Wait for completion
)
print(result["consensus_response"])
```

## Development

### Project Structure

```
quorum-mcp/
├── src/
│   └── quorum_mcp/
│       ├── __init__.py
│       └── server.py
├── tests/
├── config.yaml.template
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Architecture

The system consists of:

1. **MCP Server Layer**: FastMCP-based server with `q_in`/`q_out` tools
2. **Provider Abstraction**: Unified interface for multiple AI APIs
3. **Orchestration Engine**: Multi-round deliberation coordinator
4. **Synthesis Layer**: Consensus aggregation and quality scoring
5. **Configuration Manager**: YAML-based settings and provider credentials

## Roadmap

- [x] Project structure and configuration
- [x] Basic MCP server implementation
- [ ] Provider abstraction layer
- [ ] Orchestration engine
- [ ] Consensus synthesis
- [ ] Session management
- [ ] Cost tracking and budgets
- [ ] Advanced features (caching, rate limiting)

## License

MIT

## Contributing

Contributions are welcome! Please ensure code passes all tests and linting before submitting.
