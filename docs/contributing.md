---
layout: page
title: Contributing
description: Guidelines for contributing to Quorum-MCP
permalink: /contributing/
---

We welcome contributions to Quorum-MCP! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/quorum-mcp.git
cd quorum-mcp
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
pytest

# Run code quality checks
pre-commit run --all-files
```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write your code
2. Add tests for new functionality
3. Update documentation if needed
4. Run tests and linting

```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_your_feature.py -v

# Run code quality
pre-commit run --all-files
```

### Committing Changes

```bash
git add .
git commit -m "Add: description of your changes"
```

Commit message prefixes:
- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Enhancement to existing feature
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Test additions or changes

### Creating a Pull Request

1. Push your branch to your fork
2. Open a Pull Request on GitHub
3. Fill out the PR template
4. Wait for review

## Code Standards

### Python Style

We use these tools for code quality:

- **Black** — Code formatting (100 char lines)
- **Ruff** — Linting (replaces flake8, isort)
- **mypy** — Type checking
- **pytest** — Testing

```bash
# Format code
black src/ tests/

# Lint and fix
ruff check src/ tests/ --fix

# Type check
mypy src/

# Run all checks
pre-commit run --all-files
```

### Type Hints

All functions should have type hints:

```python
async def send_request(
    self,
    request: ProviderRequest,
    timeout: float | None = None
) -> ProviderResponse:
    """
    Send a request to the provider.

    Args:
        request: The request to send
        timeout: Optional timeout override

    Returns:
        The provider's response

    Raises:
        ProviderError: If the request fails
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_cost(tokens_input: int, tokens_output: int) -> float:
    """
    Calculate the cost of an API call.

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens

    Returns:
        Cost in USD

    Example:
        >>> calculate_cost(1000, 500)
        0.015
    """
    pass
```

## Testing

### Writing Tests

Tests live in the `tests/` directory and use pytest:

```python
import pytest
from quorum_mcp.providers import MyProvider
from quorum_mcp.providers.base import ProviderRequest

class TestMyProvider:
    """Tests for MyProvider."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        return MyProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_send_request_success(self, provider):
        """Test successful request."""
        request = ProviderRequest(query="Test query")
        response = await provider.send_request(request)

        assert response.content
        assert response.provider == "my_provider"

    @pytest.mark.asyncio
    async def test_send_request_auth_error(self, provider):
        """Test authentication error handling."""
        provider.api_key = "invalid"
        request = ProviderRequest(query="Test")

        with pytest.raises(ProviderAuthenticationError):
            await provider.send_request(request)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quorum_mcp --cov-report=html

# Run specific test
pytest tests/test_provider.py::TestMyProvider::test_send_request_success -v

# Run excluding slow tests
pytest -m "not slow"
```

### Mocking External APIs

Use `unittest.mock` for API calls:

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    with patch.object(MyProvider, '_call_api', new_callable=AsyncMock) as mock:
        mock.return_value = {"content": "Mocked response"}

        provider = MyProvider()
        response = await provider.send_request(ProviderRequest(query="Test"))

        assert response.content == "Mocked response"
        mock.assert_called_once()
```

## Adding a New Provider

### 1. Create the Provider Class

Create `src/quorum_mcp/providers/my_provider.py`:

```python
"""MyProvider integration for Quorum-MCP."""

import os
from typing import Any

from quorum_mcp.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    ProviderError,
    ProviderAuthenticationError,
)


class MyProvider(Provider):
    """Provider implementation for MyAPI."""

    # Pricing per 1M tokens
    PRICING = {
        "my-model": {"input": 1.00, "output": 2.00},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "my-model",
    ):
        self.api_key = api_key or os.getenv("MY_API_KEY")
        if not self.api_key:
            raise ProviderAuthenticationError("MY_API_KEY not set")

        self.model = model
        # Initialize your client here

    async def send_request(self, request: ProviderRequest) -> ProviderResponse:
        """Send request to MyAPI."""
        import time
        start_time = time.time()

        try:
            # Make API call
            response = await self._call_api(request)

            return ProviderResponse(
                content=response["text"],
                model=self.model,
                provider=self.get_provider_name(),
                tokens_input=response["usage"]["input_tokens"],
                tokens_output=response["usage"]["output_tokens"],
                cost=self.get_cost(
                    response["usage"]["input_tokens"],
                    response["usage"]["output_tokens"]
                ),
                latency=time.time() - start_time,
            )

        except AuthError as e:
            raise ProviderAuthenticationError(str(e), provider=self.get_provider_name())
        except Exception as e:
            raise ProviderError(str(e), provider=self.get_provider_name())

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Use provider's tokenizer or estimate
        return len(text) // 4

    def get_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Calculate cost in USD."""
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (
            (tokens_input / 1_000_000) * pricing["input"] +
            (tokens_output / 1_000_000) * pricing["output"]
        )

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "my_provider"

    def get_model_info(self) -> dict[str, Any]:
        """Return model information."""
        return {
            "name": self.model,
            "context_window": 128000,
            "provider": self.get_provider_name(),
        }
```

### 2. Export from `__init__.py`

Add to `src/quorum_mcp/providers/__init__.py`:

```python
from quorum_mcp.providers.my_provider import MyProvider

__all__ = [
    # ... existing exports
    "MyProvider",
]
```

### 3. Add to Server Initialization

Update `src/quorum_mcp/server.py`:

```python
from quorum_mcp.providers import MyProvider

# In initialize_server():
if os.getenv("MY_API_KEY"):
    try:
        my_provider = MyProvider()
        providers.append(my_provider)
        logger.info("MyProvider initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize MyProvider: {e}")
```

### 4. Write Tests

Create `tests/test_my_provider.py`:

```python
import pytest
from quorum_mcp.providers import MyProvider
from quorum_mcp.providers.base import ProviderRequest, ProviderAuthenticationError


class TestMyProvider:
    @pytest.fixture
    def provider(self):
        return MyProvider(api_key="test-key")

    def test_initialization(self, provider):
        assert provider.model == "my-model"
        assert provider.get_provider_name() == "my_provider"

    def test_missing_api_key(self):
        with pytest.raises(ProviderAuthenticationError):
            MyProvider(api_key=None)

    @pytest.mark.asyncio
    async def test_count_tokens(self, provider):
        tokens = await provider.count_tokens("Hello world")
        assert tokens > 0

    def test_get_cost(self, provider):
        cost = provider.get_cost(1000, 500)
        assert cost > 0
```

### 5. Update Documentation

Add provider documentation page and update the providers overview.

## Documentation

Documentation lives in the `docs/` directory and uses Jekyll.

### Local Preview

```bash
cd docs
bundle install
bundle exec jekyll serve
# Open http://localhost:4000/quorum-mcp/
```

### Writing Documentation

- Use Markdown with Jekyll front matter
- Include code examples
- Keep explanations clear and concise
- Add to navigation in `_config.yml`

## Questions?

- **Issues:** [GitHub Issues](https://github.com/aj-geddes/quorum-mcp/issues)
- **Discussions:** [GitHub Discussions](https://github.com/aj-geddes/quorum-mcp/discussions)

Thank you for contributing to Quorum-MCP!
