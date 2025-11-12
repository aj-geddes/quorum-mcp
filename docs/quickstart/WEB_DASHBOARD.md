# Web Dashboard Quick Start

## Introduction

The Quorum-MCP Web Dashboard provides an interactive browser-based interface for building multi-AI consensus without writing code. Perfect for experimentation, testing, and production use.

## Features

✨ **Interactive Query Builder** - Submit queries with a user-friendly form
📊 **Real-Time Consensus View** - Watch as providers build consensus
💰 **Cost Calculator** - Estimate costs before running queries
🔍 **Provider Comparison** - See responses side-by-side
📈 **Session History** - Track all your consensus sessions
⚡ **Live Updates** - WebSocket-powered real-time notifications

## Installation

### 1. Install Dependencies

The web dashboard requires FastAPI and Uvicorn:

```bash
# Install from repository root
pip install -e .

# Or install dependencies manually
pip install fastapi uvicorn[standard] websockets
```

### 2. Set Up API Keys

Configure your preferred AI providers by setting environment variables:

```bash
# Traditional providers
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# New providers (optional)
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
export NOVITA_API_KEY="..."

# Or use local Ollama (free, no API key needed)
# Just run: ollama serve
```

**Note**: You need at least ONE provider configured. We recommend starting with Ollama (free) or Novita AI (lowest cost).

### 3. Start the Server

```bash
# Using the command-line tool
quorum-web

# Or run directly
python -m quorum_mcp.web_server
```

The server will start on `http://localhost:8000`.

## Using the Dashboard

### Submit Your First Query

1. **Open the Dashboard**
   Navigate to `http://localhost:8000` in your browser

2. **Enter Your Query**
   Type your question in the "Your Query" field:
   ```
   What are the best practices for API design in 2025?
   ```

3. **Choose Settings**
   - **Mode**: Start with "Quick Consensus" for fast results
   - **Providers**: Select which AI providers to consult
   - **Context** (optional): Add constraints or background info

4. **Build Consensus**
   Click "Build Consensus" and watch the magic happen!

5. **Review Results**
   - See the consensus summary
   - View individual provider responses
   - Check confidence score
   - See total cost

### Understanding Modes

**⚡ Quick Consensus** (1 round)
- All providers respond simultaneously
- Fast results (typically <30 seconds)
- Best for: straightforward questions, quick testing
- Cost: 1x

**🔄 Full Deliberation** (3 rounds)
- Round 1: Independent analysis
- Round 2: Cross-review and critique
- Round 3: Final synthesis
- Best for: complex decisions, thorough analysis
- Cost: 3x

**😈 Devil's Advocate**
- One provider challenges the others
- Exposes weaknesses and alternative viewpoints
- Best for: stress-testing ideas, finding blind spots
- Cost: 2x

### Provider Selection Guide

Choose providers based on your priorities:

**For Quality** (Premium)
- ✅ Anthropic Claude - Best reasoning
- ✅ OpenAI GPT-4 - Broad knowledge

**For Balance** (Mid-range)
- ✅ Google Gemini - Fast with huge context
- ✅ Cohere - Enterprise RAG capabilities

**For Budget** (Cost-effective)
- ✅ Mistral AI - Best cloud pricing ($0.04-6/M)
- ✅ Novita AI - Ultra-low cost ($0.04/M!)

**For Privacy/Free**
- ✅ Ollama - 100% local, $0.00

**Pro Tip**: Mix quality + budget providers for best value!

## Using the Cost Calculator

Estimate your costs before committing:

1. **Navigate to "Cost Calculator" tab**

2. **Enter Your Usage**
   - Queries per month: 1,000
   - Average query length: 500 characters
   - Select providers to compare

3. **Choose Mode**
   - Quick Consensus: 1x cost
   - Full Deliberation: 3x cost

4. **Calculate**
   See per-query and monthly costs broken down by provider

### Example Cost Scenarios

**Scenario 1: Budget-Conscious**
- 1,000 queries/month
- Providers: Novita AI + Mistral AI + Ollama
- Mode: Quick Consensus
- **Cost**: ~$0.60/month

**Scenario 2: Quality + Value**
- 1,000 queries/month
- Providers: Claude + Gemini + Novita
- Mode: Quick Consensus
- **Cost**: ~$4.50/month

**Scenario 3: Premium Consensus**
- 1,000 queries/month
- Providers: Claude + GPT-4 + Cohere
- Mode: Full Deliberation
- **Cost**: ~$45/month

## Session Management

### View Session History

1. Go to "Sessions" tab
2. Browse all your consensus sessions
3. Filter by status (completed, in progress, failed)
4. Click any session to view details

### Export Results

- Click the "📥 Export" button on any result
- Downloads as JSON with complete session data
- Perfect for analysis, reporting, or archiving

## Provider Status

### Check Available Providers

1. Go to "Providers" tab
2. See all configured providers
3. View pricing and model information
4. Check availability status

### Troubleshooting

**No providers available?**
- Check your API keys are set correctly
- For Ollama: ensure `ollama serve` is running
- Check logs in terminal for errors

**Provider failed?**
- Verify API key is valid
- Check if you have quota/credits
- Try a different provider

## Advanced Features

### WebSocket Real-Time Updates

The dashboard uses WebSockets for live updates:
- Session status changes
- Consensus updates
- No need to manually refresh

Connection status shown in header (green = connected).

### API Access

The web server also provides a REST API:

```bash
# Health check
curl http://localhost:8000/api/health

# List providers
curl http://localhost:8000/api/providers

# Submit query (programmatically)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "mode": "quick_consensus"
  }'
```

View full API documentation at: `http://localhost:8000/api/docs`

## Configuration

### Custom Port

```bash
# Set custom port
uvicorn quorum_mcp.web_server:app --port 3000
```

### Production Deployment

For production use:

```bash
# Use production ASGI server
uvicorn quorum_mcp.web_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --no-reload
```

### Environment Variables

```bash
# Server configuration
export QUORUM_WEB_HOST="0.0.0.0"
export QUORUM_WEB_PORT="8000"

# Session management
export QUORUM_SESSION_TTL="24"  # hours

# Provider configuration (as shown above)
```

## Tips & Best Practices

### Getting the Best Consensus

1. **Be Specific**: Clear questions get better answers
2. **Add Context**: Provide relevant background information
3. **Mix Providers**: Different AIs have different strengths
4. **Use Full Deliberation**: For important decisions, use multi-round mode

### Cost Optimization

1. **Start with Novita/Mistral**: Test with budget providers first
2. **Use Quick Mode**: Reserve full deliberation for complex queries
3. **Leverage Ollama**: Use local inference when possible
4. **Monitor Costs**: Check the cost calculator before large runs

### Performance Tips

1. **Fewer Providers = Faster**: 2-3 providers is usually sufficient
2. **Quick Mode**: 10-30 seconds vs 1-2 minutes for full
3. **Provider Selection**: Some providers are faster than others

## Keyboard Shortcuts

- `Ctrl/Cmd + Enter` in query field: Submit query
- `Esc`: Clear form
- `Tab`: Navigate between form fields

## Troubleshooting

### Dashboard Won't Load

**Problem**: Page shows "Dashboard files not found"
**Solution**: Ensure static files are in `/static` directory

**Problem**: 404 errors for CSS/JS
**Solution**: Check FastAPI is mounting static files correctly

### Query Submission Fails

**Problem**: "No providers available" error
**Solution**:
1. Check API keys are set
2. Start Ollama if using local LLMs
3. View server logs for specific errors

**Problem**: Query times out
**Solution**:
1. Reduce number of providers
2. Use quick consensus mode
3. Check provider API status

### WebSocket Disconnects

**Problem**: Connection status shows disconnected
**Solution**:
1. Check browser console for errors
2. Verify firewall allows WebSocket connections
3. Try refreshing the page

## Next Steps

- **Read Provider Guides**: Learn about each provider's strengths
- **Explore Examples**: Check `/examples` directory for use cases
- **API Integration**: Use the REST API in your applications
- **Join Community**: Share feedback and get help

## Support

- **Documentation**: https://github.com/aj-geddes/quorum-mcp
- **API Docs**: http://localhost:8000/api/docs
- **Issues**: https://github.com/aj-geddes/quorum-mcp/issues

---

**Ready to get started?** Open `http://localhost:8000` and submit your first consensus query! 🚀
