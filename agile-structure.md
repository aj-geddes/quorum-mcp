# Quorum-MCP Agile Structure

## Epic 1: Core MCP Server Infrastructure
**Priority**: CRITICAL | **Status**: Not Started

### Story 1.1: Basic MCP Server Setup
**Points**: 3 | **Priority**: P0

**Description**: Create the foundational MCP server using FastMCP with proper project structure, dependencies, and configuration.

**Acceptance Criteria**:
- [ ] Python project with proper dependencies (mcp, asyncio, pydantic)
- [ ] FastMCP server initializes successfully
- [ ] Server connects via stdio transport
- [ ] Basic health check/ping capability
- [ ] Configuration file for server metadata

**Tasks**:
- [ ] Task 1.1.1: Initialize Python project structure with pyproject.toml
- [ ] Task 1.1.2: Install MCP SDK and core dependencies
- [ ] Task 1.1.3: Create main server.py with FastMCP initialization
- [ ] Task 1.1.4: Implement stdio transport connection
- [ ] Task 1.1.5: Add configuration management (YAML/JSON)

---

### Story 1.2: Implement q_in Tool
**Points**: 5 | **Priority**: P0

**Description**: Create the `q_in` tool that accepts user queries and initiates quorum consensus sessions.

**Acceptance Criteria**:
- [ ] Tool accepts query string and optional parameters
- [ ] Input validation with Pydantic schemas
- [ ] Returns session ID for tracking
- [ ] Handles errors gracefully
- [ ] Logs all incoming requests

**Tasks**:
- [ ] Task 1.2.1: Define q_in tool schema (input/output)
- [ ] Task 1.2.2: Implement tool handler function
- [ ] Task 1.2.3: Create session ID generation logic
- [ ] Task 1.2.4: Add input validation and error handling
- [ ] Task 1.2.5: Integrate logging for audit trail

---

### Story 1.3: Implement q_out Tool
**Points**: 5 | **Priority**: P0

**Description**: Create the `q_out` tool that retrieves quorum consensus results by session ID.

**Acceptance Criteria**:
- [ ] Tool accepts session ID parameter
- [ ] Returns formatted consensus results
- [ ] Handles "in progress" vs "completed" states
- [ ] Provides error details if session failed
- [ ] Returns structured JSON response

**Tasks**:
- [ ] Task 1.3.1: Define q_out tool schema
- [ ] Task 1.3.2: Implement result retrieval logic
- [ ] Task 1.3.3: Format consensus output structure
- [ ] Task 1.3.4: Handle session state checks
- [ ] Task 1.3.5: Add error handling for invalid sessions

---

## Epic 2: Provider Abstraction Layer
**Priority**: CRITICAL | **Status**: Not Started

### Story 2.1: Provider Interface Design
**Points**: 5 | **Priority**: P0

**Description**: Create abstract base class for AI provider integrations with common interface.

**Acceptance Criteria**:
- [ ] Abstract Provider class with standard methods
- [ ] Common request/response format
- [ ] Error handling interface
- [ ] Rate limiting hooks
- [ ] Cost tracking interface

**Tasks**:
- [ ] Task 2.1.1: Design Provider abstract base class
- [ ] Task 2.1.2: Define ProviderRequest and ProviderResponse models
- [ ] Task 2.1.3: Create error handling hierarchy
- [ ] Task 2.1.4: Add rate limiting interface
- [ ] Task 2.1.5: Design cost tracking mechanism

---

### Story 2.2: Anthropic Claude Integration
**Points**: 8 | **Priority**: P1

**Description**: Implement concrete provider for Anthropic's Claude API.

**Acceptance Criteria**:
- [ ] API key configuration and secure storage
- [ ] Request formatting for Claude API
- [ ] Response parsing and normalization
- [ ] Retry logic with exponential backoff
- [ ] Token counting and cost calculation

**Tasks**:
- [ ] Task 2.2.1: Install anthropic Python SDK
- [ ] Task 2.2.2: Implement ClaudeProvider class
- [ ] Task 2.2.3: Create request formatter for Claude API
- [ ] Task 2.2.4: Implement response parser
- [ ] Task 2.2.5: Add retry and error handling logic
- [ ] Task 2.2.6: Implement token counting

---

### Story 2.3: OpenAI GPT Integration
**Points**: 8 | **Priority**: P1

**Description**: Implement concrete provider for OpenAI's GPT-4 API.

**Acceptance Criteria**:
- [ ] API key configuration
- [ ] Request formatting for OpenAI API
- [ ] Response parsing and normalization
- [ ] Retry logic with exponential backoff
- [ ] Token counting and cost calculation

**Tasks**:
- [ ] Task 2.3.1: Install openai Python SDK
- [ ] Task 2.3.2: Implement OpenAIProvider class
- [ ] Task 2.3.3: Create request formatter for OpenAI API
- [ ] Task 2.3.4: Implement response parser
- [ ] Task 2.3.5: Add retry and error handling logic
- [ ] Task 2.3.6: Implement token counting

---

### Story 2.4: Google Gemini Integration
**Points**: 8 | **Priority**: P2

**Description**: Implement concrete provider for Google's Gemini API.

**Acceptance Criteria**:
- [ ] API key configuration
- [ ] Request formatting for Gemini API
- [ ] Response parsing and normalization
- [ ] Retry logic
- [ ] Token counting

**Tasks**:
- [ ] Task 2.4.1: Install google-generativeai SDK
- [ ] Task 2.4.2: Implement GeminiProvider class
- [ ] Task 2.4.3: Create request formatter
- [ ] Task 2.4.4: Implement response parser
- [ ] Task 2.4.5: Add error handling
- [ ] Task 2.4.6: Implement token counting

---

### Story 2.5: Mistral AI Integration
**Points**: 8 | **Priority**: P2

**Description**: Implement concrete provider for Mistral AI API.

**Acceptance Criteria**:
- [ ] API key configuration
- [ ] Request formatting
- [ ] Response parsing
- [ ] Error handling
- [ ] Token counting

**Tasks**:
- [ ] Task 2.5.1: Install mistralai SDK
- [ ] Task 2.5.2: Implement MistralProvider class
- [ ] Task 2.5.3: Create request formatter
- [ ] Task 2.5.4: Implement response parser
- [ ] Task 2.5.5: Add error handling
- [ ] Task 2.5.6: Implement cost tracking

---

## Epic 3: Orchestration Engine
**Priority**: CRITICAL | **Status**: Not Started

### Story 3.1: Session State Management
**Points**: 8 | **Priority**: P0

**Description**: Implement session management system to track quorum consultations across multiple rounds.

**Acceptance Criteria**:
- [ ] Session data model with all required fields
- [ ] In-memory session storage (dict-based)
- [ ] Session lifecycle management (create/update/complete)
- [ ] Thread-safe operations with asyncio locks
- [ ] Session expiration and cleanup

**Tasks**:
- [ ] Task 3.1.1: Design Session data model (Pydantic)
- [ ] Task 3.1.2: Implement SessionManager class
- [ ] Task 3.1.3: Add create/read/update operations
- [ ] Task 3.1.4: Implement async locks for thread safety
- [ ] Task 3.1.5: Add session expiration logic
- [ ] Task 3.1.6: Create cleanup background task

---

### Story 3.2: Basic Orchestration Logic
**Points**: 13 | **Priority**: P0

**Description**: Implement core orchestration to distribute queries to multiple providers and collect responses.

**Acceptance Criteria**:
- [ ] Parallel request distribution to providers
- [ ] Response collection and aggregation
- [ ] Timeout handling per provider
- [ ] Graceful degradation on provider failures
- [ ] Basic consensus calculation

**Tasks**:
- [ ] Task 3.2.1: Design Orchestrator class
- [ ] Task 3.2.2: Implement parallel provider invocation (asyncio.gather)
- [ ] Task 3.2.3: Add timeout handling per provider
- [ ] Task 3.2.4: Implement response aggregation logic
- [ ] Task 3.2.5: Add failure handling and fallbacks
- [ ] Task 3.2.6: Create basic consensus algorithm (majority vote)

---

### Story 3.3: Multi-Round Deliberation
**Points**: 13 | **Priority**: P1

**Description**: Implement multi-round discussion where AIs respond to each other's assessments.

**Acceptance Criteria**:
- [ ] Round 1: Independent analysis
- [ ] Round 2: Cross-review and discussion
- [ ] Round 3: Consensus building
- [ ] Context preservation between rounds
- [ ] Round progression logic

**Tasks**:
- [ ] Task 3.3.1: Design Round data model
- [ ] Task 3.3.2: Implement Round 1 orchestration (independent)
- [ ] Task 3.3.3: Implement Round 2 orchestration (cross-review)
- [ ] Task 3.3.4: Implement Round 3 orchestration (consensus)
- [ ] Task 3.3.5: Add context packaging between rounds
- [ ] Task 3.3.6: Create round progression controller

---

### Story 3.4: Operational Modes
**Points**: 8 | **Priority**: P2

**Description**: Implement different operational modes (Full Deliberation, Quick Consensus, Devil's Advocate).

**Acceptance Criteria**:
- [ ] Mode selection parameter in q_in
- [ ] Full Deliberation mode (all providers, multiple rounds)
- [ ] Quick Consensus mode (subset, single round)
- [ ] Devil's Advocate mode (critical stance)
- [ ] Mode-specific configurations

**Tasks**:
- [ ] Task 3.4.1: Define OperationalMode enum
- [ ] Task 3.4.2: Implement Full Deliberation mode logic
- [ ] Task 3.4.3: Implement Quick Consensus mode logic
- [ ] Task 3.4.4: Implement Devil's Advocate mode logic
- [ ] Task 3.4.5: Add mode-specific configurations
- [ ] Task 3.4.6: Create mode selection handler

---

## Epic 4: Synthesis and Consensus
**Priority**: HIGH | **Status**: Not Started

### Story 4.1: Response Parsing and Normalization
**Points**: 5 | **Priority**: P1

**Description**: Parse and normalize responses from different providers into common format.

**Acceptance Criteria**:
- [ ] Extract key points from each response
- [ ] Identify confidence levels
- [ ] Extract concerns and recommendations
- [ ] Normalize to common schema
- [ ] Handle malformed responses

**Tasks**:
- [ ] Task 4.1.1: Design normalized response schema
- [ ] Task 4.1.2: Implement response parser
- [ ] Task 4.1.3: Add confidence extraction logic
- [ ] Task 4.1.4: Create concern/recommendation extractor
- [ ] Task 4.1.5: Add error handling for malformed responses

---

### Story 4.2: Consensus Algorithm
**Points**: 13 | **Priority**: P1

**Description**: Implement consensus-building algorithm that identifies agreement, disagreement, and synthesizes recommendations.

**Acceptance Criteria**:
- [ ] Identify unanimous agreement areas
- [ ] Detect disagreements and conflicts
- [ ] Weight recommendations by agreement level
- [ ] Preserve minority opinions
- [ ] Calculate overall confidence score

**Tasks**:
- [ ] Task 4.2.1: Design consensus algorithm
- [ ] Task 4.2.2: Implement agreement detection
- [ ] Task 4.2.3: Implement disagreement detection
- [ ] Task 4.2.4: Add recommendation weighting
- [ ] Task 4.2.5: Create minority opinion preservation
- [ ] Task 4.2.6: Calculate confidence scores

---

### Story 4.3: Report Generation
**Points**: 8 | **Priority**: P1

**Description**: Generate formatted consensus reports for end users.

**Acceptance Criteria**:
- [ ] Structured JSON output
- [ ] Human-readable summary
- [ ] Agreement/disagreement sections
- [ ] Prioritized recommendations
- [ ] Full audit trail with all responses

**Tasks**:
- [ ] Task 4.3.1: Design report schema
- [ ] Task 4.3.2: Implement summary generator
- [ ] Task 4.3.3: Create agreement section formatter
- [ ] Task 4.3.4: Create disagreement section formatter
- [ ] Task 4.3.5: Add recommendation prioritization
- [ ] Task 4.3.6: Include full audit trail

---

## Epic 5: Configuration and Management
**Priority**: MEDIUM | **Status**: Not Started

### Story 5.1: Configuration System
**Points**: 5 | **Priority**: P1

**Description**: Implement configuration management for API keys, provider settings, and operational parameters.

**Acceptance Criteria**:
- [ ] YAML/JSON configuration file
- [ ] Environment variable support
- [ ] API key secure storage
- [ ] Provider-specific settings
- [ ] Default operational parameters

**Tasks**:
- [ ] Task 5.1.1: Design configuration schema
- [ ] Task 5.1.2: Implement configuration loader
- [ ] Task 5.1.3: Add environment variable support
- [ ] Task 5.1.4: Create API key management
- [ ] Task 5.1.5: Add validation for configuration
- [ ] Task 5.1.6: Create configuration documentation

---

### Story 5.2: Cost Tracking
**Points**: 5 | **Priority**: P2

**Description**: Implement cost tracking per provider, per session, and aggregate reporting.

**Acceptance Criteria**:
- [ ] Token counting per request
- [ ] Cost calculation per provider
- [ ] Session-level cost aggregation
- [ ] Cost reporting in q_out
- [ ] Budget warnings

**Tasks**:
- [ ] Task 5.2.1: Define cost calculation formulas
- [ ] Task 5.2.2: Implement per-provider cost tracking
- [ ] Task 5.2.3: Add session-level aggregation
- [ ] Task 5.2.4: Create cost reporting formatter
- [ ] Task 5.2.5: Add budget threshold warnings

---

### Story 5.3: Logging and Debugging
**Points**: 3 | **Priority**: P2

**Description**: Implement comprehensive logging for debugging and audit trails.

**Acceptance Criteria**:
- [ ] Structured logging with log levels
- [ ] Request/response logging
- [ ] Error tracking
- [ ] Performance metrics
- [ ] Session event logging

**Tasks**:
- [ ] Task 5.3.1: Set up Python logging framework
- [ ] Task 5.3.2: Add request/response loggers
- [ ] Task 5.3.3: Implement error logging
- [ ] Task 5.3.4: Add performance timing logs
- [ ] Task 5.3.5: Create session event logger

---

## Epic 6: Testing and Documentation
**Priority**: MEDIUM | **Status**: Not Started

### Story 6.1: Unit Tests
**Points**: 8 | **Priority**: P2

**Description**: Create comprehensive unit tests for all components.

**Acceptance Criteria**:
- [ ] Provider abstraction tests
- [ ] Orchestration logic tests
- [ ] Consensus algorithm tests
- [ ] Session management tests
- [ ] 80%+ code coverage

**Tasks**:
- [ ] Task 6.1.1: Set up pytest framework
- [ ] Task 6.1.2: Write provider tests
- [ ] Task 6.1.3: Write orchestrator tests
- [ ] Task 6.1.4: Write consensus tests
- [ ] Task 6.1.5: Write session manager tests
- [ ] Task 6.1.6: Measure and improve coverage

---

### Story 6.2: Integration Tests
**Points**: 8 | **Priority**: P2

**Description**: Create end-to-end integration tests with mock providers.

**Acceptance Criteria**:
- [ ] Mock provider implementations
- [ ] Full workflow tests (q_in → q_out)
- [ ] Multi-round deliberation tests
- [ ] Error scenario tests
- [ ] Performance benchmarks

**Tasks**:
- [ ] Task 6.2.1: Create mock providers
- [ ] Task 6.2.2: Write end-to-end workflow tests
- [ ] Task 6.2.3: Test multi-round scenarios
- [ ] Task 6.2.4: Test error conditions
- [ ] Task 6.2.5: Create performance benchmarks

---

### Story 6.3: User Documentation
**Points**: 5 | **Priority**: P3

**Description**: Create comprehensive user documentation.

**Acceptance Criteria**:
- [ ] README with quickstart
- [ ] Configuration guide
- [ ] API reference for tools
- [ ] Operational mode explanations
- [ ] Troubleshooting guide

**Tasks**:
- [ ] Task 6.3.1: Write README.md
- [ ] Task 6.3.2: Create configuration guide
- [ ] Task 6.3.3: Document tool APIs
- [ ] Task 6.3.4: Explain operational modes
- [ ] Task 6.3.5: Create troubleshooting section

---

## Sprint Plan

### Sprint 1: Foundation (MVP)
**Goal**: Basic working MCP server with simple quorum functionality

**Stories**:
- Story 1.1: Basic MCP Server Setup
- Story 1.2: Implement q_in Tool
- Story 1.3: Implement q_out Tool
- Story 2.1: Provider Interface Design
- Story 2.2: Anthropic Claude Integration
- Story 2.3: OpenAI GPT Integration
- Story 3.1: Session State Management
- Story 3.2: Basic Orchestration Logic

**Deliverable**: Working MCP server that can query 2+ providers and return aggregated responses

---

### Sprint 2: Enhanced Consensus
**Goal**: Multi-round deliberation and consensus algorithms

**Stories**:
- Story 3.3: Multi-Round Deliberation
- Story 4.1: Response Parsing and Normalization
- Story 4.2: Consensus Algorithm
- Story 4.3: Report Generation
- Story 5.1: Configuration System

**Deliverable**: Full deliberation mode with synthesized consensus reports

---

### Sprint 3: Polish and Scale
**Goal**: Additional providers, modes, and production readiness

**Stories**:
- Story 2.4: Google Gemini Integration
- Story 2.5: Mistral AI Integration
- Story 3.4: Operational Modes
- Story 5.2: Cost Tracking
- Story 5.3: Logging and Debugging
- Story 6.1: Unit Tests
- Story 6.2: Integration Tests
- Story 6.3: User Documentation

**Deliverable**: Production-ready MCP server with full feature set

---

## Priority Legend
- **P0**: Critical path, must have for MVP
- **P1**: High priority, needed for full functionality
- **P2**: Medium priority, enhances usability
- **P3**: Low priority, nice to have

## Story Points Scale
- 1-2: Trivial, < 2 hours
- 3-5: Small, 2-4 hours
- 8: Medium, 1 day
- 13: Large, 2-3 days
- 21+: Too large, needs breakdown
