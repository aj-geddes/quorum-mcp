// Quorum-MCP Dashboard JavaScript

// API Configuration
const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws`;

// Global State
let providers = [];
let currentSession = null;
let ws = null;
let currentPage = 0;
const pageSize = 20;

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

async function initializeDashboard() {
    // Set up tab navigation
    setupTabNavigation();

    // Connect WebSocket
    connectWebSocket();

    // Load initial data
    await loadProviders();
    await loadSessions();
    await updateSystemStats();

    // Set up form handlers
    setupQueryForm();
    setupCostCalculator();
    setupSessionFilters();

    // Update mode descriptions
    document.getElementById('mode').addEventListener('change', updateModeDescription);
}

// Tab Navigation
function setupTabNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabs = document.querySelectorAll('.tab-content');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;

            // Update active states
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            tabs.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === `${tabName}-tab`) {
                    tab.classList.add('active');

                    // Load data when switching to certain tabs
                    if (tabName === 'sessions') {
                        loadSessions();
                    } else if (tabName === 'providers') {
                        loadProviders();
                    }
                }
            });
        });
    });
}

// WebSocket Connection
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('WebSocket connected');
            updateConnectionStatus(true);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
            // Reconnect after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
    }
}

function handleWebSocketMessage(data) {
    if (data.type === 'session_update') {
        // Update current session if it matches
        if (currentSession && currentSession.session_id === data.session_id) {
            loadSessionDetails(data.session_id);
        }
    }
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    if (connected) {
        statusEl.classList.remove('disconnected');
    } else {
        statusEl.classList.add('disconnected');
    }
}

// Load Providers
async function loadProviders() {
    try {
        const response = await fetch(`${API_BASE}/api/providers`);
        providers = await response.json();

        // Update provider count
        document.getElementById('provider-count').textContent =
            `${providers.length} Provider${providers.length !== 1 ? 's' : ''}`;

        // Populate provider checkboxes
        populateProviderCheckboxes();

        // Populate providers grid
        populateProvidersGrid();

        // Populate cost calculator providers
        populateCostProviderCheckboxes();

    } catch (error) {
        console.error('Failed to load providers:', error);
        showError('Failed to load providers');
    }
}

function populateProviderCheckboxes() {
    const container = document.getElementById('provider-checkboxes');
    container.innerHTML = '';

    providers.forEach(provider => {
        const label = document.createElement('label');
        label.innerHTML = `
            <input type="checkbox" name="providers" value="${provider.name}" checked>
            <span>${provider.display_name}</span>
        `;
        container.appendChild(label);
    });
}

function populateCostProviderCheckboxes() {
    const container = document.getElementById('cost-provider-checkboxes');
    container.innerHTML = '';

    providers.forEach(provider => {
        const label = document.createElement('label');
        label.innerHTML = `
            <input type="checkbox" class="cost-provider-checkbox" value="${provider.name}" checked>
            <span>${provider.display_name}</span>
        `;
        container.appendChild(label);
    });
}

function populateProvidersGrid() {
    const grid = document.getElementById('providers-list');
    grid.innerHTML = '';

    providers.forEach(provider => {
        const card = document.createElement('div');
        card.className = 'provider-item';

        const pricingInfo = provider.pricing || {};
        const features = provider.features || [];

        card.innerHTML = `
            <div class="provider-header">
                <h3 class="provider-title">${provider.display_name}</h3>
                <div class="provider-status ${provider.available ? '' : 'unavailable'}"></div>
            </div>
            <div class="provider-details">
                <p><strong>Default Model:</strong> ${provider.default_model}</p>
                <p><strong>Available Models:</strong> ${provider.models.length}</p>
                ${features.length > 0 ? `<p><strong>Features:</strong> ${features.join(', ')}</p>` : ''}
            </div>
            ${pricingInfo.input_per_1m !== undefined ? `
                <div class="provider-pricing">
                    <strong>Pricing:</strong><br>
                    Input: $${pricingInfo.input_per_1m}/1M tokens<br>
                    Output: $${pricingInfo.output_per_1m}/1M tokens
                </div>
            ` : ''}
        `;

        grid.appendChild(card);
    });
}

// Query Form
function setupQueryForm() {
    const form = document.getElementById('query-form');
    const submitBtn = document.getElementById('submit-btn');
    const clearBtn = document.getElementById('clear-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get form data
        const query = document.getElementById('query').value;
        const context = document.getElementById('context').value;
        const mode = document.getElementById('mode').value;

        // Get selected providers
        const selectedProviders = Array.from(
            document.querySelectorAll('input[name="providers"]:checked')
        ).map(cb => cb.value);

        if (selectedProviders.length === 0) {
            showError('Please select at least one provider');
            return;
        }

        // Disable form
        submitBtn.disabled = true;
        submitBtn.textContent = 'Building Consensus...';

        try {
            const response = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query,
                    context: context || null,
                    mode,
                    providers: selectedProviders,
                }),
            });

            if (!response.ok) {
                throw new Error('Query submission failed');
            }

            const result = await response.json();
            currentSession = result;

            // Show results section
            document.getElementById('results-section').style.display = 'block';
            document.getElementById('results-loading').style.display = 'block';
            document.getElementById('results-content').style.display = 'none';

            // Poll for results
            pollSessionResults(result.session_id);

        } catch (error) {
            console.error('Query error:', error);
            showError('Failed to submit query. Please try again.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span>Build Consensus</span>';
        }
    });

    clearBtn.addEventListener('click', () => {
        form.reset();
        document.getElementById('results-section').style.display = 'none';
    });
}

async function pollSessionResults(sessionId) {
    const maxAttempts = 60; // 60 seconds timeout
    let attempts = 0;

    const poll = async () => {
        attempts++;

        try {
            const session = await loadSessionDetails(sessionId);

            if (session.status === 'completed' || session.status === 'failed') {
                displayResults(session);
                return;
            }

            if (attempts < maxAttempts) {
                // Update loading details
                document.getElementById('loading-details').textContent =
                    `${session.status.replace('_', ' ')}... (${attempts}s)`;

                setTimeout(poll, 1000);
            } else {
                showError('Session timeout');
                document.getElementById('results-loading').style.display = 'none';
            }

        } catch (error) {
            console.error('Error polling session:', error);
            showError('Failed to load session results');
        }
    };

    poll();
}

async function loadSessionDetails(sessionId) {
    const response = await fetch(`${API_BASE}/api/session/${sessionId}`);
    if (!response.ok) {
        throw new Error('Failed to load session');
    }
    return await response.json();
}

function displayResults(session) {
    document.getElementById('results-loading').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';

    // Session info
    document.getElementById('session-id').textContent = session.session_id;
    document.getElementById('session-status').textContent = session.status;
    document.getElementById('session-status').className = `badge ${session.status}`;
    document.getElementById('session-cost').textContent =
        `$${(session.metadata?.total_cost || 0).toFixed(4)}`;

    // Consensus
    if (session.consensus) {
        const confidence = session.consensus.confidence || 0;
        document.getElementById('confidence-score').textContent = `${Math.round(confidence * 100)}%`;
        document.getElementById('confidence-bar').style.width = `${confidence * 100}%`;
        document.getElementById('consensus-text').textContent =
            session.consensus.summary || 'No consensus summary available';
    }

    // Provider responses
    displayProviderResponses(session.provider_responses);

    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function displayProviderResponses(responses) {
    const container = document.getElementById('provider-responses-list');
    container.innerHTML = '';

    Object.entries(responses).forEach(([providerName, rounds]) => {
        // Display latest round
        const latestRound = Math.max(...Object.keys(rounds).map(Number));
        const response = rounds[latestRound];

        if (response && response.content) {
            const card = document.createElement('div');
            card.className = 'provider-card';

            card.innerHTML = `
                <div class="provider-card-header">
                    <span class="provider-name">${providerName}</span>
                    <span class="provider-meta">
                        Round ${latestRound} • ${response.tokens_input + response.tokens_output} tokens
                    </span>
                </div>
                <div class="provider-content">${escapeHtml(response.content)}</div>
            `;

            container.appendChild(card);
        }
    });
}

// Sessions List
async function loadSessions() {
    const statusFilter = document.getElementById('status-filter')?.value || '';

    try {
        const url = `${API_BASE}/api/sessions?limit=${pageSize}&offset=${currentPage * pageSize}${statusFilter ? `&status=${statusFilter}` : ''}`;
        const response = await fetch(url);
        const data = await response.json();

        displaySessions(data.sessions);
        updatePagination(data.total, data.offset);

    } catch (error) {
        console.error('Failed to load sessions:', error);
        showError('Failed to load sessions');
    }
}

function displaySessions(sessions) {
    const container = document.getElementById('sessions-list');
    container.innerHTML = '';

    if (sessions.length === 0) {
        container.innerHTML = '<div class="loading">No sessions found</div>';
        return;
    }

    sessions.forEach(session => {
        const card = document.createElement('div');
        card.className = 'session-card';
        card.onclick = () => viewSession(session.session_id);

        const date = new Date(session.created_at).toLocaleString();

        card.innerHTML = `
            <div class="session-info">
                <div class="session-query">${escapeHtml(session.query)}</div>
                <div class="session-meta">
                    <span>${date}</span>
                    <span>${session.mode}</span>
                    <span>$${session.cost.toFixed(4)}</span>
                </div>
            </div>
            <span class="badge ${session.status}">${session.status}</span>
        `;

        container.appendChild(card);
    });
}

function updatePagination(total, offset) {
    const totalPages = Math.ceil(total / pageSize);
    const currentPageNum = Math.floor(offset / pageSize) + 1;

    document.getElementById('page-info').textContent = `Page ${currentPageNum} of ${totalPages}`;

    document.getElementById('prev-page').disabled = currentPageNum === 1;
    document.getElementById('next-page').disabled = currentPageNum === totalPages;
}

function setupSessionFilters() {
    document.getElementById('refresh-sessions')?.addEventListener('click', loadSessions);
    document.getElementById('status-filter')?.addEventListener('change', () => {
        currentPage = 0;
        loadSessions();
    });

    document.getElementById('prev-page')?.addEventListener('click', () => {
        if (currentPage > 0) {
            currentPage--;
            loadSessions();
        }
    });

    document.getElementById('next-page')?.addEventListener('click', () => {
        currentPage++;
        loadSessions();
    });
}

async function viewSession(sessionId) {
    // Switch to query tab and display session
    document.querySelector('[data-tab="query"]').click();

    // Load and display session
    const session = await loadSessionDetails(sessionId);
    displayResults(session);
}

// Cost Calculator
function setupCostCalculator() {
    const calcButton = document.getElementById('calculate-cost');
    const queriesInput = document.getElementById('queries-per-month');
    const slider = document.getElementById('queries-slider');

    // Sync slider and input
    slider.addEventListener('input', (e) => {
        queriesInput.value = e.target.value;
    });

    queriesInput.addEventListener('input', (e) => {
        slider.value = e.target.value;
    });

    calcButton.addEventListener('click', calculateCost);
}

async function calculateCost() {
    const queriesPerMonth = parseInt(document.getElementById('queries-per-month').value);
    const queryLength = parseInt(document.getElementById('query-length').value);
    const mode = document.getElementById('cost-mode').value;

    const selectedProviders = Array.from(
        document.querySelectorAll('.cost-provider-checkbox:checked')
    ).map(cb => cb.value);

    if (selectedProviders.length === 0) {
        showError('Please select at least one provider');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/estimate-cost`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                queries_per_month: queriesPerMonth,
                query_length: queryLength,
                providers: selectedProviders,
                mode,
            }),
        });

        const data = await response.json();
        displayCostResults(data);

    } catch (error) {
        console.error('Cost calculation error:', error);
        showError('Failed to calculate costs');
    }
}

function displayCostResults(data) {
    const resultsDiv = document.getElementById('cost-results');
    resultsDiv.style.display = 'block';

    document.getElementById('cost-per-query').textContent = `$${data.per_query_average.toFixed(4)}`;
    document.getElementById('cost-monthly').textContent = `$${data.monthly_total.toFixed(2)}`;

    // Breakdown by provider
    const breakdownContainer = document.getElementById('cost-by-provider');
    breakdownContainer.innerHTML = '';

    Object.entries(data.by_provider).forEach(([provider, costs]) => {
        const item = document.createElement('div');
        item.className = 'provider-cost-item';
        item.innerHTML = `
            <span>${provider}</span>
            <span>$${costs.monthly.toFixed(2)}/month ($${costs.per_query.toFixed(4)}/query)</span>
        `;
        breakdownContainer.appendChild(item);
    });

    // Assumptions
    const assumptionsList = document.getElementById('cost-assumptions-list');
    assumptionsList.innerHTML = '';

    const assumptions = [
        `Average input: ${data.assumptions.avg_input_tokens} tokens`,
        `Average output: ${data.assumptions.avg_output_tokens} tokens`,
        `Mode multiplier: ${data.assumptions.mode_multiplier}x`,
    ];

    assumptions.forEach(assumption => {
        const li = document.createElement('li');
        li.textContent = assumption;
        assumptionsList.appendChild(li);
    });
}

// System Stats
async function updateSystemStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const stats = await response.json();

        // Update any stats displays
        console.log('System stats:', stats);

    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Mode Description
function updateModeDescription() {
    const mode = document.getElementById('mode').value;
    const descriptions = {
        'quick_consensus': 'Fast consensus from all providers simultaneously',
        'full_deliberation': 'Multi-round deliberation with cross-review and synthesis',
        'devils_advocate': 'One provider challenges the others for critical analysis',
    };

    document.getElementById('mode-description').textContent =
        descriptions[mode] || '';
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // Simple error display - could be enhanced with a toast/notification system
    alert(message);
}

// Export functionality
document.getElementById('export-btn')?.addEventListener('click', () => {
    if (currentSession) {
        const dataStr = JSON.stringify(currentSession, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quorum-session-${currentSession.session_id}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
});
