// Quorum-MCP Web UI Application Logic

// API Base URL
const API_BASE = window.location.origin;

// WebSocket connection
let ws = null;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initQueryForm();
    initBudgetForm();
    loadProviders();
    loadSessions();
    loadBudgetStatus();
    loadBenchmarks();
    connectWebSocket();

    // Refresh data periodically
    setInterval(loadProviders, 30000); // Every 30 seconds
    setInterval(loadBudgetStatus, 30000);
    setInterval(loadBenchmarks, 60000); // Every minute
});

// Tab Management
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;

            // Update active states
            tabButtons.forEach(btn => {
                btn.classList.remove('border-blue-500', 'text-blue-600');
                btn.classList.add('border-transparent', 'text-gray-500');
            });
            button.classList.remove('border-transparent', 'text-gray-500');
            button.classList.add('border-blue-500', 'text-blue-600');

            // Show/hide content
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${tabName}-tab`).classList.add('active');

            // Load data for the tab
            switch(tabName) {
                case 'providers':
                    loadProviders();
                    break;
                case 'sessions':
                    loadSessions();
                    break;
                case 'budget':
                    loadBudgetStatus();
                    loadBudgetAlerts();
                    break;
                case 'benchmark':
                    loadBenchmarks();
                    break;
            }
        });
    });
}

// Query Form
function initQueryForm() {
    const form = document.getElementById('query-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = document.getElementById('query-input').value.trim();
        if (!query) {
            alert('Please enter a query');
            return;
        }

        const context = document.getElementById('context-input').value.trim();
        const mode = document.getElementById('mode-select').value;
        const temperature = parseFloat(document.getElementById('temperature-input').value);

        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';

        // Hide previous results
        document.getElementById('results-section').classList.add('hidden');

        try {
            const response = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    context: context || null,
                    mode,
                    temperature
                })
            });

            if (!response.ok) {
                throw new Error('Failed to submit query');
            }

            const result = await response.json();
            displayQueryResult(result);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = originalText;
        }
    });
}

function displayQueryResult(result) {
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');

    let html = `
        <div class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <span class="text-sm text-gray-500">Session ID:</span>
                    <code class="ml-2 text-sm bg-gray-200 px-2 py-1 rounded">${result.session_id}</code>
                </div>
                <div>
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium
                        ${result.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}">
                        <i class="fas fa-${result.status === 'completed' ? 'check' : 'clock'} mr-2"></i>
                        ${result.status}
                    </span>
                </div>
            </div>

            ${result.consensus ? `
                <div>
                    <h4 class="font-semibold mb-2">Consensus Summary</h4>
                    <div class="bg-white rounded-lg p-4 border border-gray-200">
                        <p class="text-gray-800">${result.consensus.summary || 'No summary available'}</p>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <span class="text-sm text-gray-500">Confidence:</span>
                        <div class="mt-1">
                            <div class="flex items-center">
                                <div class="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                    <div class="bg-blue-600 h-2 rounded-full"
                                        style="width: ${(result.consensus.confidence * 100).toFixed(0)}%"></div>
                                </div>
                                <span class="text-sm font-medium">${(result.consensus.confidence * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <span class="text-sm text-gray-500">Total Cost:</span>
                        <div class="mt-1 text-lg font-semibold text-gray-900">
                            $${result.consensus.cost?.total_cost?.toFixed(4) || '0.0000'}
                        </div>
                    </div>
                </div>

                ${result.consensus.agreement_areas && result.consensus.agreement_areas.length > 0 ? `
                    <div>
                        <h4 class="font-semibold mb-2">Areas of Agreement</h4>
                        <ul class="list-disc list-inside space-y-1">
                            ${result.consensus.agreement_areas.map(area => `<li class="text-gray-700">${area}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                ${result.consensus.disagreement_areas && result.consensus.disagreement_areas.length > 0 ? `
                    <div>
                        <h4 class="font-semibold mb-2">Areas of Disagreement</h4>
                        <ul class="list-disc list-inside space-y-1">
                            ${result.consensus.disagreement_areas.map(area => `<li class="text-gray-700">${area}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            ` : '<p class="text-gray-500">No consensus data available</p>'}
        </div>
    `;

    resultsContainer.innerHTML = html;
    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Providers
async function loadProviders() {
    try {
        const response = await fetch(`${API_BASE}/api/providers`);
        const data = await response.json();

        const grid = document.getElementById('providers-grid');
        if (data.providers.length === 0) {
            grid.innerHTML = '<div class="col-span-full text-center py-12 text-gray-500">No providers available</div>';
            return;
        }

        grid.innerHTML = data.providers.map(provider => `
            <div class="provider-card bg-white rounded-lg shadow p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold">${provider.name}</h3>
                    <span class="status-${provider.health} w-3 h-3 rounded-full"></span>
                </div>
                <div class="space-y-2">
                    <div class="flex justify-between text-sm">
                        <span class="text-gray-500">Model:</span>
                        <span class="font-medium">${provider.model}</span>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span class="text-gray-500">Status:</span>
                        <span class="font-medium ${provider.health === 'healthy' ? 'text-green-600' : provider.health === 'degraded' ? 'text-yellow-600' : 'text-red-600'}">
                            ${provider.health}
                        </span>
                    </div>
                    ${provider.health_details.response_time ? `
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-500">Response Time:</span>
                            <span class="font-medium">${provider.health_details.response_time.toFixed(2)}s</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load providers:', error);
    }
}

// Sessions
async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions?limit=20`);
        const data = await response.json();

        const list = document.getElementById('sessions-list');
        if (data.sessions.length === 0) {
            list.innerHTML = '<div class="text-center py-12 text-gray-500">No sessions yet</div>';
            return;
        }

        list.innerHTML = `
            <div class="space-y-3">
                ${data.sessions.map(session => `
                    <div class="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 cursor-pointer"
                        onclick="viewSession('${session.session_id}')">
                        <div class="flex items-start justify-between">
                            <div class="flex-1">
                                <p class="text-sm font-medium text-gray-900">${session.query}</p>
                                <div class="mt-1 flex items-center space-x-3 text-xs text-gray-500">
                                    <span><i class="fas fa-clock mr-1"></i>${new Date(session.created_at).toLocaleString()}</span>
                                    <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium
                                        ${session.status === 'completed' ? 'bg-green-100 text-green-800' :
                                          session.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                                          'bg-red-100 text-red-800'}">
                                        ${session.status}
                                    </span>
                                    <span><i class="fas fa-cog mr-1"></i>${session.mode}</span>
                                </div>
                            </div>
                            <i class="fas fa-chevron-right text-gray-400"></i>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

async function viewSession(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/api/session/${sessionId}`);
        const session = await response.json();

        // Switch to query tab and show results
        document.querySelector('[data-tab="query"]').click();
        displayQueryResult(session);
    } catch (error) {
        alert(`Failed to load session: ${error.message}`);
    }
}

// Budget
function initBudgetForm() {
    const form = document.getElementById('budget-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const provider = document.getElementById('budget-provider').value || null;
        const limit = parseFloat(document.getElementById('budget-limit').value);
        const period = document.getElementById('budget-period').value;
        const enforce = document.getElementById('budget-enforce').checked;

        try {
            const response = await fetch(`${API_BASE}/api/budget`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, limit, period, enforce })
            });

            if (!response.ok) throw new Error('Failed to set budget');

            alert('Budget set successfully');
            loadBudgetStatus();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
}

async function loadBudgetStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/budget`);
        const data = await response.json();

        const container = document.getElementById('budget-status');
        const budgets = Object.entries(data.budgets);

        if (budgets.length === 0) {
            container.innerHTML = '<p class="text-gray-500 text-center py-8">No budgets configured</p>';
            return;
        }

        container.innerHTML = budgets.map(([name, budget]) => `
            <div class="mb-4 last:mb-0">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-medium text-gray-700">${budget.provider} - ${budget.period}</h3>
                    <span class="text-sm font-semibold">${budget.current_cost.toFixed(4)} / $${budget.limit}</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="h-2 rounded-full ${budget.utilization >= 1.0 ? 'bg-red-600' : budget.utilization >= 0.8 ? 'bg-yellow-600' : 'bg-green-600'}"
                        style="width: ${Math.min(budget.utilization * 100, 100).toFixed(0)}%"></div>
                </div>
                <div class="mt-1 flex justify-between text-xs text-gray-500">
                    <span>Remaining: $${budget.remaining.toFixed(4)}</span>
                    <span>${(budget.utilization * 100).toFixed(0)}% used</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load budget status:', error);
    }
}

async function loadBudgetAlerts() {
    try {
        const response = await fetch(`${API_BASE}/api/budget/alerts`);
        const data = await response.json();

        const container = document.getElementById('budget-alerts');
        if (data.alerts.length === 0) {
            container.innerHTML = '<p class="text-xs text-gray-500">No recent alerts</p>';
            return;
        }

        container.innerHTML = data.alerts.slice(0, 5).map(alert => `
            <div class="flex items-start space-x-2 text-sm ${
                alert.type === 'limit_exceeded' ? 'text-red-600' : 'text-yellow-600'
            }">
                <i class="fas fa-${alert.type === 'limit_exceeded' ? 'exclamation-circle' : 'exclamation-triangle'} mt-0.5"></i>
                <div class="flex-1">
                    <p class="font-medium">${alert.message}</p>
                    <p class="text-xs text-gray-500">${new Date(alert.timestamp).toLocaleString()}</p>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load budget alerts:', error);
    }
}

// Benchmarks
async function loadBenchmarks() {
    try {
        // Load summary
        const summaryResponse = await fetch(`${API_BASE}/api/benchmark/summary`);
        const summaryData = await summaryResponse.json();
        displayBenchmarkSummary(summaryData.summary);

        // Load provider comparison
        const providersResponse = await fetch(`${API_BASE}/api/benchmark/providers`);
        const providersData = await providersResponse.json();
        displayProviderBenchmarks(providersData.providers);
    } catch (error) {
        console.error('Failed to load benchmarks:', error);
    }
}

function displayBenchmarkSummary(summary) {
    const container = document.getElementById('benchmark-summary');
    container.innerHTML = `
        <div class="text-center">
            <div class="text-3xl font-bold text-gray-900">${summary.total_requests || 0}</div>
            <div class="text-sm text-gray-500">Total Requests</div>
        </div>
        <div class="text-center">
            <div class="text-3xl font-bold text-green-600">${((summary.success_rate || 0) * 100).toFixed(1)}%</div>
            <div class="text-sm text-gray-500">Success Rate</div>
        </div>
        <div class="text-center">
            <div class="text-3xl font-bold text-blue-600">${(summary.avg_latency || 0).toFixed(2)}s</div>
            <div class="text-sm text-gray-500">Avg Latency</div>
        </div>
        <div class="text-center">
            <div class="text-3xl font-bold text-purple-600">$${(summary.total_cost || 0).toFixed(4)}</div>
            <div class="text-sm text-gray-500">Total Cost</div>
        </div>
    `;
}

function displayProviderBenchmarks(providers) {
    const tbody = document.getElementById('benchmark-tbody');

    if (Object.keys(providers).length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-8 text-gray-500">No benchmark data available</td></tr>';
        return;
    }

    tbody.innerHTML = Object.entries(providers).map(([name, perf]) => `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${name}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${perf.avg_latency.toFixed(2)}s</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${perf.p95_latency.toFixed(2)}s</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${perf.avg_throughput.toFixed(0)} tok/s</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">$${perf.avg_cost_per_1k_tokens.toFixed(4)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${(perf.success_rate * 100).toFixed(1)}%</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${perf.total_requests}</td>
        </tr>
    `).join('');

    // Update charts
    updateBenchmarkCharts(providers);
}

function updateBenchmarkCharts(providers) {
    const labels = Object.keys(providers);
    const latencies = Object.values(providers).map(p => p.avg_latency);
    const costs = Object.values(providers).map(p => p.avg_cost_per_1k_tokens);

    // Latency chart
    const latencyCtx = document.getElementById('latency-chart');
    if (latencyCtx) {
        new Chart(latencyCtx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Average Latency (seconds)',
                    data: latencies,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    borderColor: 'rgb(59, 130, 246)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    // Cost chart
    const costCtx = document.getElementById('cost-chart');
    if (costCtx) {
        new Chart(costCtx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Cost per 1K Tokens ($)',
                    data: costs,
                    backgroundColor: 'rgba(147, 51, 234, 0.5)',
                    borderColor: 'rgb(147, 51, 234)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
}

// WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function updateConnectionStatus(connected) {
    const status = document.getElementById('connection-status');
    const indicator = status.querySelector('.w-3');
    const text = status.querySelector('span:last-child');

    if (connected) {
        indicator.classList.remove('bg-red-500');
        indicator.classList.add('bg-green-500');
        text.textContent = 'Connected';
    } else {
        indicator.classList.remove('bg-green-500');
        indicator.classList.add('bg-red-500');
        text.textContent = 'Disconnected';
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'status_update':
            // Update connection status
            break;
        case 'session_update':
            // Refresh sessions list
            loadSessions();
            break;
        case 'provider_update':
            // Refresh providers
            loadProviders();
            break;
        case 'budget_alert':
            // Show budget alert
            loadBudgetAlerts();
            break;
    }
}
