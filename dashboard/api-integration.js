/**
 * API Integration for OChem Helper Dashboard
 * 
 * This script provides functions to connect the dashboard to the OChem Helper API
 */

// Configuration
const API_CONFIG = {
    baseURL: 'http://localhost:8000',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json'
    }
};

// API Client
class OChemAPI {
    constructor(config = API_CONFIG) {
        this.config = config;
    }

    async request(endpoint, options = {}) {
        const url = `${this.config.baseURL}${endpoint}`;
        const config = {
            ...options,
            headers: {
                ...this.config.headers,
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    // Generation endpoints
    async generateMolecules(params) {
        return this.request('/api/v1/generate', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    // Prediction endpoints
    async predictProperties(molecules, properties) {
        return this.request('/api/v1/predict/properties', {
            method: 'POST',
            body: JSON.stringify({ molecules, properties })
        });
    }

    async predictADMET(molecule) {
        return this.request('/api/v1/predict/admet', {
            method: 'POST',
            body: JSON.stringify({ molecule })
        });
    }

    // Optimization endpoints
    async optimizeLead(leadSmiles, goals, maintainScaffold = true) {
        return this.request('/api/v1/optimize', {
            method: 'POST',
            body: JSON.stringify({
                lead_smiles: leadSmiles,
                optimization_goals: goals,
                maintain_scaffold: maintainScaffold
            })
        });
    }

    // Synthesis endpoints
    async suggestSynthesis(targetSmiles, maxSteps = 5) {
        return this.request('/api/v1/synthesis/routes', {
            method: 'POST',
            body: JSON.stringify({
                target_smiles: targetSmiles,
                max_steps: maxSteps
            })
        });
    }

    async checkReaction(reactants, products, conditions = null) {
        return this.request('/api/v1/synthesis/predict-reaction', {
            method: 'POST',
            body: JSON.stringify({
                reactants,
                products,
                conditions
            })
        });
    }

    // Analysis endpoints
    async analyzeScaffold(molecules) {
        return this.request('/api/v1/analyze/scaffold', {
            method: 'POST',
            body: JSON.stringify({ molecules })
        });
    }

    async calculateSimilarity(molecule1, molecule2) {
        return this.request('/api/v1/analyze/similarity', {
            method: 'POST',
            body: JSON.stringify({ molecule1, molecule2 })
        });
    }

    // Health check
    async checkHealth() {
        return this.request('/health');
    }
}

// Dashboard Integration Functions
const api = new OChemAPI();

// Update dashboard with API data
async function updateDashboardFromAPI() {
    try {
        // Check API health
        const health = await api.checkHealth();
        updateServiceStatus('api', health.status === 'healthy');

        // You can add more API calls here to populate the dashboard
    } catch (error) {
        console.error('Failed to update dashboard:', error);
        updateServiceStatus('api', false);
    }
}

// Generate molecules with progress updates
async function generateMoleculesWithAPI() {
    const smiles = document.getElementById('smilesInput').value;
    const mode = document.getElementById('modeSelect').value;
    const target = document.getElementById('targetSelect').value;
    const num = parseInt(document.getElementById('numMolecules').value);

    // Show loading state
    showLoading('generation');
    addActivity('API Request', `Sending generation request to server...`);

    try {
        // Prepare parameters based on mode
        let params = { n_molecules: num };
        
        if (mode === 'similar' && smiles) {
            params.reference_smiles = smiles;
        }
        
        if (target === 'drug-like') {
            params.target_properties = {
                logP: [2.0, 5.0],
                MW: [200, 500],
                QED: 0.5
            };
        }

        // Call API
        const result = await api.generateMolecules(params);
        
        // Update dashboard
        if (result.molecules && result.molecules.length > 0) {
            updateGeneratedMolecules(result.molecules);
            addActivity('Generation Complete', `Generated ${result.molecules.length} molecules`);
            
            // Update metrics
            updateMetrics();
            
            // Predict properties for first molecule
            if (result.molecules[0]) {
                updateMoleculeViewer(result.molecules[0]);
                await updatePropertiesChart(result.molecules[0]);
            }
        }
    } catch (error) {
        addActivity('Generation Failed', error.message);
    } finally {
        hideLoading('generation');
    }
}

// Update molecular properties chart
async function updatePropertiesChart(smiles) {
    try {
        const properties = ['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'QED'];
        const result = await api.predictProperties([smiles], properties);
        
        if (result.results && result.results[0]) {
            const props = result.results[0];
            const values = properties.map(p => 
                props[p] ? normalizeProperty(p, props[p]) : 0
            );
            
            // Update radar chart
            Plotly.update('propertyRadar', {
                r: [values]
            });
        }
    } catch (error) {
        console.error('Failed to update properties:', error);
    }
}

// Update ADMET chart
async function updateADMETChart(smiles) {
    try {
        const result = await api.predictADMET(smiles);
        
        if (result.admet_properties) {
            const categories = ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity'];
            const values = categories.map(cat => 
                result.admet_properties[cat]?.probability || 0
            );
            
            Plotly.update('admetChart', {
                y: [values]
            });
        }
    } catch (error) {
        console.error('Failed to update ADMET:', error);
    }
}

// Optimize lead compound
async function optimizeLeadWithAPI() {
    const smiles = document.getElementById('smilesInput').value;
    if (!smiles) {
        alert('Please enter a SMILES string');
        return;
    }

    showLoading('optimization');
    addActivity('Optimization Started', 'Optimizing lead compound...');

    try {
        const goals = {
            logP: [2.0, 4.0],
            MW: [300, 450],
            QED: 0.8
        };

        const result = await api.optimizeLead(smiles, goals);
        
        if (result.optimized_molecules && result.optimized_molecules.length > 0) {
            updateOptimizedMolecules(result.optimized_molecules);
            addActivity('Optimization Complete', 
                `Generated ${result.optimized_molecules.length} optimized molecules`);
        }
    } catch (error) {
        addActivity('Optimization Failed', error.message);
    } finally {
        hideLoading('optimization');
    }
}

// WebSocket connection for real-time updates
function connectWebSocket() {
    const ws = new WebSocket(`ws://localhost:8000/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateServiceStatus('websocket', true);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateServiceStatus('websocket', false);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateServiceStatus('websocket', false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
    
    return ws;
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'generation_progress':
            updateGenerationProgress(data.progress);
            break;
        case 'property_update':
            updatePropertyDisplay(data.properties);
            break;
        case 'activity':
            addActivity(data.title, data.description);
            break;
        default:
            console.log('Unknown WebSocket message:', data);
    }
}

// Utility functions
function normalizeProperty(property, value) {
    const ranges = {
        MW: [0, 1000],
        logP: [-5, 10],
        TPSA: [0, 200],
        HBD: [0, 10],
        HBA: [0, 20],
        QED: [0, 1]
    };
    
    const [min, max] = ranges[property] || [0, 1];
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function updateServiceStatus(service, isActive) {
    const statusDot = document.querySelector(`[data-service="${service}"] .status-dot`);
    if (statusDot) {
        statusDot.classList.toggle('active', isActive);
        statusDot.classList.toggle('processing', !isActive);
    }
}

function showLoading(section) {
    const element = document.getElementById(`${section}Loading`);
    if (element) element.style.display = 'flex';
}

function hideLoading(section) {
    const element = document.getElementById(`${section}Loading`);
    if (element) element.style.display = 'none';
}

// Initialize API integration when dashboard loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if API is available
    api.checkHealth().then(() => {
        console.log('API connection established');
        updateServiceStatus('api', true);
        
        // Start periodic updates
        setInterval(updateDashboardFromAPI, 30000);
        
        // Connect WebSocket
        connectWebSocket();
    }).catch(error => {
        console.error('API not available:', error);
        updateServiceStatus('api', false);
    });
});

// Export for use in dashboard
window.OChemAPI = api;
window.generateMoleculesWithAPI = generateMoleculesWithAPI;
window.optimizeLeadWithAPI = optimizeLeadWithAPI;
window.updatePropertiesChart = updatePropertiesChart;
window.updateADMETChart = updateADMETChart;