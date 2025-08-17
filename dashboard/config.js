/**
 * Configuration for OChem Helper Dashboard
 * 
 * This file contains configuration settings for API endpoints,
 * authentication, and service connections.
 */

const CONFIG = {
    // API Endpoints
    api: {
        ochem: 'http://localhost:8000',      // OChem Helper API
        mcp: 'http://localhost:8001',        // MCP HTTP Bridge
        timeout: 30000
    },

    // Claude API Configuration
    claude: {
        // API key can be set here or via environment
        apiKey: window.CLAUDE_API_KEY || null,
        
        // Model selection
        model: 'claude-3-opus-20240229',    // or 'claude-3-sonnet-20240229'
        
        // Generation parameters
        maxTokens: 1024,
        temperature: 0.7,
        
        // Enable/disable Claude integration
        enabled: true
    },

    // MCP Server Configuration
    mcp: {
        // Enable/disable MCP integration
        enabled: true,
        
        // Retry settings
        maxRetries: 3,
        retryDelay: 1000,
        
        // Tool preferences
        defaultProperties: ['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBD', 'HBA'],
        maxSynthesisSteps: 5,
        optimizationMolecules: 10
    },

    // Dashboard Settings
    dashboard: {
        // Auto-save session data
        autoSave: true,
        autoSaveInterval: 60000, // 1 minute
        
        // Chart update frequency
        chartUpdateInterval: 2000,
        
        // Activity log settings
        maxActivityItems: 50,
        
        // 3D viewer settings
        viewer: {
            backgroundColor: 'rgba(10, 14, 39, 0.7)',
            defaultStyle: 'stick',
            spinSpeed: 0.5
        }
    },

    // AI Chat Settings
    aiChat: {
        // Welcome message
        welcomeMessage: `Hello! I'm your AI Chemistry Assistant powered by Claude. I can help you with:
        • Generating molecules with specific properties
        • Analyzing molecular structures
        • Planning synthesis routes
        • Optimizing lead compounds
        • Predicting ADMET properties
        
        I'm connected to the MCP server for advanced chemistry operations. How can I assist you today?`,
        
        // Suggestion chips
        suggestions: [
            'Generate 10 drug-like molecules',
            'Analyze the current molecule',
            'Suggest synthesis route',
            'Optimize for better solubility',
            'Predict ADMET properties'
        ],
        
        // Chat behavior
        showTypingIndicator: true,
        messageDelay: 500,
        maxHistoryLength: 100
    },

    // Security Settings
    security: {
        // Require API key for Claude
        requireApiKey: true,
        
        // CORS settings
        allowedOrigins: ['http://localhost:*', 'https://localhost:*'],
        
        // Input validation
        maxSmilesLength: 500,
        maxMoleculesPerBatch: 100
    },

    // Development Settings
    development: {
        // Enable debug logging
        debug: true,
        
        // Show performance metrics
        showPerformanceMetrics: true,
        
        // Enable mock data when APIs unavailable
        useMockData: true,
        
        // Log API calls
        logApiCalls: true
    }
};

// Helper function to get nested config values
function getConfig(path, defaultValue = null) {
    const keys = path.split('.');
    let value = CONFIG;
    
    for (const key of keys) {
        if (value && typeof value === 'object' && key in value) {
            value = value[key];
        } else {
            return defaultValue;
        }
    }
    
    return value;
}

// Helper function to update config values
function updateConfig(path, value) {
    const keys = path.split('.');
    let obj = CONFIG;
    
    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        if (!(key in obj)) {
            obj[key] = {};
        }
        obj = obj[key];
    }
    
    obj[keys[keys.length - 1]] = value;
}

// Load config from localStorage if available
function loadStoredConfig() {
    try {
        const stored = localStorage.getItem('ochem_config');
        if (stored) {
            const storedConfig = JSON.parse(stored);
            // Merge with default config
            Object.assign(CONFIG, storedConfig);
        }
    } catch (error) {
        console.error('Failed to load stored config:', error);
    }
}

// Save config to localStorage
function saveConfig() {
    try {
        localStorage.setItem('ochem_config', JSON.stringify(CONFIG));
    } catch (error) {
        console.error('Failed to save config:', error);
    }
}

// API Key management
function setClaudeApiKey(apiKey) {
    CONFIG.claude.apiKey = apiKey;
    localStorage.setItem('claude_api_key', apiKey);
    saveConfig();
}

function getClaudeApiKey() {
    return CONFIG.claude.apiKey || 
           localStorage.getItem('claude_api_key') || 
           window.CLAUDE_API_KEY;
}

// Environment detection
function detectEnvironment() {
    const hostname = window.location.hostname;
    
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'development';
    } else if (hostname.includes('staging')) {
        return 'staging';
    } else {
        return 'production';
    }
}

// Apply environment-specific settings
function applyEnvironmentConfig() {
    const env = detectEnvironment();
    
    if (env === 'production') {
        CONFIG.development.debug = false;
        CONFIG.development.logApiCalls = false;
        CONFIG.security.requireApiKey = true;
    }
}

// Initialize configuration
loadStoredConfig();
applyEnvironmentConfig();

// Export configuration
window.CONFIG = CONFIG;
window.getConfig = getConfig;
window.updateConfig = updateConfig;
window.saveConfig = saveConfig;
window.setClaudeApiKey = setClaudeApiKey;
window.getClaudeApiKey = getClaudeApiKey;

// Log configuration status
if (CONFIG.development.debug) {
    console.log('OChem Helper Configuration loaded:', CONFIG);
    console.log('Environment:', detectEnvironment());
    console.log('Claude API Key:', getClaudeApiKey() ? 'Set' : 'Not set');
}