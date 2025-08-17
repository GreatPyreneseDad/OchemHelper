/**
 * Claude API Integration for OChem Helper
 * 
 * This module handles communication with Claude API for intelligent
 * chemistry assistance and MCP server integration.
 */

class ClaudeAPI {
    constructor(apiKey = null) {
        // Use config if available
        const config = window.CONFIG?.claude || {};
        
        this.apiKey = apiKey || config.apiKey || this.getApiKeyFromEnv();
        this.baseURL = 'https://api.anthropic.com/v1';
        this.model = config.model || 'claude-3-opus-20240229';
        this.maxTokens = config.maxTokens || 1024;
        this.temperature = config.temperature || 0.7;
        
        // Chemistry-specific system prompt
        this.systemPrompt = `You are an expert chemistry AI assistant integrated with the OChem Helper platform. You have access to:
1. Molecular generation and optimization tools
2. Property prediction and ADMET profiling
3. Retrosynthetic analysis capabilities
4. MCP server for advanced chemistry operations

When users ask chemistry questions:
- Analyze their intent (generate, analyze, optimize, synthesize, predict)
- Use the available tools through the automation bridge
- Provide clear, actionable responses
- Include relevant molecular properties and safety information
- Suggest next steps in the discovery workflow

You can control the dashboard through window.automationBridge.executeAction() calls.
Available actions: ${this.getAvailableActions()}`;
    }

    getApiKeyFromEnv() {
        // Try to get from various sources
        return window.CLAUDE_API_KEY || 
               localStorage.getItem('claude_api_key') || 
               null;
    }

    checkApiKey() {
        if (!this.apiKey) {
            // Redirect to setup page if no API key
            if (confirm('Claude API key not found. Would you like to set it up now?')) {
                window.location.href = 'setup-api-key.html';
            }
            return false;
        }
        return true;
    }

    getAvailableActions() {
        if (window.automationBridge) {
            return window.automationBridge.getAvailableActions()
                .map(a => `${a.name}: ${a.description}`)
                .join(', ');
        }
        return 'Dashboard automation not yet loaded';
    }

    async sendMessage(message, context = {}) {
        if (!this.checkApiKey()) {
            throw new Error('Claude API key not configured');
        }

        try {
            const response = await fetch(`${this.baseURL}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.apiKey,
                    'anthropic-version': '2023-06-01'
                },
                body: JSON.stringify({
                    model: this.model,
                    max_tokens: this.maxTokens,
                    temperature: this.temperature,
                    system: this.systemPrompt,
                    messages: [
                        {
                            role: 'user',
                            content: this.buildPrompt(message, context)
                        }
                    ]
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Claude API error: ${error.error?.message || response.statusText}`);
            }

            const data = await response.json();
            return this.parseResponse(data.content[0].text);
        } catch (error) {
            console.error('Claude API error:', error);
            throw error;
        }
    }

    buildPrompt(message, context) {
        let prompt = message;

        // Add context about current state
        if (context.currentMolecule) {
            prompt += `\n\nCurrent molecule: ${context.currentMolecule}`;
        }
        if (context.recentActivity) {
            prompt += `\n\nRecent activities: ${JSON.stringify(context.recentActivity)}`;
        }

        // Add information about dashboard state
        prompt += `\n\nDashboard state: ${JSON.stringify(context)}`;
        
        // Remind about available tools
        prompt += `\n\nRemember: You can execute actions using JavaScript calls like:
- await window.automationBridge.executeAction('setInputSMILES', 'CCO')
- await window.automationBridge.executeAction('generateMolecules')
- await window.automationBridge.executeAction('analyzeCurrentMolecule')

Respond with clear text and any JavaScript commands that should be executed.`;

        return prompt;
    }

    parseResponse(responseText) {
        // Extract JavaScript commands and regular text
        const jsPattern = /```javascript\n([\s\S]*?)\n```/g;
        const commands = [];
        let textResponse = responseText;

        let match;
        while ((match = jsPattern.exec(responseText)) !== null) {
            commands.push(match[1]);
            textResponse = textResponse.replace(match[0], '');
        }

        // Also look for inline await commands
        const inlinePattern = /await window\.[^\n;]+[;\n]/g;
        while ((match = inlinePattern.exec(textResponse)) !== null) {
            commands.push(match[0]);
            textResponse = textResponse.replace(match[0], '');
        }

        return {
            message: textResponse.trim(),
            commands: commands,
            actions: this.extractActions(responseText)
        };
    }

    extractActions(responseText) {
        const actions = [];
        
        // Look for specific action patterns
        if (/generate.*molecules?/i.test(responseText)) {
            actions.push({
                type: 'button',
                label: 'Generate Molecules',
                action: 'generateMolecules'
            });
        }
        
        if (/optimize.*lead|improve.*structure/i.test(responseText)) {
            actions.push({
                type: 'button',
                label: 'Optimize Structure',
                action: 'optimizeLead'
            });
        }
        
        if (/predict.*properties|analyze.*molecule/i.test(responseText)) {
            actions.push({
                type: 'button',
                label: 'Analyze Properties',
                action: 'analyzeCurrentMolecule'
            });
        }

        return actions;
    }

    async executeCommands(commands) {
        const results = [];
        
        for (const command of commands) {
            try {
                // Create a safe execution environment
                const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
                const fn = new AsyncFunction('window', command);
                const result = await fn(window);
                results.push({ command, result, success: true });
            } catch (error) {
                results.push({ command, error: error.message, success: false });
            }
        }
        
        return results;
    }

    updateSettings(settings) {
        if (settings.apiKey) this.apiKey = settings.apiKey;
        if (settings.model) this.model = settings.model;
        if (settings.maxTokens) this.maxTokens = settings.maxTokens;
        if (settings.temperature) this.temperature = settings.temperature;
    }
}

// MCP Server Connection
class MCPConnection {
    constructor(serverUrl = null) {
        const config = window.CONFIG?.mcp || {};
        this.serverUrl = serverUrl || window.CONFIG?.api?.mcp || 'http://localhost:8001';
        this.connected = false;
        this.tools = [];
        this.config = config;
    }

    async connect() {
        try {
            const response = await fetch(`${this.serverUrl}/list_tools`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            if (!response.ok) {
                throw new Error(`MCP server error: ${response.statusText}`);
            }

            const data = await response.json();
            this.tools = data.tools || [];
            this.connected = true;
            console.log('Connected to MCP server. Available tools:', this.tools.map(t => t.name));
            return true;
        } catch (error) {
            console.error('Failed to connect to MCP server:', error);
            this.connected = false;
            return false;
        }
    }

    async callTool(toolName, params) {
        if (!this.connected) {
            await this.connect();
        }

        try {
            const response = await fetch(`${this.serverUrl}/call_tool`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tool_name: toolName,
                    arguments: params
                })
            });

            if (!response.ok) {
                throw new Error(`MCP tool error: ${response.statusText}`);
            }

            const data = await response.json();
            return data.result;
        } catch (error) {
            console.error(`Failed to call MCP tool ${toolName}:`, error);
            throw error;
        }
    }

    // Chemistry-specific MCP tools
    async predictProperties(smiles, properties = null) {
        return this.callTool('predict_properties', {
            smiles: smiles,
            properties: properties || ['MW', 'logP', 'TPSA', 'QED', 'SA']
        });
    }

    async suggestSynthesis(targetSmiles, maxSteps = 5) {
        return this.callTool('suggest_synthesis', {
            target_smiles: targetSmiles,
            max_steps: maxSteps
        });
    }

    async optimizeStructure(leadSmiles, optimizationGoals, numMolecules = 10) {
        return this.callTool('optimize_structure', {
            lead_smiles: leadSmiles,
            optimization_goals: optimizationGoals,
            num_molecules: numMolecules
        });
    }

    async checkReactionFeasibility(reactants, products) {
        return this.callTool('reaction_prediction', {
            reactants: reactants,
            products: products
        });
    }
}

// Enhanced AI Chemistry Agent with Claude + MCP
class EnhancedAIChemistryAgent {
    constructor() {
        this.claudeAPI = new ClaudeAPI();
        this.mcpConnection = new MCPConnection();
        this.originalAgent = window.aiAgent;
        this.setupEnhancements();
    }

    async setupEnhancements() {
        // Try to connect to MCP server
        const mcpConnected = await this.mcpConnection.connect();
        
        if (mcpConnected && this.originalAgent) {
            this.originalAgent.mcpConnected = true;
            this.originalAgent.addMessage(
                'MCP server connected! I now have access to advanced chemistry tools.',
                'assistant'
            );
        }

        // Override the processMessage method
        if (this.originalAgent) {
            this.originalAgent.processMessage = this.processMessage.bind(this);
            this.originalAgent.queryMCPServer = this.queryMCPServer.bind(this);
        }
    }

    async processMessage(message) {
        const intent = this.originalAgent.analyzeIntent(message);
        const context = this.originalAgent.getCurrentContext();

        try {
            // First, try to get a response from Claude
            const claudeResponse = await this.claudeAPI.sendMessage(message, context);
            
            // Execute any commands Claude suggested
            if (claudeResponse.commands && claudeResponse.commands.length > 0) {
                const commandResults = await this.claudeAPI.executeCommands(claudeResponse.commands);
                console.log('Executed Claude commands:', commandResults);
            }

            // If Claude suggested using MCP tools, use them
            if (claudeResponse.message.includes('MCP:')) {
                const mcpResults = await this.handleMCPRequest(claudeResponse.message, context);
                claudeResponse.message += '\n\n' + mcpResults;
            }

            return {
                message: claudeResponse.message,
                actions: claudeResponse.actions || []
            };
        } catch (error) {
            console.error('Error processing with Claude:', error);
            
            // Fallback to original intent-based processing
            return this.fallbackProcessing(intent, context, message);
        }
    }

    async handleMCPRequest(message, context) {
        const results = [];

        try {
            // Property prediction
            if (message.includes('MCP:predict_properties')) {
                const smiles = context.currentMolecule;
                if (smiles) {
                    const props = await this.mcpConnection.predictProperties(smiles);
                    results.push(`Properties predicted: ${JSON.stringify(props)}`);
                }
            }

            // Synthesis suggestion
            if (message.includes('MCP:suggest_synthesis')) {
                const smiles = context.currentMolecule;
                if (smiles) {
                    const routes = await this.mcpConnection.suggestSynthesis(smiles);
                    results.push(`Found ${routes.length} synthesis routes`);
                }
            }

            // Structure optimization
            if (message.includes('MCP:optimize_structure')) {
                const smiles = context.currentMolecule;
                if (smiles) {
                    const optimized = await this.mcpConnection.optimizeStructure(smiles, {
                        MW: [200, 500],
                        logP: [1, 5],
                        QED: 0.7
                    });
                    results.push(`Generated ${optimized.length} optimized structures`);
                }
            }
        } catch (error) {
            results.push(`MCP error: ${error.message}`);
        }

        return results.join('\n');
    }

    async queryMCPServer(message, context) {
        try {
            // Use Claude API for intelligent responses
            const response = await this.claudeAPI.sendMessage(message, context);
            
            return {
                answer: response.message,
                actions: response.actions || []
            };
        } catch (error) {
            // Fallback response
            return {
                answer: "I'm having trouble connecting to the AI service. Please check your API configuration.",
                actions: []
            };
        }
    }

    async fallbackProcessing(intent, context, message) {
        // Use original agent's intent handlers as fallback
        switch (intent.type) {
            case 'generate':
                return this.originalAgent.handleGenerateIntent(intent, context);
            case 'analyze':
                return this.originalAgent.handleAnalyzeIntent(intent, context);
            case 'optimize':
                return this.originalAgent.handleOptimizeIntent(intent, context);
            case 'synthesize':
                return this.originalAgent.handleSynthesisIntent(intent, context);
            case 'predict':
                return this.originalAgent.handlePredictIntent(intent, context);
            case 'help':
                return this.originalAgent.handleHelpIntent(intent);
            default:
                return {
                    message: "I understand you're asking about chemistry. Could you be more specific?",
                    actions: []
                };
        }
    }
}

// Initialize enhanced agent when ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for original agent to load
    setTimeout(() => {
        if (window.aiAgent) {
            window.enhancedAgent = new EnhancedAIChemistryAgent();
            console.log('Enhanced AI Chemistry Agent with Claude + MCP initialized');
        }
    }, 1000);
});

// Export for use
window.ClaudeAPI = ClaudeAPI;
window.MCPConnection = MCPConnection;
window.EnhancedAIChemistryAgent = EnhancedAIChemistryAgent;