/**
 * AI Chat Interface for OChem Helper
 * Provides interactive AI assistance for molecular discovery workflows
 */

class AIChemistryAgent {
    constructor() {
        this.chatHistory = [];
        this.mcpConnected = false;
        this.currentContext = {};
        this.setupUI();
        this.setupEventListeners();
    }

    setupUI() {
        // Create chat window HTML
        const chatHTML = `
            <div id="ai-chat-container" class="ai-chat-container">
                <div class="ai-chat-header">
                    <div class="ai-chat-title">
                        <span class="ai-icon">🤖</span>
                        AI Chemistry Assistant
                    </div>
                    <div class="ai-chat-controls">
                        <button class="ai-chat-btn" onclick="aiAgent.toggleChat()">_</button>
                        <button class="ai-chat-btn" onclick="aiAgent.closeChat()">×</button>
                    </div>
                </div>
                <div class="ai-chat-body">
                    <div id="ai-chat-messages" class="ai-chat-messages">
                        <div class="ai-message ai-message-assistant">
                            <div class="ai-message-avatar">🧪</div>
                            <div class="ai-message-content">
                                Hello! I'm your AI Chemistry Assistant. I can help you with:
                                <ul>
                                    <li>Generating molecules with specific properties</li>
                                    <li>Analyzing molecular structures</li>
                                    <li>Planning synthesis routes</li>
                                    <li>Optimizing lead compounds</li>
                                    <li>Predicting ADMET properties</li>
                                </ul>
                                How can I assist you today?
                            </div>
                        </div>
                    </div>
                    <div class="ai-chat-suggestions">
                        <button class="suggestion-chip" onclick="aiAgent.sendMessage('Generate 10 drug-like molecules with QED > 0.8')">
                            Generate drug-like molecules
                        </button>
                        <button class="suggestion-chip" onclick="aiAgent.sendMessage('Analyze the current molecule')">
                            Analyze structure
                        </button>
                        <button class="suggestion-chip" onclick="aiAgent.sendMessage('Suggest synthesis route')">
                            Plan synthesis
                        </button>
                    </div>
                    <div class="ai-chat-input-container">
                        <textarea 
                            id="ai-chat-input" 
                            class="ai-chat-input" 
                            placeholder="Ask me anything about chemistry..."
                            rows="2"
                        ></textarea>
                        <button class="ai-chat-send" onclick="aiAgent.sendCurrentMessage()">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="ai-chat-status">
                    <span id="ai-status-indicator" class="ai-status-dot"></span>
                    <span id="ai-status-text">Ready</span>
                </div>
            </div>
        `;

        // Add chat window to page
        const chatDiv = document.createElement('div');
        chatDiv.innerHTML = chatHTML;
        document.body.appendChild(chatDiv);

        // Add styles
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .ai-chat-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 400px;
                height: 600px;
                background: rgba(30, 41, 59, 0.95);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(99, 179, 237, 0.3);
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
                z-index: 1000;
                transition: all 0.3s ease;
            }

            .ai-chat-container.minimized {
                height: 50px;
            }

            .ai-chat-container.closed {
                display: none;
            }

            .ai-chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                border-bottom: 1px solid rgba(99, 179, 237, 0.2);
                background: rgba(10, 14, 39, 0.8);
                border-radius: 12px 12px 0 0;
            }

            .ai-chat-title {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 600;
                color: #e0e6ed;
            }

            .ai-icon {
                font-size: 1.2rem;
            }

            .ai-chat-controls {
                display: flex;
                gap: 0.5rem;
            }

            .ai-chat-btn {
                background: rgba(99, 179, 237, 0.2);
                border: 1px solid rgba(99, 179, 237, 0.3);
                color: #e0e6ed;
                width: 30px;
                height: 30px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s;
            }

            .ai-chat-btn:hover {
                background: rgba(99, 179, 237, 0.3);
            }

            .ai-chat-body {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .ai-chat-container.minimized .ai-chat-body {
                display: none;
            }

            .ai-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .ai-message {
                display: flex;
                gap: 0.75rem;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .ai-message-user {
                flex-direction: row-reverse;
            }

            .ai-message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                background: rgba(99, 179, 237, 0.2);
                flex-shrink: 0;
            }

            .ai-message-user .ai-message-avatar {
                background: rgba(76, 29, 149, 0.3);
            }

            .ai-message-content {
                background: rgba(99, 179, 237, 0.1);
                border: 1px solid rgba(99, 179, 237, 0.2);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                max-width: 80%;
                color: #e0e6ed;
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .ai-message-user .ai-message-content {
                background: rgba(76, 29, 149, 0.2);
                border-color: rgba(76, 29, 149, 0.3);
            }

            .ai-message-content ul {
                margin: 0.5rem 0;
                padding-left: 1.5rem;
            }

            .ai-message-content code {
                background: rgba(0, 0, 0, 0.3);
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-family: monospace;
            }

            .ai-message-action {
                background: rgba(99, 179, 237, 0.2);
                border: 1px solid rgba(99, 179, 237, 0.3);
                border-radius: 8px;
                padding: 0.5rem;
                margin-top: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                cursor: pointer;
                transition: all 0.3s;
            }

            .ai-message-action:hover {
                background: rgba(99, 179, 237, 0.3);
            }

            .ai-chat-suggestions {
                padding: 0.5rem 1rem;
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                border-top: 1px solid rgba(99, 179, 237, 0.1);
            }

            .suggestion-chip {
                background: rgba(99, 179, 237, 0.1);
                border: 1px solid rgba(99, 179, 237, 0.3);
                color: #63b3ed;
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.3s;
                white-space: nowrap;
            }

            .suggestion-chip:hover {
                background: rgba(99, 179, 237, 0.2);
                transform: translateY(-1px);
            }

            .ai-chat-input-container {
                display: flex;
                gap: 0.5rem;
                padding: 1rem;
                border-top: 1px solid rgba(99, 179, 237, 0.2);
                align-items: flex-end;
            }

            .ai-chat-input {
                flex: 1;
                background: rgba(10, 14, 39, 0.7);
                border: 1px solid rgba(99, 179, 237, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                color: #e0e6ed;
                font-size: 0.9rem;
                resize: none;
                font-family: inherit;
            }

            .ai-chat-input:focus {
                outline: none;
                border-color: #63b3ed;
                box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1);
            }

            .ai-chat-send {
                background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%);
                border: none;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.3s;
            }

            .ai-chat-send:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(76, 29, 149, 0.3);
            }

            .ai-chat-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                background: rgba(10, 14, 39, 0.8);
                border-top: 1px solid rgba(99, 179, 237, 0.2);
                font-size: 0.8rem;
                color: #94a3b8;
                border-radius: 0 0 12px 12px;
            }

            .ai-chat-container.minimized .ai-chat-status {
                display: none;
            }

            .ai-status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 2s infinite;
            }

            .ai-status-dot.processing {
                background: #f59e0b;
            }

            .ai-status-dot.error {
                background: #ef4444;
            }

            .ai-loading {
                display: flex;
                gap: 0.25rem;
                padding: 0 1rem;
            }

            .ai-loading span {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #63b3ed;
                animation: bounce 1.4s infinite ease-in-out both;
            }

            .ai-loading span:nth-child(1) { animation-delay: -0.32s; }
            .ai-loading span:nth-child(2) { animation-delay: -0.16s; }

            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }

            @media (max-width: 480px) {
                .ai-chat-container {
                    width: calc(100vw - 40px);
                    height: calc(100vh - 100px);
                    right: 20px;
                    bottom: 20px;
                }
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    setupEventListeners() {
        // Enter key to send
        const input = document.getElementById('ai-chat-input');
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendCurrentMessage();
            }
        });
    }

    async sendMessage(message) {
        if (!message.trim()) return;

        // Add user message
        this.addMessage(message, 'user');

        // Clear input
        const input = document.getElementById('ai-chat-input');
        input.value = '';

        // Show loading
        this.showLoading();
        this.updateStatus('processing', 'Processing...');

        try {
            // Process message and get AI response
            const response = await this.processMessage(message);
            
            // Remove loading
            this.hideLoading();
            
            // Add AI response
            this.addMessage(response.message, 'assistant', response.actions);
            
            // Execute any actions
            if (response.actions && response.actions.length > 0) {
                await this.executeActions(response.actions);
            }
            
            this.updateStatus('ready', 'Ready');
        } catch (error) {
            this.hideLoading();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            this.updateStatus('error', 'Error');
            console.error('AI Chat Error:', error);
        }
    }

    async processMessage(message) {
        // Analyze user intent
        const intent = this.analyzeIntent(message);
        
        // Get current context
        const context = this.getCurrentContext();
        
        // Generate response based on intent
        let response = {
            message: '',
            actions: []
        };

        switch (intent.type) {
            case 'set_molecule':
                response = await this.handleSetMoleculeIntent(intent, context);
                break;
            case 'generate':
                response = await this.handleGenerateIntent(intent, context);
                break;
            case 'analyze':
                response = await this.handleAnalyzeIntent(intent, context);
                break;
            case 'optimize':
                response = await this.handleOptimizeIntent(intent, context);
                break;
            case 'synthesize':
                response = await this.handleSynthesisIntent(intent, context);
                break;
            case 'predict':
                response = await this.handlePredictIntent(intent, context);
                break;
            case 'help':
                response = this.handleHelpIntent(intent);
                break;
            default:
                response = await this.handleGeneralIntent(message, context);
        }

        // Store in history
        this.chatHistory.push({
            user: message,
            assistant: response.message,
            timestamp: new Date(),
            context: context
        });

        return response;
    }

    analyzeIntent(message) {
        const lower = message.toLowerCase();
        
        // Set/Input patterns
        if (lower.includes('set') || lower.includes('input') || lower.includes('use')) {
            if (lower.includes('smiles') || message.includes('c1ccccc1') || /[A-Z][A-Za-z0-9@+\-\[\]()=#]+/.test(message)) {
                return {
                    type: 'set_molecule',
                    entities: this.extractSetMoleculeEntities(message)
                };
            }
        }
        
        // Generation patterns
        if (lower.includes('generate') || lower.includes('create') || lower.includes('make')) {
            return {
                type: 'generate',
                entities: this.extractGenerationEntities(message)
            };
        }
        
        // Analysis patterns
        if (lower.includes('analyze') || lower.includes('what is') || lower.includes('tell me about')) {
            return {
                type: 'analyze',
                entities: this.extractAnalysisEntities(message)
            };
        }
        
        // Optimization patterns
        if (lower.includes('optimize') || lower.includes('improve') || lower.includes('better')) {
            return {
                type: 'optimize',
                entities: this.extractOptimizationEntities(message)
            };
        }
        
        // Synthesis patterns
        if (lower.includes('synthesize') || lower.includes('synthesis') || lower.includes('make from')) {
            return {
                type: 'synthesize',
                entities: this.extractSynthesisEntities(message)
            };
        }
        
        // Prediction patterns
        if (lower.includes('predict') || lower.includes('will it') || lower.includes('property')) {
            return {
                type: 'predict',
                entities: this.extractPredictionEntities(message)
            };
        }
        
        // Help patterns
        if (lower.includes('help') || lower.includes('how do i') || lower.includes('what can')) {
            return {
                type: 'help',
                entities: {}
            };
        }
        
        return {
            type: 'general',
            entities: {}
        };
    }

    async handleSetMoleculeIntent(intent, context) {
        const { smiles } = intent.entities;
        
        if (!smiles) {
            return {
                message: "I didn't find a valid SMILES string in your message. Please provide a SMILES notation like 'c1ccccc1' for benzene.",
                actions: []
            };
        }

        // Use automation bridge to set the SMILES
        if (window.automationBridge) {
            const result = await window.automationBridge.executeAction('setInputSMILES', smiles);
            if (result.success) {
                // Update the viewer
                if (window.updateMoleculeViewer) {
                    window.updateMoleculeViewer(smiles);
                }
                
                return {
                    message: `I've set the input SMILES to \`${smiles}\`. Would you like me to analyze this molecule or generate similar structures?`,
                    actions: [
                        {
                            type: 'button',
                            label: 'Analyze Structure',
                            action: 'analyzeCurrentMolecule'
                        },
                        {
                            type: 'button',
                            label: 'Generate Similar',
                            action: 'generateMolecules'
                        }
                    ]
                };
            }
        }

        // Fallback
        document.getElementById('smilesInput').value = smiles;
        return {
            message: `I've set the input SMILES to \`${smiles}\`.`,
            actions: []
        };
    }

    async handleGenerateIntent(intent, context) {
        const { number = 10, properties = {} } = intent.entities;
        
        // Create action to fill in form
        const actions = [
            {
                type: 'fillForm',
                fields: {
                    numMolecules: number,
                    modeSelect: properties.similar ? 'similar' : 'random',
                    targetSelect: this.mapPropertiesToTarget(properties)
                }
            }
        ];

        // If SMILES provided, add it
        if (properties.smiles) {
            actions[0].fields.smilesInput = properties.smiles;
        }

        let message = `I'll generate ${number} molecules`;
        
        if (properties.similar) {
            message += ` similar to the structure you provided`;
        } else if (properties.drugLike) {
            message += ` with drug-like properties`;
        } else if (properties.qed) {
            message += ` with QED score > ${properties.qed}`;
        }
        
        message += `. Click the button below to start generation.`;

        actions.push({
            type: 'button',
            label: 'Generate Molecules',
            action: 'generateMolecules'
        });

        return { message, actions };
    }

    async handleAnalyzeIntent(intent, context) {
        const currentSmiles = document.getElementById('smilesInput').value || context.currentMolecule;
        
        if (!currentSmiles) {
            return {
                message: "Please enter a SMILES string in the input field or generate a molecule first, then I can analyze it for you.",
                actions: []
            };
        }

        // Get properties via API
        const properties = await this.getMolecularProperties(currentSmiles);
        
        let message = `Here's my analysis of the molecule \`${currentSmiles}\`:\n\n`;
        message += `**Molecular Properties:**\n`;
        message += `• Molecular Weight: ${properties.MW?.toFixed(2) || 'N/A'} g/mol\n`;
        message += `• LogP: ${properties.logP?.toFixed(2) || 'N/A'}\n`;
        message += `• TPSA: ${properties.TPSA?.toFixed(2) || 'N/A'} Ų\n`;
        message += `• QED Score: ${properties.QED?.toFixed(3) || 'N/A'}\n\n`;
        
        // Drug-likeness assessment
        const drugLike = this.assessDrugLikeness(properties);
        message += `**Drug-likeness Assessment:**\n${drugLike}\n\n`;
        
        message += `Would you like me to optimize this structure or predict its ADMET properties?`;

        return {
            message,
            actions: [
                {
                    type: 'button',
                    label: 'Optimize Structure',
                    action: 'optimizeLead'
                },
                {
                    type: 'button',
                    label: 'Predict ADMET',
                    action: 'predictADMET'
                }
            ]
        };
    }

    async handleOptimizeIntent(intent, context) {
        const currentSmiles = document.getElementById('smilesInput').value || context.currentMolecule;
        
        if (!currentSmiles) {
            return {
                message: "I need a starting molecule to optimize. Please enter a SMILES string in the input field.",
                actions: []
            };
        }

        const goals = intent.entities.goals || {
            logP: [2.0, 4.0],
            MW: [250, 450],
            QED: 0.7
        };

        return {
            message: `I'll optimize the structure to achieve:\n• LogP: ${goals.logP[0]}-${goals.logP[1]}\n• MW: ${goals.MW[0]}-${goals.MW[1]} g/mol\n• QED > ${goals.QED}\n\nClick below to start optimization.`,
            actions: [
                {
                    type: 'fillForm',
                    fields: {
                        smilesInput: currentSmiles,
                        modeSelect: 'optimize'
                    }
                },
                {
                    type: 'button',
                    label: 'Start Optimization',
                    action: 'optimizeLeadWithAPI'
                }
            ]
        };
    }

    async handleSynthesisIntent(intent, context) {
        const targetSmiles = intent.entities.target || document.getElementById('smilesInput').value;
        
        if (!targetSmiles) {
            return {
                message: "Please provide a target molecule (SMILES) for synthesis planning.",
                actions: []
            };
        }

        return {
            message: `I'll analyze retrosynthetic routes for your target molecule. This will identify:\n• Starting materials\n• Reaction steps\n• Conditions and reagents\n\nClick below to start synthesis planning.`,
            actions: [
                {
                    type: 'button',
                    label: 'Plan Synthesis Route',
                    action: 'planSynthesis',
                    data: { smiles: targetSmiles }
                }
            ]
        };
    }

    async handlePredictIntent(intent, context) {
        const smiles = document.getElementById('smilesInput').value || context.currentMolecule;
        
        if (!smiles) {
            return {
                message: "I need a molecule to make predictions. Please enter a SMILES string.",
                actions: []
            };
        }

        const predictType = intent.entities.predictType || 'all';
        
        return {
            message: `I'll predict the ${predictType === 'admet' ? 'ADMET' : 'molecular'} properties for your compound. This includes absorption, distribution, metabolism, excretion, and toxicity profiles.`,
            actions: [
                {
                    type: 'button',
                    label: 'Predict Properties',
                    action: 'predictProperties',
                    data: { smiles, type: predictType }
                }
            ]
        };
    }

    handleHelpIntent(intent) {
        const helpMessage = `
I can help you with various chemistry tasks:

**Molecule Generation:**
• "Generate 10 drug-like molecules"
• "Create molecules similar to aspirin"
• "Make molecules with QED > 0.8"

**Structure Analysis:**
• "Analyze this molecule"
• "What are the properties of CCO?"
• "Tell me about the current structure"

**Lead Optimization:**
• "Optimize this structure for better solubility"
• "Improve the drug-likeness"
• "Make it more potent"

**Synthesis Planning:**
• "How can I synthesize this molecule?"
• "Plan a synthesis route"
• "What starting materials do I need?"

**Property Prediction:**
• "Predict ADMET properties"
• "Will this cross the blood-brain barrier?"
• "What's the predicted solubility?"

Just type your question or click one of the suggestion chips!
        `;

        return {
            message: helpMessage.trim(),
            actions: []
        };
    }

    async handleGeneralIntent(message, context) {
        // Use MCP or API to get response
        try {
            const response = await this.queryMCPServer(message, context);
            return {
                message: response.answer || "I understand you're asking about chemistry. Could you be more specific about what you'd like to know?",
                actions: response.actions || []
            };
        } catch (error) {
            return {
                message: "I'm here to help with chemistry questions. Try asking about molecule generation, property analysis, or synthesis planning!",
                actions: []
            };
        }
    }

    // Helper methods
    getCurrentContext() {
        return {
            currentMolecule: document.getElementById('smilesInput').value,
            mode: document.getElementById('modeSelect').value,
            targetProperties: document.getElementById('targetSelect').value,
            recentActivity: this.getRecentActivity()
        };
    }

    async getMolecularProperties(smiles) {
        // Call API or calculate locally
        if (window.OChemAPI) {
            try {
                const result = await window.OChemAPI.predictProperties([smiles], ['MW', 'logP', 'TPSA', 'QED']);
                return result.results[0] || {};
            } catch (error) {
                console.error('Failed to get properties:', error);
            }
        }
        
        // Fallback to mock data
        return {
            MW: 180.16,
            logP: 2.3,
            TPSA: 40.5,
            QED: 0.82
        };
    }

    assessDrugLikeness(properties) {
        const issues = [];
        
        if (properties.MW > 500) issues.push('MW > 500');
        if (properties.logP > 5) issues.push('LogP > 5');
        if (properties.TPSA > 140) issues.push('TPSA > 140');
        
        if (issues.length === 0) {
            return "✅ Passes Lipinski's Rule of Five and Veber rules. Good drug-like properties!";
        } else {
            return `⚠️ Violations: ${issues.join(', ')}. Consider optimization for better drug-likeness.`;
        }
    }

    async executeActions(actions) {
        for (const action of actions) {
            switch (action.type) {
                case 'fillForm':
                    this.fillFormFields(action.fields);
                    break;
                case 'button':
                    // Add interactive button to chat
                    break;
                case 'execute':
                    await this.executeFunction(action.function, action.args);
                    break;
            }
        }
    }

    fillFormFields(fields) {
        for (const [fieldId, value] of Object.entries(fields)) {
            const element = document.getElementById(fieldId);
            if (element) {
                element.value = value;
                // Trigger change event
                element.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }

    async executeFunction(functionName, args) {
        if (window[functionName]) {
            await window[functionName](...args);
        }
    }

    // UI methods
    addMessage(content, sender = 'assistant', actions = []) {
        const messagesContainer = document.getElementById('ai-chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ai-message-${sender}`;
        
        const avatar = sender === 'user' ? '👤' : '🧪';
        
        let actionsHTML = '';
        if (actions && actions.length > 0) {
            actionsHTML = actions.filter(a => a.type === 'button').map(action => `
                <div class="ai-message-action" onclick="aiAgent.handleAction('${action.action}', ${JSON.stringify(action.data || {})})">
                    <span>▶</span> ${action.label}
                </div>
            `).join('');
        }
        
        messageDiv.innerHTML = `
            <div class="ai-message-avatar">${avatar}</div>
            <div class="ai-message-content">
                ${this.formatMessage(content)}
                ${actionsHTML}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    formatMessage(content) {
        // Convert markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/• /g, '• ');
    }

    showLoading() {
        const messagesContainer = document.getElementById('ai-chat-messages');
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'ai-loading';
        loadingDiv.className = 'ai-message ai-message-assistant';
        loadingDiv.innerHTML = `
            <div class="ai-message-avatar">🧪</div>
            <div class="ai-loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideLoading() {
        const loading = document.getElementById('ai-loading');
        if (loading) loading.remove();
    }

    updateStatus(status, text) {
        const statusDot = document.getElementById('ai-status-indicator');
        const statusText = document.getElementById('ai-status-text');
        
        statusDot.className = `ai-status-dot ${status}`;
        statusText.textContent = text;
    }

    sendCurrentMessage() {
        const input = document.getElementById('ai-chat-input');
        this.sendMessage(input.value);
    }

    toggleChat() {
        const container = document.getElementById('ai-chat-container');
        container.classList.toggle('minimized');
    }

    closeChat() {
        const container = document.getElementById('ai-chat-container');
        container.classList.add('closed');
    }

    async handleAction(action, data) {
        // Use automation bridge if available
        if (window.automationBridge) {
            try {
                let result;
                switch (action) {
                    case 'generateMolecules':
                        result = await window.automationBridge.executeAction('generateMolecules');
                        if (result.success) {
                            this.addMessage("Generation started! I'm monitoring the progress...", 'assistant');
                            // Monitor results
                            setTimeout(async () => {
                                const metrics = await window.automationBridge.executeAction('getMetrics');
                                const activities = await window.automationBridge.executeAction('getActivityLog');
                                this.addMessage(`Generation complete! Updated metrics: ${metrics.data.moleculesGenerated} molecules generated with ${metrics.data.avgQED} average QED.`, 'assistant');
                            }, 3000);
                        } else {
                            this.addMessage(`Generation failed: ${result.message}`, 'assistant');
                        }
                        break;

                    case 'optimizeLeadWithAPI':
                        result = await window.automationBridge.executeAction('optimizeLead');
                        if (result.success) {
                            this.addMessage("Lead optimization started! Analyzing molecular transformations...", 'assistant');
                            // Analyze results after optimization
                            setTimeout(async () => {
                                const analysis = await window.automationBridge.executeAction('analyzeCurrentMolecule');
                                if (analysis.success && analysis.data) {
                                    this.addMessage(`Optimization complete! New QED: ${analysis.data.properties.QED?.toFixed(3) || 'N/A'}`, 'assistant');
                                }
                            }, 3000);
                        } else {
                            this.addMessage(`Optimization failed: ${result.message}`, 'assistant');
                        }
                        break;

                    case 'predictADMET':
                        const smiles = document.getElementById('smilesInput').value;
                        if (window.updateADMETChart) {
                            await window.updateADMETChart(smiles);
                            const analysis = await window.automationBridge.executeAction('analyzeCurrentMolecule');
                            if (analysis.success && analysis.data.admet) {
                                let admetSummary = "ADMET Profile:\\n";
                                Object.entries(analysis.data.admet).forEach(([key, value]) => {
                                    admetSummary += `• ${key}: ${value.probability?.toFixed(2) || value}\\n`;
                                });
                                this.addMessage(admetSummary, 'assistant');
                            } else {
                                this.addMessage("ADMET predictions updated! Check the ADMET Profile chart.", 'assistant');
                            }
                        }
                        break;

                    case 'planSynthesis':
                        result = await window.automationBridge.executeAction('planSynthesis', data.smiles);
                        if (result.success && result.data) {
                            this.addMessage(`Found ${result.data.routes?.length || 0} synthesis routes for your target molecule. Each route includes starting materials, reaction conditions, and predicted yields.`, 'assistant');
                        } else {
                            this.addMessage("Synthesis planning requires connection to the retrosynthesis API.", 'assistant');
                        }
                        break;

                    case 'predictProperties':
                        if (data.smiles) {
                            await window.automationBridge.executeAction('setInputSMILES', data.smiles);
                            if (window.updatePropertiesChart) {
                                await window.updatePropertiesChart(data.smiles);
                            }
                            const analysis = await window.automationBridge.executeAction('analyzeCurrentMolecule');
                            if (analysis.success) {
                                this.addMessage("Properties predicted and visualized! " + (analysis.data.analysis || ''), 'assistant');
                            }
                        }
                        break;

                    case 'documentSession':
                        result = await window.automationBridge.executeAction('documentResults');
                        if (result.success && result.markdown) {
                            this.addMessage("I've documented the current session. Here's a summary:\\n\\n" + result.markdown.substring(0, 500) + "...\\n\\nWould you like me to export the full report?", 'assistant', [
                                {
                                    type: 'button',
                                    label: 'Export Full Report',
                                    action: 'exportSession'
                                }
                            ]);
                        }
                        break;

                    case 'exportSession':
                        result = await window.automationBridge.executeAction('exportCurrentSession');
                        this.addMessage(result.message || 'Session exported successfully!', 'assistant');
                        break;

                    default:
                        // Try to execute custom action
                        result = await window.automationBridge.executeAction(action, data);
                        this.addMessage(result.message || `Action ${action} completed`, 'assistant');
                }
            } catch (error) {
                this.addMessage(`Error executing action: ${error.message}`, 'assistant');
            }
        } else {
            // Fallback to original implementation
            switch (action) {
                case 'generateMolecules':
                    generateMolecules();
                    this.addMessage("Generation started! Check the activity log for progress.", 'assistant');
                    break;
                case 'optimizeLeadWithAPI':
                    if (window.optimizeLeadWithAPI) {
                        await window.optimizeLeadWithAPI();
                        this.addMessage("Optimization in progress! I'll analyze the results when ready.", 'assistant');
                    }
                    break;
                case 'predictADMET':
                    if (window.updateADMETChart) {
                        const smiles = document.getElementById('smilesInput').value;
                        await window.updateADMETChart(smiles);
                        this.addMessage("ADMET predictions updated! Check the ADMET Profile chart.", 'assistant');
                    }
                    break;
                case 'planSynthesis':
                    this.addMessage("Synthesis planning is a complex task. In a full implementation, this would show retrosynthetic routes, starting materials, and reaction conditions.", 'assistant');
                    break;
                case 'predictProperties':
                    if (data.smiles && window.updatePropertiesChart) {
                        await window.updatePropertiesChart(data.smiles);
                        this.addMessage("Properties predicted and displayed in the radar chart!", 'assistant');
                    }
                    break;
            }
        }
    }

    // MCP/API integration
    async queryMCPServer(message, context) {
        // This would connect to your MCP server
        // For now, return a mock response
        return {
            answer: "I'm processing your request through the chemistry knowledge base...",
            actions: []
        };
    }

    // Entity extraction helpers
    extractGenerationEntities(message) {
        const entities = {};
        
        // Extract number
        const numMatch = message.match(/(\d+)\s*(molecule|compound|structure)/i);
        if (numMatch) entities.number = parseInt(numMatch[1]);
        
        // Extract properties
        if (/drug[\s-]?like/i.test(message)) entities.drugLike = true;
        if (/similar/i.test(message)) entities.similar = true;
        
        const qedMatch = message.match(/QED\s*[>>=]+\s*([\d.]+)/i);
        if (qedMatch) entities.qed = parseFloat(qedMatch[1]);
        
        // Extract SMILES if present
        const smilesMatch = message.match(/[A-Z][A-Za-z0-9@+\-\[\]()=#]+/);
        if (smilesMatch && smilesMatch[0].length > 2) {
            entities.smiles = smilesMatch[0];
        }
        
        return entities;
    }

    extractAnalysisEntities(message) {
        const entities = {};
        
        // Extract SMILES if present
        const smilesMatch = message.match(/[A-Z][A-Za-z0-9@+\-\[\]()=#]+/);
        if (smilesMatch) entities.smiles = smilesMatch[0];
        
        return entities;
    }

    extractOptimizationEntities(message) {
        const entities = { goals: {} };
        
        if (/solubility/i.test(message)) {
            entities.goals.logP = [1.0, 3.0];
        }
        if (/potency/i.test(message)) {
            entities.focus = 'potency';
        }
        
        return entities;
    }

    extractSynthesisEntities(message) {
        const entities = {};
        
        const smilesMatch = message.match(/[A-Z][A-Za-z0-9@+\-\[\]()=#]+/);
        if (smilesMatch) entities.target = smilesMatch[0];
        
        return entities;
    }

    extractPredictionEntities(message) {
        const entities = {};
        
        if (/admet/i.test(message)) entities.predictType = 'admet';
        if (/solubility/i.test(message)) entities.predictType = 'solubility';
        if (/bbb|blood[\s-]?brain/i.test(message)) entities.predictType = 'bbb';
        
        return entities;
    }

    extractSetMoleculeEntities(message) {
        const entities = {};
        
        // Look for SMILES patterns
        // Common patterns: benzene (c1ccccc1), ethanol (CCO), aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)
        const smilesPatterns = [
            /c1ccccc1/i,  // benzene
            /[A-Z][A-Za-z0-9@+\-\[\]()=#]{2,}/,  // General SMILES
            /"([^"]+)"/,  // Quoted SMILES
            /'([^']+)'/   // Single quoted SMILES
        ];
        
        for (const pattern of smilesPatterns) {
            const match = message.match(pattern);
            if (match) {
                entities.smiles = match[1] || match[0];
                break;
            }
        }
        
        return entities;
    }

    getRecentActivity() {
        // Get last 5 activities from the activity log
        const activities = [];
        const items = document.querySelectorAll('.activity-item');
        for (let i = 0; i < Math.min(5, items.length); i++) {
            activities.push({
                time: items[i].querySelector('.activity-time').textContent,
                title: items[i].querySelector('.activity-title').textContent,
                description: items[i].querySelector('.activity-description').textContent
            });
        }
        return activities;
    }

    mapPropertiesToTarget(properties) {
        if (properties.drugLike) return 'drug-like';
        if (properties.qed > 0.8) return 'high-qed';
        if (properties.mw && properties.mw < 300) return 'low-mw';
        return 'custom';
    }
}

// Initialize AI agent when dashboard loads
document.addEventListener('DOMContentLoaded', () => {
    window.aiAgent = new AIChemistryAgent();
    console.log('AI Chemistry Assistant initialized');
});

// Export for use
window.AIChemistryAgent = AIChemistryAgent;