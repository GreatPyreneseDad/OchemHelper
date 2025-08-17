/**
 * Automation Bridge for AI Control
 * 
 * This script provides a comprehensive interface for AI agents to control
 * the OChem Helper dashboard programmatically.
 */

class AutomationBridge {
    constructor() {
        this.actions = new Map();
        this.currentState = {};
        this.actionHistory = [];
        this.setupActions();
        this.exposeToWindow();
    }

    setupActions() {
        // Form control actions
        this.registerAction('setInputSMILES', async (smiles) => {
            const input = document.getElementById('smilesInput');
            if (input) {
                input.value = smiles;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                this.updateState({ currentSMILES: smiles });
                return { success: true, message: `Set SMILES to: ${smiles}` };
            }
            return { success: false, message: 'SMILES input not found' };
        });

        this.registerAction('setGenerationMode', async (mode) => {
            const select = document.getElementById('modeSelect');
            if (select && Array.from(select.options).some(opt => opt.value === mode)) {
                select.value = mode;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                this.updateState({ generationMode: mode });
                return { success: true, message: `Set generation mode to: ${mode}` };
            }
            return { success: false, message: 'Invalid generation mode' };
        });

        this.registerAction('setTargetProperties', async (target) => {
            const select = document.getElementById('targetSelect');
            if (select && Array.from(select.options).some(opt => opt.value === target)) {
                select.value = target;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                this.updateState({ targetProperties: target });
                return { success: true, message: `Set target properties to: ${target}` };
            }
            return { success: false, message: 'Invalid target properties' };
        });

        this.registerAction('setNumMolecules', async (num) => {
            const input = document.getElementById('numMolecules');
            if (input && num >= 1 && num <= 100) {
                input.value = num;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                this.updateState({ numMolecules: num });
                return { success: true, message: `Set number of molecules to: ${num}` };
            }
            return { success: false, message: 'Invalid number of molecules (1-100)' };
        });

        // Action execution
        this.registerAction('generateMolecules', async () => {
            this.logAction('generateMolecules', {});
            
            // Check if we have API integration
            if (window.generateMoleculesWithAPI) {
                await window.generateMoleculesWithAPI();
                return { success: true, message: 'Generation started via API' };
            } else if (window.generateMolecules) {
                window.generateMolecules();
                return { success: true, message: 'Generation started (demo mode)' };
            }
            return { success: false, message: 'Generation function not available' };
        });

        this.registerAction('optimizeLead', async () => {
            const smiles = document.getElementById('smilesInput').value;
            if (!smiles) {
                return { success: false, message: 'No SMILES input provided' };
            }
            
            this.logAction('optimizeLead', { smiles });
            
            if (window.optimizeLeadWithAPI) {
                await window.optimizeLeadWithAPI();
                return { success: true, message: 'Lead optimization started' };
            }
            return { success: false, message: 'Optimization function not available' };
        });

        // Data extraction actions
        this.registerAction('getCurrentMolecule', async () => {
            const smiles = document.getElementById('smilesInput').value;
            return { 
                success: true, 
                data: { 
                    smiles, 
                    hasStructure: !!smiles 
                } 
            };
        });

        this.registerAction('getMetrics', async () => {
            const metrics = {
                moleculesGenerated: document.getElementById('moleculesGenerated')?.textContent || '0',
                avgQED: document.getElementById('avgQED')?.textContent || '0',
                validityRate: document.getElementById('validityRate')?.textContent || '0%',
                synthesisScore: document.getElementById('synthesisScore')?.textContent || '0/10'
            };
            return { success: true, data: metrics };
        });

        this.registerAction('getActivityLog', async () => {
            const activities = [];
            document.querySelectorAll('.activity-item').forEach((item, index) => {
                if (index < 10) { // Last 10 activities
                    activities.push({
                        time: item.querySelector('.activity-time')?.textContent,
                        title: item.querySelector('.activity-title')?.textContent,
                        description: item.querySelector('.activity-description')?.textContent
                    });
                }
            });
            return { success: true, data: activities };
        });

        // Chart data extraction
        this.registerAction('getChartData', async (chartId) => {
            try {
                const chartDiv = document.getElementById(chartId);
                if (chartDiv && chartDiv._fullData) {
                    return { 
                        success: true, 
                        data: {
                            data: chartDiv._fullData,
                            layout: chartDiv._fullLayout
                        }
                    };
                }
                return { success: false, message: 'Chart not found or no data' };
            } catch (error) {
                return { success: false, message: error.message };
            }
        });

        // Molecule viewer control
        this.registerAction('updateMoleculeViewer', async (smiles) => {
            if (window.updateMoleculeViewer) {
                window.updateMoleculeViewer(smiles);
                return { success: true, message: 'Molecule viewer updated' };
            }
            return { success: false, message: 'Viewer update function not available' };
        });

        this.registerAction('rotateViewer', async () => {
            if (window.toggleSpin) {
                window.toggleSpin();
                return { success: true, message: 'Toggled molecule rotation' };
            }
            return { success: false, message: 'Rotation function not available' };
        });

        // Analysis actions
        this.registerAction('analyzeCurrentMolecule', async () => {
            const smiles = document.getElementById('smilesInput').value;
            if (!smiles) {
                return { success: false, message: 'No molecule to analyze' };
            }

            if (window.OChemAPI) {
                try {
                    const properties = await window.OChemAPI.predictProperties(
                        [smiles], 
                        ['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'QED', 'SA']
                    );
                    const admet = await window.OChemAPI.predictADMET(smiles);
                    
                    return {
                        success: true,
                        data: {
                            smiles,
                            properties: properties.results[0],
                            admet: admet.admet_properties,
                            analysis: this.generateAnalysisReport(properties.results[0], admet.admet_properties)
                        }
                    };
                } catch (error) {
                    return { success: false, message: `Analysis failed: ${error.message}` };
                }
            }
            
            // Fallback to mock analysis
            return {
                success: true,
                data: {
                    smiles,
                    properties: {
                        MW: 180.16,
                        logP: 2.3,
                        TPSA: 40.5,
                        QED: 0.82
                    },
                    analysis: 'Mock analysis: Good drug-like properties'
                }
            };
        });

        // Batch operations
        this.registerAction('batchGenerate', async (configs) => {
            const results = [];
            for (const config of configs) {
                // Set up configuration
                if (config.smiles) await this.executeAction('setInputSMILES', config.smiles);
                if (config.mode) await this.executeAction('setGenerationMode', config.mode);
                if (config.target) await this.executeAction('setTargetProperties', config.target);
                if (config.num) await this.executeAction('setNumMolecules', config.num);
                
                // Generate
                const result = await this.executeAction('generateMolecules');
                results.push({
                    config,
                    result,
                    timestamp: new Date().toISOString()
                });
                
                // Wait between batches
                await this.sleep(2000);
            }
            return { success: true, data: results };
        });

        // Documentation actions
        this.registerAction('documentResults', async () => {
            const documentation = {
                timestamp: new Date().toISOString(),
                currentMolecule: document.getElementById('smilesInput').value,
                metrics: await this.executeAction('getMetrics'),
                activities: await this.executeAction('getActivityLog'),
                charts: {}
            };

            // Capture chart data
            const chartIds = ['propertyRadar', 'generationChart', 'chemicalSpace', 'admetChart'];
            for (const chartId of chartIds) {
                const chartData = await this.executeAction('getChartData', chartId);
                if (chartData.success) {
                    documentation.charts[chartId] = chartData.data;
                }
            }

            return { 
                success: true, 
                data: documentation,
                markdown: this.generateMarkdownReport(documentation)
            };
        });

        // Export actions
        this.registerAction('exportCurrentSession', async () => {
            const session = {
                timestamp: new Date().toISOString(),
                state: this.currentState,
                history: this.actionHistory,
                results: await this.executeAction('documentResults')
            };

            // Create downloadable file
            const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ochem-session-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);

            return { success: true, message: 'Session exported' };
        });

        // Synthesis planning
        this.registerAction('planSynthesis', async (targetSmiles) => {
            const smiles = targetSmiles || document.getElementById('smilesInput').value;
            if (!smiles) {
                return { success: false, message: 'No target molecule provided' };
            }

            if (window.OChemAPI) {
                try {
                    const routes = await window.OChemAPI.suggestSynthesis(smiles);
                    return {
                        success: true,
                        data: {
                            target: smiles,
                            routes: routes.synthesis_routes,
                            summary: `Found ${routes.synthesis_routes.length} viable synthesis routes`
                        }
                    };
                } catch (error) {
                    return { success: false, message: `Synthesis planning failed: ${error.message}` };
                }
            }

            return { success: false, message: 'Synthesis API not available' };
        });
    }

    registerAction(name, handler) {
        this.actions.set(name, handler);
    }

    async executeAction(actionName, ...args) {
        if (!this.actions.has(actionName)) {
            return { success: false, message: `Unknown action: ${actionName}` };
        }

        try {
            const result = await this.actions.get(actionName)(...args);
            this.logAction(actionName, args, result);
            return result;
        } catch (error) {
            const errorResult = { success: false, message: error.message };
            this.logAction(actionName, args, errorResult);
            return errorResult;
        }
    }

    logAction(actionName, args, result) {
        this.actionHistory.push({
            action: actionName,
            args,
            result,
            timestamp: new Date().toISOString()
        });

        // Keep history size manageable
        if (this.actionHistory.length > 100) {
            this.actionHistory = this.actionHistory.slice(-100);
        }
    }

    updateState(updates) {
        this.currentState = {
            ...this.currentState,
            ...updates,
            lastUpdated: new Date().toISOString()
        };
    }

    generateAnalysisReport(properties, admet) {
        let report = '## Molecular Analysis Report\n\n';
        
        // Property analysis
        report += '### Molecular Properties\n';
        report += `- Molecular Weight: ${properties.MW?.toFixed(2) || 'N/A'} g/mol\n`;
        report += `- LogP: ${properties.logP?.toFixed(2) || 'N/A'}\n`;
        report += `- TPSA: ${properties.TPSA?.toFixed(2) || 'N/A'} Ų\n`;
        report += `- QED Score: ${properties.QED?.toFixed(3) || 'N/A'}\n\n`;

        // Drug-likeness
        report += '### Drug-likeness Assessment\n';
        const lipinski = this.checkLipinski(properties);
        report += lipinski.passed ? '✅ Passes Lipinski\'s Rule of Five\n' : `⚠️ Lipinski violations: ${lipinski.violations.join(', ')}\n`;

        // ADMET if available
        if (admet) {
            report += '\n### ADMET Predictions\n';
            Object.entries(admet).forEach(([key, value]) => {
                report += `- ${key}: ${value.probability?.toFixed(2) || value}\n`;
            });
        }

        return report;
    }

    checkLipinski(properties) {
        const violations = [];
        if (properties.MW > 500) violations.push('MW > 500');
        if (properties.logP > 5) violations.push('LogP > 5');
        if (properties.HBD > 5) violations.push('HBD > 5');
        if (properties.HBA > 10) violations.push('HBA > 10');
        
        return {
            passed: violations.length === 0,
            violations
        };
    }

    generateMarkdownReport(documentation) {
        let report = `# OChem Helper Analysis Report\n\n`;
        report += `**Generated**: ${new Date(documentation.timestamp).toLocaleString()}\n\n`;
        
        if (documentation.currentMolecule) {
            report += `## Current Molecule\n\`\`\`\n${documentation.currentMolecule}\n\`\`\`\n\n`;
        }

        report += `## Metrics\n`;
        if (documentation.metrics?.data) {
            Object.entries(documentation.metrics.data).forEach(([key, value]) => {
                report += `- **${key}**: ${value}\n`;
            });
        }

        report += `\n## Recent Activities\n`;
        if (documentation.activities?.data) {
            documentation.activities.data.forEach(activity => {
                report += `- **${activity.time}** - ${activity.title}: ${activity.description}\n`;
            });
        }

        return report;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    exposeToWindow() {
        // Expose the automation bridge globally
        window.automationBridge = this;
        
        // Convenience method for AI agents
        window.executeChemistryAction = async (action, ...args) => {
            return await this.executeAction(action, ...args);
        };

        // Quick access methods
        window.setMolecule = async (smiles) => {
            return await this.executeAction('setInputSMILES', smiles);
        };

        window.analyzeMolecule = async () => {
            return await this.executeAction('analyzeCurrentMolecule');
        };

        window.documentSession = async () => {
            return await this.executeAction('documentResults');
        };

        console.log('Automation Bridge initialized. Use window.automationBridge or window.executeChemistryAction()');
    }

    // Helper method for AI to get available actions
    getAvailableActions() {
        return Array.from(this.actions.keys()).map(action => ({
            name: action,
            description: this.getActionDescription(action)
        }));
    }

    getActionDescription(action) {
        const descriptions = {
            setInputSMILES: 'Set the SMILES input field',
            setGenerationMode: 'Set generation mode (random, similar, optimize, interpolate)',
            setTargetProperties: 'Set target properties (drug-like, high-qed, low-mw, custom)',
            setNumMolecules: 'Set number of molecules to generate (1-100)',
            generateMolecules: 'Start molecule generation',
            optimizeLead: 'Optimize the current lead compound',
            getCurrentMolecule: 'Get the current SMILES input',
            getMetrics: 'Get dashboard metrics',
            getActivityLog: 'Get recent activity log',
            getChartData: 'Get data from a specific chart',
            updateMoleculeViewer: 'Update 3D molecule viewer',
            rotateViewer: 'Toggle molecule rotation',
            analyzeCurrentMolecule: 'Analyze current molecule properties and ADMET',
            batchGenerate: 'Batch generate molecules with different configs',
            documentResults: 'Create comprehensive documentation of results',
            exportCurrentSession: 'Export current session data',
            planSynthesis: 'Plan synthesis routes for target molecule'
        };
        return descriptions[action] || 'No description available';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.automationBridge = new AutomationBridge();
    console.log('OChem Helper Automation Bridge ready');
    
    // Notify AI chat if available
    if (window.aiAgent) {
        window.aiAgent.addMessage(
            'Automation bridge connected! I can now control the dashboard directly.',
            'assistant'
        );
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AutomationBridge;
}