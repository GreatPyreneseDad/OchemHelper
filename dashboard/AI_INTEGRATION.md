# AI Integration for OChem Helper Dashboard

## Overview

The OChem Helper dashboard now includes a powerful AI chat interface and automation bridge that allows AI agents (including LLMs) to fully control and operate the molecular discovery platform.

## Components

### 1. AI Chat Interface (`ai-chat.js`)

An interactive chat window that provides:
- Natural language interaction with the chemistry platform
- Intent recognition for common chemistry tasks
- Automatic form filling and action execution
- Real-time feedback and result analysis
- Context-aware suggestions

#### Key Features:
- **Intent Analysis**: Recognizes patterns like "generate", "analyze", "optimize", "synthesize"
- **Smart Actions**: Automatically fills forms and executes appropriate functions
- **Visual Feedback**: Shows loading states, action buttons, and formatted responses
- **Persistent History**: Maintains conversation context for better assistance

### 2. Automation Bridge (`automation-bridge.js`)

A comprehensive API that allows programmatic control of the dashboard:

#### Available Actions:

**Form Control:**
- `setInputSMILES(smiles)` - Set the SMILES input field
- `setGenerationMode(mode)` - Set generation mode (random/similar/optimize/interpolate)
- `setTargetProperties(target)` - Set target properties (drug-like/high-qed/low-mw/custom)
- `setNumMolecules(num)` - Set number of molecules to generate (1-100)

**Execution:**
- `generateMolecules()` - Start molecule generation
- `optimizeLead()` - Optimize the current lead compound
- `planSynthesis(targetSmiles)` - Plan synthesis routes

**Data Extraction:**
- `getCurrentMolecule()` - Get current SMILES input
- `getMetrics()` - Get dashboard metrics
- `getActivityLog()` - Get recent activities
- `getChartData(chartId)` - Extract data from charts

**Analysis:**
- `analyzeCurrentMolecule()` - Full property and ADMET analysis
- `documentResults()` - Create comprehensive documentation
- `exportCurrentSession()` - Export session data as JSON

**Visualization:**
- `updateMoleculeViewer(smiles)` - Update 3D structure
- `rotateViewer()` - Toggle molecule rotation

### 3. API Integration (`api-integration.js`)

Connects the dashboard to the OChem Helper backend API:
- Real-time property predictions
- ADMET profiling
- Lead optimization
- Synthesis route planning
- WebSocket support for live updates

## Usage Examples

### For AI Agents / LLMs

The automation bridge exposes global functions that AI agents can call:

```javascript
// Set a molecule
await window.setMolecule("CC(=O)OC1=CC=CC=C1C(=O)O");

// Analyze it
const analysis = await window.analyzeMolecule();

// Generate similar molecules
await window.executeChemistryAction('setGenerationMode', 'similar');
await window.executeChemistryAction('setNumMolecules', 10);
await window.executeChemistryAction('generateMolecules');

// Document the session
const docs = await window.documentSession();
```

### Natural Language Examples

Users can interact with the AI chat using natural language:

1. **"Generate 10 drug-like molecules"**
   - AI sets target to drug-like
   - Sets number to 10
   - Executes generation

2. **"Analyze the current molecule"**
   - AI reads current SMILES
   - Calls property prediction
   - Displays formatted analysis

3. **"Optimize this structure for better solubility"**
   - AI initiates lead optimization
   - Sets goals for improved LogP
   - Monitors results

4. **"How can I synthesize this molecule?"**
   - AI calls retrosynthesis planning
   - Shows available routes
   - Provides starting materials

## Integration with MCP Server

The dashboard is designed to work with the MCP (Model Context Protocol) server:

1. **MCP Tools Available:**
   - `predict_properties` - Property and ADMET predictions
   - `suggest_synthesis` - Retrosynthetic analysis
   - `optimize_structure` - Lead optimization
   - `reaction_prediction` - Reaction feasibility

2. **Connection:**
   ```javascript
   // The AI chat can query the MCP server
   const response = await this.queryMCPServer(message, context);
   ```

## Batch Operations

The automation bridge supports batch operations for efficiency:

```javascript
const configs = [
    { smiles: "CCO", mode: "similar", num: 5 },
    { smiles: "c1ccccc1", mode: "optimize", num: 10 },
    { mode: "random", target: "drug-like", num: 20 }
];

const results = await window.automationBridge.executeAction('batchGenerate', configs);
```

## Data Export and Documentation

The system can automatically document and export sessions:

```javascript
// Get comprehensive documentation
const docs = await window.automationBridge.executeAction('documentResults');
console.log(docs.markdown); // Markdown report

// Export full session
await window.automationBridge.executeAction('exportCurrentSession');
// Downloads JSON file with complete session data
```

## Error Handling

All actions return structured responses:

```javascript
{
    success: true/false,
    message: "Description of result or error",
    data: { /* Optional data payload */ }
}
```

## Best Practices for AI Integration

1. **Always check molecule input** before operations
2. **Monitor async operations** with setTimeout or polling
3. **Provide clear feedback** to users about what's happening
4. **Document results** for reproducibility
5. **Handle errors gracefully** with informative messages

## Extending the System

To add new automation actions:

```javascript
automationBridge.registerAction('customAction', async (param1, param2) => {
    // Your custom logic here
    return { success: true, data: result };
});
```

## Security Considerations

- Input validation on all SMILES strings
- Rate limiting for API calls
- Sanitization of user inputs
- Secure storage of session data

## For Developers

The system is designed to be extended:

1. Add new intents in `analyzeIntent()`
2. Create new action handlers in `handleAction()`
3. Register automation actions in `setupActions()`
4. Extend the MCP server with new tools

## Example Workflow

```javascript
// Complete drug discovery workflow
async function drugDiscoveryWorkflow() {
    // 1. Set initial lead compound
    await window.setMolecule("c1ccc(cc1)C(=O)O");
    
    // 2. Analyze properties
    const analysis = await window.analyzeMolecule();
    console.log("Initial QED:", analysis.data.properties.QED);
    
    // 3. Generate analogs
    await window.executeChemistryAction('setGenerationMode', 'similar');
    await window.executeChemistryAction('setNumMolecules', 25);
    await window.executeChemistryAction('generateMolecules');
    
    // 4. Wait for generation
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // 5. Document results
    const documentation = await window.documentSession();
    console.log("Session documented:", documentation);
    
    // 6. Export for further analysis
    await window.executeChemistryAction('exportCurrentSession');
}
```

## Troubleshooting

1. **AI Chat not appearing**: Check console for JavaScript errors
2. **Actions not executing**: Verify automation bridge is loaded
3. **API calls failing**: Ensure backend server is running
4. **Charts not updating**: Check Plotly library is loaded

## Future Enhancements

- Voice control integration
- Multi-language support
- Advanced batch processing
- Real-time collaboration features
- Integration with more AI models
- Automated experiment planning