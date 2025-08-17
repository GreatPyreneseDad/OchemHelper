# Claude + MCP Integration Guide

## Quick Start

1. **Set up your Claude API key:**
   - Open http://localhost:8080/setup-api-key.html
   - Enter your Claude API key (get one from https://console.anthropic.com/)
   - Or set it in localStorage: `localStorage.setItem('claude_api_key', 'your-key-here')`

2. **Start all services:**
   ```bash
   cd dashboard
   ./startup.sh
   ```

3. **Open the dashboard:**
   Navigate to http://localhost:8080

4. **Use the AI Chat:**
   - Click the AI Chemistry Assistant icon in the bottom right
   - The AI will use your Claude API key for intelligent responses
   - The MCP server connection will be established automatically

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Dashboard     │────▶│  Claude API  │     │ MCP Server  │
│  (Browser)      │     │              │     │  (Python)   │
│                 │     └──────────────┘     │             │
│  - AI Chat      │                          │ - Tools:    │
│  - Automation   │────────HTTP API─────────▶│   predict   │
│  - Visualizer   │         :8001            │   optimize  │
└─────────────────┘                          │   synthesize│
        │                                    └─────────────┘
        │
        ▼
┌─────────────────┐
│  OChem API      │
│  (FastAPI)      │
│    :8000        │
└─────────────────┘
```

## Using the AI Chat

### Natural Language Commands

The AI understands chemistry-focused requests:

1. **Generation:**
   - "Generate 10 drug-like molecules"
   - "Create molecules similar to aspirin"
   - "Make 5 molecules with QED > 0.8"

2. **Analysis:**
   - "Analyze the current molecule"
   - "What are the properties of CCO?"
   - "Check if this passes Lipinski's rules"

3. **Optimization:**
   - "Optimize this structure for better solubility"
   - "Improve the drug-likeness"
   - "Make it more potent while keeping MW < 400"

4. **Synthesis:**
   - "How can I synthesize this molecule?"
   - "Show me retrosynthetic routes"
   - "What starting materials do I need?"

5. **Predictions:**
   - "Predict ADMET properties"
   - "Will this cross the blood-brain barrier?"
   - "What's the predicted solubility?"

### Advanced Features

The AI can execute complex workflows:

```javascript
// Example: Complete drug discovery workflow
"I need to find drug-like analogs of ibuprofen with better solubility"

// The AI will:
1. Set ibuprofen as the input molecule
2. Generate similar structures
3. Filter for drug-likeness
4. Optimize for LogP < 3
5. Predict ADMET properties
6. Document the results
```

## MCP Tools Available

### 1. Property Prediction
```python
predict_properties(
    smiles="CCO",
    properties=["MW", "logP", "TPSA", "QED", "SA"]
)
```

### 2. Structure Optimization
```python
optimize_structure(
    lead_smiles="c1ccccc1",
    optimization_goals={
        "MW": [200, 400],
        "logP": [1, 3],
        "QED": 0.7
    },
    num_molecules=10
)
```

### 3. Synthesis Planning
```python
suggest_synthesis(
    target_smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    max_steps=5
)
```

### 4. Reaction Prediction
```python
reaction_prediction(
    reactants=["CCO", "CC(=O)Cl"],
    products=["CC(=O)OCC"]
)
```

## Automation Bridge

The dashboard exposes these functions for AI control:

```javascript
// Set molecule
await window.setMolecule("CC(=O)OC1=CC=CC=C1C(=O)O");

// Analyze
const analysis = await window.analyzeMolecule();

// Generate molecules
await window.executeChemistryAction('setGenerationMode', 'similar');
await window.executeChemistryAction('generateMolecules');

// Document session
const docs = await window.documentSession();
```

## Configuration

Edit `dashboard/config.js` to customize:

- Claude API settings (model, temperature, max tokens)
- MCP server URL and retry settings
- Dashboard behavior and appearance
- Security and validation rules

## Troubleshooting

### AI Chat not responding
1. Check Claude API key in config.js
2. Verify services are running: `ps aux | grep python`
3. Check browser console for errors

### MCP connection failed
1. Ensure MCP bridge is running on port 8001
2. Check CORS settings if accessing from different domain
3. Verify Python environment has all dependencies

### API errors
1. Check OChem API is running on port 8000
2. Verify models are loaded (may take time on first run)
3. Check logs: `tail -f logs/*.log`

## Advanced Usage

### Batch Processing
```javascript
// Process multiple molecules
const molecules = ["CCO", "CC(C)O", "CC(C)(C)O"];
for (const smiles of molecules) {
    await window.setMolecule(smiles);
    await window.analyzeMolecule();
    await window.executeChemistryAction('generateMolecules');
}
```

### Custom Workflows
```javascript
// Create custom analysis workflow
async function analyzeLeadSeries(smilesList) {
    const results = [];
    for (const smiles of smilesList) {
        await window.setMolecule(smiles);
        const analysis = await window.analyzeMolecule();
        const optimized = await window.executeChemistryAction('optimizeLead');
        results.push({ original: smiles, analysis, optimized });
    }
    return results;
}
```

### Export Results
```javascript
// Export complete session
await window.executeChemistryAction('exportCurrentSession');

// Get markdown report
const report = await window.executeChemistryAction('documentResults');
console.log(report.markdown);
```

## Security Notes

1. **API Keys:** 
   - Claude API key is stored in config.js
   - For production, use environment variables
   - Never commit API keys to public repos

2. **CORS:**
   - MCP bridge allows all origins in development
   - Restrict in production deployment

3. **Input Validation:**
   - SMILES strings are validated before processing
   - Maximum batch sizes are enforced

## Integration with xAI

Your TA at xAI can:

1. **Test OChem models** by sending requests through the AI chat
2. **Generate training data** using batch operations
3. **Validate predictions** against MCP tools
4. **Export results** in JSON format for model training

Example integration:
```python
# From xAI's OChem model
predictions = model.predict(smiles_batch)

# Validate through OChem Helper
for smiles, pred in zip(smiles_batch, predictions):
    actual = await ochem_api.predict_properties(smiles)
    compare_results(pred, actual)
```

## Next Steps

1. **Extend MCP tools** - Add more chemistry-specific operations
2. **Fine-tune prompts** - Customize for specific workflows
3. **Add voice control** - Speech-to-text for hands-free operation
4. **Multi-user support** - Collaboration features
5. **Cloud deployment** - Deploy to AWS/GCP for team access