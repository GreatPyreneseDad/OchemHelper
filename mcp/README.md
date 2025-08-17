# OChem Helper MCP Server

This MCP (Model Context Protocol) server enables AI assistants like Claude and Grok to perform advanced organic chemistry tasks.

## Features

The MCP server provides these tools:

### 1. **analyze_molecule**
- Analyzes molecular structure and properties
- Calculates descriptors (MW, logP, TPSA, etc.)
- Checks drug-likeness rules (Lipinski, Veber, etc.)
- Identifies structural alerts and functional groups

### 2. **predict_activity**
- Predicts biological activity
- ADMET property prediction
- Target-specific activity prediction

### 3. **suggest_synthesis**
- Retrosynthetic analysis
- Suggests synthetic routes
- Identifies key disconnections
- Recommends starting materials

### 4. **optimize_lead**
- Lead compound optimization
- Multi-objective optimization
- Maintains core scaffold
- Generates optimized analogs

### 5. **reaction_feasibility**
- Checks if reactions are chemically feasible
- Validates reaction conditions
- Predicts reaction outcomes

### 6. **generate_analogs**
- Creates molecular analogs
- Controls similarity threshold
- Maintains desired properties

## Setup

### 1. Install MCP SDK
```bash
npm install @modelcontextprotocol/sdk
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ochem-helper": {
      "command": "python",
      "args": ["-m", "mcp.server.ochem_mcp"],
      "cwd": "/path/to/ochem-helper"
    }
  }
}
```

### 4. Configure for Grok/xAI

For integration with Grok, use the API endpoint:

```python
import requests

# Example API call
response = requests.post(
    "http://localhost:8000/mcp/tools/analyze_molecule",
    json={
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "properties": ["MW", "logP", "QED"]
    }
)
```

## Usage Examples

### With Claude

```
Human: Can you analyze aspirin for me and suggest some analogs with better solubility?

Claude: I'll analyze aspirin and generate analogs with improved solubility.

[Uses analyze_molecule tool with SMILES: CC(=O)Oc1ccccc1C(=O)O]

Aspirin Analysis:
- Molecular Weight: 180.16
- logP: 1.19
- TPSA: 63.6
- Solubility: Poor (due to carboxylic acid)

[Uses generate_analogs tool with optimization for solubility]

Generated analogs with better solubility:
1. Aspirin methyl ester: MW 194, improved solubility
2. Aspirin ethyl ester: MW 208, better bioavailability
3. Aspirin lysine salt: MW 339, highly water-soluble
```

### With Grok/xAI

```python
# Integration example for Grok
from ochem_helper_mcp import ChemistryTools

tools = ChemistryTools()

# Analyze drug candidate
result = tools.analyze_molecule(
    "CC1=C(C(=O)N2CCCC2)C(=O)N(C)C1=O",  # Piracetam
    properties=["all"]
)

# Suggest synthesis
synthesis = tools.suggest_synthesis(
    target_smiles="CC1=C(C(=O)N2CCCC2)C(=O)N(C)C1=O",
    max_steps=3
)
```

## Prompts

The MCP server includes specialized prompts:

1. **drug_discovery** - Complete drug discovery workflow
2. **lead_optimization** - Systematic lead optimization
3. **synthesis_planning** - Complex molecule synthesis

Example:
```
Human: Use the drug discovery prompt for EGFR kinase inhibitors

Claude: [Activates drug_discovery prompt with target="EGFR kinase"]
```

## API Endpoints

For direct integration:

- `POST /mcp/tools/{tool_name}` - Execute specific tool
- `GET /mcp/tools` - List available tools
- `GET /mcp/prompts` - List available prompts

## Contributing

To add new chemistry tools:

1. Create tool in `mcp/tools/`
2. Register in `ochem_mcp.py`
3. Add tests in `tests/mcp/`
4. Update documentation

## Security

- All SMILES inputs are validated
- Computation limits prevent DoS
- No external chemical database queries without auth
- Sandboxed execution environment