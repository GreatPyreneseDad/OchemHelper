# Claude Code Task Delegation

## Repository Location
- **Local repo path**: /Users/chris/ochem-helper
- **GitHub repo**: https://github.com/GreatPyreneseDad/OchemHelper
- **Main branch**: `main`

## Priority Tasks for Claude Code

### 1. Complete Missing Core Modules (HIGH PRIORITY)

#### A. Create `src/core/descriptors.py`
```python
# Implement molecular descriptor calculations
# Include: MW, LogP, TPSA, QED, SA Score, etc.
# Use RDKit for calculations
# Add caching for expensive computations
```

#### B. Create `src/core/validators.py`  
```python
# Implement SMILES validation
# Chemical rule checking
# Drug-likeness filters (Lipinski, Veber, etc.)
# Structural alerts (PAINS, toxicophores)
```

#### C. Create `src/core/utils.py`
```python
# Utility functions for SMILES/molecule conversion
# Standardization functions
# Common chemical transformations
```

### 2. Implement MCP Tools (HIGH PRIORITY)

#### A. Create `mcp/tools/predict_properties.py`
```python
# Property prediction using RDKit and ML models
# ADMET predictions
# Activity prediction frameworks
```

#### B. Create `mcp/tools/suggest_synthesis.py`
```python
# Retrosynthetic analysis
# Reaction template matching
# Synthetic route planning
```

#### C. Create `mcp/tools/optimize_structure.py`
```python
# Lead optimization algorithms
# Analog generation
# Multi-objective optimization
```

#### D. Create `mcp/tools/reaction_prediction.py`
```python
# Reaction feasibility prediction
# Product prediction
# Condition optimization
```

#### E. Create `mcp/tools/retrosynthesis.py`
```python
# Advanced retrosynthetic planning
# Starting material identification
# Route scoring
```

### 3. Expand Test Suite (MEDIUM PRIORITY)

#### A. Create `tests/unit/test_vae.py`
```python
# Test VAE training pipeline
# Test molecule generation
# Test property calculations
```

#### B. Create `tests/unit/test_api.py`
```python
# API endpoint testing
# Request/response validation
# Error handling tests
```

#### C. Create `tests/unit/test_mcp.py`
```python
# MCP server functionality
# Tool execution tests
# Prompt testing
```

### 4. Documentation and Examples (MEDIUM PRIORITY)

#### A. Create `examples/` directory with notebooks
- `01_basic_generation.ipynb`
- `02_property_prediction.ipynb`
- `03_lead_optimization.ipynb`
- `04_mcp_integration.ipynb`

#### B. Create comprehensive API docs
- Update docstrings
- Generate OpenAPI specs
- Create usage examples

### 5. Configuration and Deployment (LOW PRIORITY)

#### A. Create proper config management
- `configs/default.yaml`
- `configs/production.yaml`
- Environment-specific settings

#### B. Improve containerization
- Multi-stage Docker builds
- GPU support in containers
- Health checks

## Implementation Guidelines

### Code Quality Standards
- Follow existing code style (Black formatting)
- Add comprehensive docstrings
- Include type hints
- Add error handling and logging
- Write accompanying tests

### Integration Requirements
- Ensure RDKit compatibility
- Maintain MCP protocol compliance
- Follow FastAPI patterns
- Use existing tokenizer/model classes

### Testing Requirements
- Unit tests for all new functions
- Integration tests for workflows
- Mock external dependencies
- Validate chemical correctness

## File Locations to Reference

When implementing, reference these existing files for patterns:
- `src/models/generative/smiles_vae.py` - Model architecture patterns
- `src/api/app.py` - API endpoint patterns
- `mcp/server/ochem_mcp.py` - MCP server structure
- `mcp/tools/analyze_molecule.py` - Tool implementation example
- `scripts/train_vae.py` - Training script patterns

## Success Criteria

### Functional Requirements
- All imports resolve without errors
- Tests pass with >90% coverage
- API endpoints return valid responses
- MCP tools execute successfully

### Chemical Validation
- Generated molecules are chemically valid
- Property predictions are reasonable
- SMILES parsing works correctly
- Chemical rules are enforced

## Next Actions for Claude Code

1. **Start with core modules** (`descriptors.py`, `validators.py`, `utils.py`)
2. **Implement MCP tools** in order of complexity
3. **Add comprehensive tests** for each module
4. **Create example notebooks** demonstrating functionality
5. **Update documentation** and README with new features

After completing these tasks, create a pull request with all changes for review and integration into the main repository.