# OChem Helper Examples

This directory contains Jupyter notebooks demonstrating various features and workflows of the OChem Helper platform.

## Prerequisites

Before running the notebooks, ensure you have:

1. Installed all requirements:
   ```bash
   pip install -r requirements.txt
   pip install jupyter notebook ipywidgets
   ```

2. Downloaded training data (if using pretrained models):
   ```bash
   python scripts/download_data.py
   ```

3. Set up environment variables (if needed):
   ```bash
   export OPENAI_API_KEY="your-key-here"  # If using OpenAI models
   ```

## Notebooks

### 1. Getting Started (`01_getting_started.ipynb`)
Introduction to basic OChem Helper functionality:
- Molecule generation using VAE
- Targeted generation with property constraints
- Molecule validation and drug-likeness checking
- Property calculation and analysis
- Molecular filtering

**Key concepts**: VAE generation, SMILES validation, molecular descriptors, Lipinski's Rule of Five

### 2. Lead Optimization (`02_lead_optimization.ipynb`)
Advanced lead compound optimization:
- Multi-parameter optimization
- Scaffold-constrained analog generation
- Property comparison and goal achievement analysis
- Candidate selection and ranking
- Export functionality for further development

**Key concepts**: Lead optimization, structure-activity relationships, multi-objective optimization

### 3. Synthesis Planning (`03_synthesis_planning.ipynb`)
Retrosynthetic analysis and reaction planning:
- Retrosynthetic route discovery
- Reaction feasibility checking
- Condition optimization
- Starting material analysis
- Synthesis plan generation

**Key concepts**: Retrosynthesis, reaction prediction, synthetic accessibility

### 4. MCP Integration (`04_mcp_integration.ipynb`)
Model Context Protocol (MCP) server for AI integration:
- MCP server setup and tool exploration
- Property and ADMET predictions via MCP
- Structure optimization through MCP
- Complex drug discovery workflows
- Batch processing capabilities

**Key concepts**: MCP, AI tool integration, workflow automation

## Running the Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `examples/` directory

3. Open any notebook and run cells sequentially

## Tips

- **GPU Support**: For faster VAE operations, use a CUDA-enabled GPU
- **Memory**: Some operations require significant memory (>8GB RAM recommended)
- **Async Operations**: Notebooks use `nest_asyncio` for async MCP operations
- **Visualization**: RDKit is used for molecule visualization

## Common Issues

### Import Errors
If you encounter import errors, ensure the src directory is in your Python path:
```python
import sys
sys.path.append('../src')
```

### Async Errors
For async operations in Jupyter:
```python
import nest_asyncio
nest_asyncio.apply()
```

### Missing Models
If pretrained models are missing:
```bash
python scripts/train_vae.py --epochs 10  # Train a basic model
```

## Extending Examples

Feel free to modify and extend these examples:
- Try different molecules and property targets
- Implement custom optimization strategies
- Add new visualization methods
- Create domain-specific workflows

## Additional Resources

- [RDKit Documentation](https://www.rdkit.org/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MCP Specification](https://github.com/anthropics/mcp)
- [SMILES Tutorial](http://opensmiles.org/opensmiles.html)

## Contributing

To contribute new examples:
1. Create a new notebook following the naming convention
2. Include clear markdown explanations
3. Test all code cells
4. Update this README

## License

These examples are part of the OChem Helper project and follow the same MIT license.