# OChem Helper 🧪🤖

An advanced neural network system for molecular discovery and organic chemistry simulations.

## Features

- 🔬 **Molecular Generation**: Design novel molecules with desired properties
- 📊 **Property Prediction**: Predict chemical and biological properties
- 🧬 **Retrosynthesis**: Plan synthetic routes automatically
- ⚛️ **Quantum Chemistry**: Interface with QM calculations
- 🎯 **Drug Discovery**: Optimize lead compounds
- 📈 **ADMET Prediction**: Assess drug-like properties

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/GreatPyreneseDad/OchemHelper.git
cd ochem-helper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Basic Usage

```python
from ochem_helper import MolecularGraph, MoleculeGenerator, PropertyPredictor

# Initialize models
generator = MoleculeGenerator.from_pretrained("vae-chembl")
predictor = PropertyPredictor.from_pretrained("gnn-admet")

# Generate molecules
molecules = generator.generate(
    n_molecules=100,
    target_properties={"logP": 2.5, "MW": 350}
)

# Predict properties
properties = predictor.predict(molecules)
```

### Training Custom Models

```bash
# Train property prediction model
ochem-train --config configs/models/gnn_config.yaml

# Train generative model
ochem-train --config configs/models/vae_config.yaml --data chembl
```

## Documentation

Full documentation available at [https://ochem-helper.readthedocs.io](https://ochem-helper.readthedocs.io)

## API Usage

Start the API server:

```bash
ochem-api
```

Generate molecules via API:

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"n_molecules": 10, "target_properties": {"logP": 2.5}}'
```

## Container Usage (Podman)

Build and run with Podman:

```bash
# Build container
make podman-build

# Run container
make podman-run

# Or use podman-compose
cd containers
podman-compose up
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## Citation

If you use OChem Helper in your research, please cite:

```bibtex
@software{ochem_helper,
  title = {OChem Helper: Neural Networks for Molecular Discovery},
  author = {OChem Helper Contributors},
  year = {2024},
  url = {https://github.com/GreatPyreneseDad/OchemHelper}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- RDKit for cheminformatics
- PyTorch Geometric for graph neural networks
- DeepChem for chemical machine learning