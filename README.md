# OChem Helper 🧪🤖

An advanced neural network system for molecular discovery and organic chemistry simulations.

## Features

- 🔬 **Molecular Generation**: Design novel molecules with desired properties using VAE
- 📊 **Property Prediction**: Predict chemical and biological properties
- 🎯 **Molecule Optimization**: Optimize molecules for specific objectives
- 🧬 **Graph Neural Networks**: Molecular representation learning
- 📈 **ADMET Prediction**: Assess drug-like properties
- 🚀 **REST API**: Production-ready API for all functionalities

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/GreatPyreneseDad/OchemHelper.git
cd ochem-helper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Download & Prepare Data

```bash
# Download all datasets (ChEMBL, ZINC, PubChem, QM9)
python scripts/download_data.py --all

# Or download specific dataset
python scripts/download_data.py --dataset chembl
```

### Train the VAE Model

```bash
# Train on combined dataset
python scripts/train_vae.py \
    --data data/processed/combined_training.csv \
    --epochs 100 \
    --batch-size 128 \
    --latent-dim 128 \
    --hidden-dim 256

# Monitor training with tensorboard or wandb
wandb login  # First time only
python scripts/train_vae.py --data data/processed/combined_training.csv --wandb-project molecular-vae
```

### Generate Molecules

```bash
# Generate random molecules
python scripts/generate_molecules.py \
    --checkpoint models/checkpoints/best_model.pt \
    --mode random \
    --n-molecules 100 \
    --output generated.csv \
    --visualize

# Generate molecules similar to a reference
python scripts/generate_molecules.py \
    --checkpoint models/checkpoints/best_model.pt \
    --mode similar \
    --reference "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --n-molecules 50

# Generate with target properties
python scripts/generate_molecules.py \
    --checkpoint models/checkpoints/best_model.pt \
    --mode properties \
    --properties '{"MW": 350, "logP": 2.5}' \
    --n-molecules 100
```

### Basic Usage in Python

```python
from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer
from src.core.molecular_graph import MolecularGraph
import torch

# Load trained model
checkpoint = torch.load('models/checkpoints/best_model.pt')
config = checkpoint['model_config']

model = MolecularVAE(
    vocab_size=config['vocab_size'],
    hidden_dim=config['hidden_dim'],
    latent_dim=config['latent_dim'],
    max_length=config['max_length']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate molecules
molecules = model.generate(n_samples=100)
print(f"Generated {len(molecules)} valid molecules")

# Molecular interpolation
aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C"
interpolated = model.interpolate(aspirin, ibuprofen, n_steps=10)

# Convert to molecular graphs
for smiles in molecules[:5]:
    mol_graph = MolecularGraph.from_smiles(smiles)
    graph_data = mol_graph.to_graph()
    print(f"Graph with {mol_graph.num_atoms} atoms, {mol_graph.num_bonds} bonds")
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