#!/bin/bash

echo "🚀 Installing Full ML Stack for M4 Max with Metal Acceleration"
echo

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with Metal Performance Shaders (MPS) support
echo "🔥 Installing PyTorch with Metal support..."
pip install torch torchvision torchaudio

# Install core scientific packages
echo "📊 Installing core scientific packages..."
pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly

# Install chemistry packages
echo "🧪 Installing chemistry packages..."
pip install rdkit-pypi  # Use rdkit-pypi for easier installation
pip install mendeleev mordred

# Install deep learning and ML packages
echo "🤖 Installing ML packages..."
pip install transformers datasets accelerate
pip install dgl-cu118  # DGL for graph neural networks
pip install torch-geometric
pip install xgboost lightgbm catboost

# Install API and web packages
echo "🌐 Installing API packages..."
pip install fastapi uvicorn pydantic httpx aiohttp requests
pip install python-multipart python-jose[cryptography] passlib[bcrypt]

# Install development tools
echo "🛠️ Installing development tools..."
pip install ipython jupyter notebook jupyterlab
pip install pytest pytest-cov black flake8 mypy
pip install rich typer tqdm wandb

# Install MCP and utilities
echo "⚙️ Installing utilities..."
pip install mcp pyyaml python-dotenv hydra-core omegaconf

# Install visualization tools
echo "🎨 Installing visualization tools..."
pip install py3Dmol streamlit gradio dash

# Test PyTorch Metal support
echo
echo "🧪 Testing PyTorch Metal support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS (Metal) available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f'Using device: {device}')
    # Test tensor operation
    x = torch.randn(5, 3, device=device)
    print(f'Test tensor on Metal: {x.shape}')
"

echo
echo "✅ Installation complete!"
echo
echo "To verify the installation:"
echo "  source venv/bin/activate"
echo "  python -c 'import torch; print(torch.backends.mps.is_available())'"
echo
echo "Your M4 Max is ready for ML with Metal acceleration! 🎉"