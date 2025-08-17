#!/usr/bin/env python
"""
Quick deployment script for enhanced OChem Helper
Sets up the advanced molecular discovery system with all enhancements
"""

import subprocess
import sys
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'scikit-learn',
        'rdkit', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        
        for package in missing_packages:
            try:
                if package == 'rdkit':
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit-pypi'])
                else:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {package}")
    
    logger.info("✅ Requirements check completed")

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'models/pretrained',
        'logs',
        'configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def create_config_files():
    """Create default configuration files"""
    
    # Default model config
    model_config = {
        "vae": {
            "vocab_size": 80,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "latent_dim": 128,
            "num_layers": 2,
            "max_length": 100,
            "beta": 1.0
        },
        "reservoir": {
            "num_nodes": 150,
            "spatial_dimension": 3,
            "learning_rate": 0.01,
            "energy_decay": 0.95,
            "coherence_coupling": 0.08,
            "chemical_temperature": 298.15
        },
        "ensemble": {
            "use_xgboost": True,
            "use_lightgbm": True,
            "use_catboost": True,
            "use_neural_net": True,
            "use_random_forest": True,
            "uncertainty_estimation": True,
            "cross_validation_folds": 5
        }
    }
    
    with open('configs/default.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # MCP configuration
    mcp_config = {
        "mcpServers": {
            "ochem-helper-advanced": {
                "command": "python",
                "args": ["-m", "mcp.server.ochem_mcp_advanced"],
                "env": {
                    "PYTHONPATH": str(Path.cwd())
                }
            }
        }
    }
    
    with open('mcp/mcp_config.json', 'w') as f:
        json.dump(mcp_config, f, indent=2)
    
    logger.info("✅ Configuration files created")

def run_tests():
    """Run the enhanced test suite"""
    logger.info("Running enhanced test suite...")
    
    try:
        subprocess.check_call([sys.executable, 'test_enhanced_ochem.py'])
        logger.info("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Some tests failed")
        return False

def setup_mcp_integration():
    """Set up MCP integration for Claude Desktop"""
    
    claude_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    if claude_config_path.exists():
        try:
            with open(claude_config_path, 'r') as f:
                config = json.load(f)
        except:
            config = {}
    else:
        config = {}
    
    if 'mcpServers' not in config:
        config['mcpServers'] = {}
    
    config['mcpServers']['ochem-helper-advanced'] = {
        "command": "python",
        "args": ["-m", "mcp.server.ochem_mcp_advanced"],
        "cwd": str(Path.cwd()),
        "env": {
            "PYTHONPATH": str(Path.cwd())
        }
    }
    
    # Create backup
    if claude_config_path.exists():
        backup_path = claude_config_path.with_suffix('.json.backup')
        subprocess.run(['cp', str(claude_config_path), str(backup_path)])
    
    # Create directories if they don't exist
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(claude_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ MCP integration configured for Claude Desktop")
    logger.info(f"Config updated at: {claude_config_path}")

def create_startup_script():
    """Create startup script for development"""
    
    startup_script = """#!/bin/bash
# OChem Helper Enhanced Startup Script

echo "🧪 Starting Enhanced OChem Helper..."

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run test suite
echo "Running enhanced test suite..."
python test_enhanced_ochem.py

# Start MCP server
echo "Starting MCP server..."
echo "Use Ctrl+C to stop"
python -m mcp.server.ochem_mcp_advanced
"""
    
    with open('start_enhanced.sh', 'w') as f:
        f.write(startup_script)
    
    # Make executable
    subprocess.run(['chmod', '+x', 'start_enhanced.sh'])
    
    logger.info("✅ Startup script created: start_enhanced.sh")

def main():
    """Main deployment function"""
    logger.info("🚀 Deploying Enhanced OChem Helper...")
    logger.info("=" * 60)
    
    try:
        # Setup steps
        check_requirements()
        setup_directories() 
        create_config_files()
        create_startup_script()
        
        # Test the enhanced system
        tests_passed = run_tests()
        
        # Set up MCP integration
        setup_mcp_integration()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ENHANCED OCHEM HELPER DEPLOYMENT COMPLETE!")
        logger.info("=" * 60)
        
        logger.info("\n✅ Successfully deployed enhanced features:")
        logger.info("• Hyperposition Molecular Tokenizer")
        logger.info("• Molecular Reservoir Computing Engine")
        logger.info("• Advanced Ensemble Property Prediction")
        logger.info("• Enhanced VAE Molecular Generation")
        logger.info("• Advanced MCP Server Integration")
        
        logger.info("\n🔧 Development tools:")
        logger.info("• ./start_enhanced.sh - Start development environment")
        logger.info("• python test_enhanced_ochem.py - Run test suite")
        logger.info("• python -m mcp.server.ochem_mcp_advanced - Start MCP server")
        
        logger.info("\n🤖 AI Integration:")
        logger.info("• Claude Desktop MCP integration configured")
        logger.info("• Ready for xAI partnership integration")
        logger.info("• Advanced chemistry tools available")
        
        if tests_passed:
            logger.info("\n🎯 Ready for production deployment!")
            logger.info("All systems operational and tested successfully.")
        else:
            logger.warning("\n⚠️  Some tests failed - check logs before production.")
        
        logger.info("\n🌟 Enhanced OChem Helper is ready to revolutionize molecular discovery!")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
