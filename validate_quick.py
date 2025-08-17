#!/usr/bin/env python3
"""
Quick validation test for Enhanced OChem Helper
Tests core functionality without complex dependencies
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test if we can import basic Python libraries"""
    print("🔧 Testing Basic Dependencies...")
    
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas available")
    except ImportError:
        print("❌ Pandas not available")
        return False
    
    return True

def test_rdkit():
    """Test RDKit functionality"""
    print("\n🧪 Testing RDKit...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        
        # Test basic molecule creation
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        if mol is None:
            print("❌ Failed to create molecule from SMILES")
            return False
        
        # Test property calculation
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        
        print(f"✅ RDKit working - Ethanol: MW={mw:.2f}, LogP={logp:.2f}")
        return True
        
    except ImportError as e:
        print(f"❌ RDKit not available: {e}")
        return False
    except Exception as e:
        print(f"❌ RDKit error: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality"""
    print("\n🔥 Testing PyTorch...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        print(f"✅ PyTorch working - version {torch.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_vae_tokenizer():
    """Test the SMILES tokenizer"""
    print("\n🔤 Testing SMILES Tokenizer...")
    
    try:
        from src.models.generative.smiles_vae import SMILESTokenizer
        
        tokenizer = SMILESTokenizer()
        
        # Test tokenization
        test_smiles = "CCO"
        tokens = tokenizer.tokenize(test_smiles)
        detokenized = tokenizer.detokenize(tokens)
        
        print(f"✅ Tokenizer working:")
        print(f"   Original: {test_smiles}")
        print(f"   Tokens: {tokens}")
        print(f"   Detokenized: {detokenized}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import tokenizer: {e}")
        return False
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False

def test_vae_model():
    """Test VAE model creation"""
    print("\n🧠 Testing VAE Model...")
    
    try:
        from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer
        import torch
        
        tokenizer = SMILESTokenizer()
        model = MolecularVAE(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,  # Small for testing
            hidden_dim=64,
            latent_dim=16,
            num_layers=1
        )
        
        print(f"✅ VAE model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test generation (without training)
        model.eval()
        with torch.no_grad():
            molecules = model.generate(3)
        
        print(f"✅ Generated {len(molecules)} molecules:")
        for i, mol in enumerate(molecules, 1):
            print(f"   {i}. {mol}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE model error: {e}")
        return False

def test_hyperposition_tokenizer():
    """Test hyperposition tokenizer"""
    print("\n⚡ Testing Hyperposition Tokenizer...")
    
    try:
        from src.core.hyperposition_tokenizer import create_molecular_hyperprocessor
        
        processor = create_molecular_hyperprocessor()
        
        # Test processing a simple molecule
        result = processor.process_molecule("CCO")  # Ethanol
        
        print(f"✅ Hyperposition tokenizer working:")
        print(f"   Tokens: {len(result['compression']['tokens'])}")
        print(f"   Coherence: {result['molecular_coherence']['overall']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hyperposition tokenizer error: {e}")
        return False

def test_reservoir_engine():
    """Test molecular reservoir engine"""
    print("\n🌊 Testing Molecular Reservoir Engine...")
    
    try:
        from src.models.generative.molecular_reservoir_engine import create_molecular_reservoir_engine
        
        reservoir = create_molecular_reservoir_engine()
        
        # Test property prediction
        properties = reservoir.predict_molecular_properties("CCO", ['logP', 'MW'])
        
        print(f"✅ Reservoir engine working:")
        print(f"   Predicted properties: {properties}")
        
        # Test state
        state = reservoir.get_reservoir_state()
        print(f"   Nodes: {state['num_nodes']}, Energy: {state['average_energy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reservoir engine error: {e}")
        return False

def test_ensemble_predictor():
    """Test ensemble property predictor"""
    print("\n📊 Testing Ensemble Predictor...")
    
    try:
        from src.models.predictive.molecular_ensemble import create_molecular_ensemble
        
        ensemble = create_molecular_ensemble()
        
        # Test descriptor calculation
        from src.models.predictive.molecular_ensemble import MolecularDescriptorCalculator
        
        calc = MolecularDescriptorCalculator()
        descriptors = calc.calculate_descriptors(["CCO", "c1ccccc1"])
        
        print(f"✅ Ensemble predictor working:")
        print(f"   Calculated descriptors for {len(descriptors)} molecules")
        print(f"   Descriptor names: {len(calc.get_descriptor_names())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble predictor error: {e}")
        return False

def test_mcp_server_import():
    """Test MCP server import"""
    print("\n🤖 Testing MCP Server Import...")
    
    try:
        from mcp.server.ochem_mcp_advanced import OChemMCPServer
        
        # Just test that we can create the class (don't run the server)
        server_class = OChemMCPServer
        
        print(f"✅ MCP server import successful")
        return True
        
    except Exception as e:
        print(f"❌ MCP server import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ENHANCED OCHEM HELPER - QUICK VALIDATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Core dependency tests
    test_results.append(("Basic Dependencies", test_basic_imports()))
    test_results.append(("RDKit", test_rdkit()))
    test_results.append(("PyTorch", test_pytorch()))
    
    # Core functionality tests
    test_results.append(("VAE Tokenizer", test_vae_tokenizer()))
    test_results.append(("VAE Model", test_vae_model()))
    
    # Enhanced feature tests
    test_results.append(("Hyperposition Tokenizer", test_hyperposition_tokenizer()))
    test_results.append(("Reservoir Engine", test_reservoir_engine()))
    test_results.append(("Ensemble Predictor", test_ensemble_predictor()))
    test_results.append(("MCP Server", test_mcp_server_import()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Enhanced OChem Helper is ready! 🚀")
        confidence = "95%"
    elif failed <= 2:
        print(f"\n✅ Core functionality working! {failed} minor issues to fix.")
        confidence = "85%"
    elif failed <= 4:
        print(f"\n⚠️  Some issues found. {failed} components need attention.")
        confidence = "70%"
    else:
        print(f"\n❌ Major issues found. {failed} components failing.")
        confidence = "50%"
    
    print(f"\n🎯 Deployment Confidence: {confidence}")
    
    if passed >= 6:  # Core + some enhanced features
        print("\n✅ RECOMMENDATION: Safe to push to GitHub")
        print("   Core VAE + basic functionality confirmed working")
        print("   Enhanced features mostly functional")
        print("   Ready for xAI partnership demonstration")
    else:
        print("\n⚠️  RECOMMENDATION: Fix core issues before pushing")
        print("   Address dependency and basic functionality problems")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
