#!/usr/bin/env python
"""Test script for OChem Helper functionality."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer
from src.core.molecular_graph import MolecularGraph
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tokenizer():
    """Test SMILES tokenizer."""
    logger.info("Testing SMILES tokenizer...")
    
    tokenizer = SMILESTokenizer()
    
    test_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C",  # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",   # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    for smiles in test_smiles:
        tokens = tokenizer.tokenize(smiles)
        reconstructed = tokenizer.detokenize(tokens)
        logger.info(f"Original: {smiles}")
        logger.info(f"Tokens: {tokens[:10]}...")
        logger.info(f"Reconstructed: {reconstructed}")
        
        # Verify with RDKit
        mol_orig = Chem.MolFromSmiles(smiles)
        mol_recon = Chem.MolFromSmiles(reconstructed)
        
        if mol_orig and mol_recon:
            canonical_orig = Chem.MolToSmiles(mol_orig)
            canonical_recon = Chem.MolToSmiles(mol_recon)
            logger.info(f"Canonical match: {canonical_orig == canonical_recon}\n")
    
    logger.info("Tokenizer test completed!\n")


def test_molecular_graph():
    """Test molecular graph conversion."""
    logger.info("Testing molecular graph representation...")
    
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    # Create molecular graph
    mol_graph = MolecularGraph.from_smiles(smiles)
    
    logger.info(f"SMILES: {smiles}")
    logger.info(f"Number of atoms: {mol_graph.num_atoms}")
    logger.info(f"Number of bonds: {mol_graph.num_bonds}")
    
    # Convert to PyTorch Geometric Data
    graph_data = mol_graph.to_graph()
    
    logger.info(f"Node features shape: {graph_data.x.shape}")
    logger.info(f"Edge indices shape: {graph_data.edge_index.shape}")
    logger.info(f"Edge features shape: {graph_data.edge_attr.shape}")
    
    logger.info("Molecular graph test completed!\n")


def test_vae_model():
    """Test VAE model initialization and basic operations."""
    logger.info("Testing VAE model...")
    
    tokenizer = SMILESTokenizer()
    model = MolecularVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        latent_dim=56,
        num_layers=1,
        max_length=50
    )
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test generation
    logger.info("Testing molecule generation...")
    molecules = model.generate(5)
    
    for i, smiles in enumerate(molecules, 1):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logger.info(f"{i}. {smiles} (MW: {mw:.2f})")
    
    # Test encoding
    if molecules:
        logger.info("\nTesting encoding...")
        z = model.encode_smiles(molecules[:1])
        logger.info(f"Latent vector shape: {z.shape}")
    
    logger.info("VAE model test completed!\n")


def test_data_pipeline():
    """Test data download and preparation."""
    logger.info("Testing data pipeline...")
    
    from scripts.download_data import MolecularDatasetDownloader
    
    downloader = MolecularDatasetDownloader('data')
    
    # Generate small synthetic dataset
    logger.info("Generating synthetic dataset...")
    smiles_list = downloader._generate_drug_like_smiles(100)
    
    valid_count = 0
    for smiles in smiles_list[:10]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_count += 1
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            logger.info(f"  {smiles[:30]}... MW={mw:.1f}, LogP={logp:.2f}")
    
    logger.info(f"Valid molecules: {valid_count}/10")
    logger.info("Data pipeline test completed!\n")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("OChem Helper Test Suite")
    logger.info("=" * 60 + "\n")
    
    try:
        test_tokenizer()
        test_molecular_graph()
        test_vae_model()
        test_data_pipeline()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
