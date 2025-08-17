#!/usr/bin/env python
"""Generate molecules using trained models."""

import torch
import argparse
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Draw
import matplotlib.pyplot as plt

from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoleculeGeneratorCLI:
    """Command-line interface for molecule generation."""
    
    def __init__(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        config = checkpoint['model_config']
        self.model = MolecularVAE(
            vocab_size=config['vocab_size'],
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            num_layers=config.get('num_layers', 2),
            max_length=config['max_length'],
            beta=config.get('beta', 1.0)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    def generate_random(self, n_molecules: int) -> List[str]:
        """Generate random molecules from prior."""
        logger.info(f"Generating {n_molecules} random molecules...")
        return self.model.generate(n_molecules, device=self.device)
    
    def generate_similar(self, reference_smiles: str, n_molecules: int, variance: float = 0.1) -> List[str]:
        """Generate molecules similar to reference."""
        logger.info(f"Generating {n_molecules} molecules similar to {reference_smiles}")
        
        # Encode reference
        z_ref = self.model.encode_smiles([reference_smiles]).to(self.device)
        
        # Add noise to latent vector
        noise = torch.randn(n_molecules, z_ref.shape[1]).to(self.device) * variance
        z_samples = z_ref + noise
        
        # Generate
        return self.model.generate(n_molecules, z=z_samples, device=self.device)
    
    def interpolate(self, smiles1: str, smiles2: str, n_steps: int = 10) -> List[str]:
        """Interpolate between two molecules."""
        logger.info(f"Interpolating between {smiles1} and {smiles2}")
        return self.model.interpolate(smiles1, smiles2, n_steps)
    
    def generate_with_properties(
        self,
        n_molecules: int,
        target_properties: Dict[str, float],
        n_attempts: int = 10
    ) -> List[str]:
        """Generate molecules with target properties."""
        logger.info(f"Generating molecules with properties: {target_properties}")
        
        all_molecules = []
        
        for _ in range(n_attempts):
            # Generate batch
            molecules = self.model.generate(n_molecules * 10, device=self.device)
            
            # Filter by properties
            filtered = self._filter_by_properties(molecules, target_properties)
            all_molecules.extend(filtered)
            
            if len(all_molecules) >= n_molecules:
                break
        
        return all_molecules[:n_molecules]
    
    def _filter_by_properties(
        self,
        molecules: List[str],
        target_properties: Dict[str, float],
        tolerance: float = 0.2
    ) -> List[str]:
        """Filter molecules by target properties."""
        filtered = []
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate properties
            props = {
                'MW': Descriptors.MolWt(mol),
                'logP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'rotatable': Descriptors.NumRotatableBonds(mol),
                'rings': Descriptors.RingCount(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol)
            }
            
            # Check if matches target
            match = True
            for prop_name, target_value in target_properties.items():
                if prop_name in props:
                    actual_value = props[prop_name]
                    
                    # Different tolerance for different properties
                    if prop_name in ['MW']:
                        if abs(actual_value - target_value) / target_value > tolerance:
                            match = False
                    elif prop_name in ['logP', 'TPSA']:
                        if abs(actual_value - target_value) > target_value * tolerance:
                            match = False
                    else:  # Integer properties
                        if abs(actual_value - target_value) > 1:
                            match = False
            
            if match:
                filtered.append(smiles)
        
        return filtered
    
    def analyze_molecules(self, molecules: List[str]) -> pd.DataFrame:
        """Analyze generated molecules."""
        data = []
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate properties
            props = {
                'SMILES': smiles,
                'MW': Descriptors.MolWt(mol),
                'logP': Crippen.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'QED': Descriptors.qed(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'Rotatable': Descriptors.NumRotatableBonds(mol),
                'Rings': Descriptors.RingCount(mol),
                'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
                'SA_Score': self._calculate_sa_score(mol),
                'Lipinski': self._check_lipinski(mol)
            }
            
            data.append(props)
        
        return pd.DataFrame(data)
    
    def _calculate_sa_score(self, mol) -> float:
        """Calculate synthetic accessibility score."""
        # Simplified SA score calculation
        # In production, use rdkit.Chem.rdMolDescriptors.CalcNumBridgeheadAtoms
        # or external SA score calculators
        
        # Simple heuristic based on complexity
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        n_rings = Descriptors.RingCount(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol))
        
        complexity = n_atoms + n_bonds * 0.5 + n_rings * 2 + n_stereo * 3
        
        # Normalize to 1-10 scale (1=easy, 10=hard)
        sa_score = min(10, max(1, complexity / 10))
        
        return sa_score
    
    def _check_lipinski(self, mol) -> bool:
        """Check Lipinski's Rule of Five."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        
        return violations <= 1  # Allow one violation
    
    def visualize_molecules(self, molecules: List[str], output_file: str = 'molecules.png'):
        """Create grid visualization of molecules."""
        mols = []
        for smiles in molecules[:12]:  # Limit to 12 for visualization
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
        
        if mols:
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=4,
                subImgSize=(200, 200),
                legends=[Chem.MolToSmiles(mol) for mol in mols]
            )
            img.save(output_file)
            logger.info(f"Saved visualization to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate molecules using trained VAE')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'similar', 'interpolate', 'properties'],
                        help='Generation mode')
    parser.add_argument('--n-molecules', type=int, default=100, help='Number of molecules to generate')
    parser.add_argument('--reference', type=str, help='Reference SMILES for similar mode')
    parser.add_argument('--smiles1', type=str, help='First SMILES for interpolation')
    parser.add_argument('--smiles2', type=str, help='Second SMILES for interpolation')
    parser.add_argument('--properties', type=str, help='Target properties as JSON string')
    parser.add_argument('--output', type=str, default='generated_molecules.csv', help='Output file')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--analyze', action='store_true', help='Analyze generated molecules')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MoleculeGeneratorCLI(args.checkpoint)
    
    # Generate molecules based on mode
    if args.mode == 'random':
        molecules = generator.generate_random(args.n_molecules)
    
    elif args.mode == 'similar':
        if not args.reference:
            raise ValueError("Reference SMILES required for similar mode")
        molecules = generator.generate_similar(args.reference, args.n_molecules)
    
    elif args.mode == 'interpolate':
        if not args.smiles1 or not args.smiles2:
            raise ValueError("Two SMILES required for interpolation")
        molecules = generator.interpolate(args.smiles1, args.smiles2, args.n_molecules)
    
    elif args.mode == 'properties':
        if not args.properties:
            raise ValueError("Target properties required")
        target_props = json.loads(args.properties)
        molecules = generator.generate_with_properties(args.n_molecules, target_props)
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    logger.info(f"Generated {len(molecules)} valid molecules")
    
    # Analyze if requested
    if args.analyze:
        df = generator.analyze_molecules(molecules)
        print("\nMolecule Statistics:")
        print(df.describe())
        print(f"\nLipinski compliant: {df['Lipinski'].sum()}/{len(df)}")
        print(f"Average QED: {df['QED'].mean():.3f}")
    else:
        df = pd.DataFrame({'SMILES': molecules})
    
    # Save results
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} molecules to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        output_img = args.output.replace('.csv', '.png')
        generator.visualize_molecules(molecules, output_img)
    
    # Print sample molecules
    print("\nSample generated molecules:")
    for i, smiles in enumerate(molecules[:5], 1):
        print(f"{i}. {smiles}")


if __name__ == '__main__':
    main()
