"""Generative models for molecular design."""

import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MoleculeGenerator:
    """Base class for molecule generation models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        
    @classmethod
    def from_pretrained(cls, model_name: str) -> "MoleculeGenerator":
        """Load a pretrained model."""
        instance = cls()
        
        if model_name == 'vae-chembl':
            # Load pretrained VAE
            from .smiles_vae import MolecularVAE, SMILESTokenizer
            
            checkpoint_path = Path('models/pretrained/vae_chembl.pt')
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Create model with saved config
                config = checkpoint['model_config']
                instance.model = MolecularVAE(
                    vocab_size=config['vocab_size'],
                    embedding_dim=config.get('embedding_dim', 128),
                    hidden_dim=config['hidden_dim'],
                    latent_dim=config['latent_dim'],
                    num_layers=config.get('num_layers', 2),
                    max_length=config['max_length'],
                    beta=config.get('beta', 1.0)
                )
                
                # Load weights
                instance.model.load_state_dict(checkpoint['model_state_dict'])
                instance.model.eval()
                
                logger.info(f"Loaded pretrained VAE from {checkpoint_path}")
            else:
                logger.warning(f"Pretrained model not found at {checkpoint_path}")
                # Initialize with default VAE
                from .smiles_vae import MolecularVAE, SMILESTokenizer
                tokenizer = SMILESTokenizer()
                instance.model = MolecularVAE(vocab_size=tokenizer.vocab_size)
        
        return instance
    
    def generate(
        self,
        n_molecules: int,
        target_properties: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[str]:
        """Generate new molecules."""
        if self.model is None:
            logger.warning("No model loaded, returning empty list")
            return []
        
        # Generate using VAE
        if hasattr(self.model, 'generate'):
            device = kwargs.get('device', 'cpu')
            molecules = self.model.generate(n_molecules, device=device)
            
            # Filter by target properties if specified
            if target_properties:
                molecules = self._filter_by_properties(molecules, target_properties)
            
            return molecules
        
        return []
    
    def _filter_by_properties(
        self,
        molecules: List[str],
        target_properties: Dict[str, float],
        tolerance: float = 0.2
    ) -> List[str]:
        """Filter molecules by target properties."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        
        filtered = []
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Check properties
            match = True
            
            if 'MW' in target_properties:
                mw = Descriptors.MolWt(mol)
                target_mw = target_properties['MW']
                if abs(mw - target_mw) / target_mw > tolerance:
                    match = False
            
            if 'logP' in target_properties:
                logp = Crippen.MolLogP(mol)
                target_logp = target_properties['logP']
                if abs(logp - target_logp) > tolerance * 5:  # LogP tolerance
                    match = False
            
            if match:
                filtered.append(smiles)
        
        return filtered
