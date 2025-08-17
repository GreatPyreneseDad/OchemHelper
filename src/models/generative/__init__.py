"""Enhanced generative models for molecular design with advanced computing."""

from .smiles_vae import MolecularVAE, SMILESTokenizer
from .molecular_reservoir_engine import (
    MolecularReservoirEngine, 
    MolecularReservoirConfig,
    ChemicalCoherenceDimensions,
    create_molecular_reservoir_engine
)

# Import error handling for optional dependencies
import logging
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. VAE functionality will be limited.")

class MoleculeGenerator:
    """Enhanced molecule generator with reservoir computing and VAE"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.vae_model = None
        self.reservoir_engine = None
        
        # Initialize components
        if TORCH_AVAILABLE:
            try:
                tokenizer = SMILESTokenizer()
                self.vae_model = MolecularVAE(vocab_size=tokenizer.vocab_size)
                logger.info("VAE model initialized")
            except Exception as e:
                logger.error(f"Error initializing VAE: {e}")
        
        try:
            self.reservoir_engine = create_molecular_reservoir_engine()
            logger.info("Molecular reservoir engine initialized")
        except Exception as e:
            logger.error(f"Error initializing reservoir engine: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = 'default') -> "MoleculeGenerator":
        """Load a pretrained model"""
        instance = cls()
        
        # Try to load VAE checkpoint
        from pathlib import Path
        vae_path = Path(f'models/pretrained/{model_name}_vae.pt')
        
        if vae_path.exists() and TORCH_AVAILABLE:
            try:
                import torch
                checkpoint = torch.load(vae_path, map_location='cpu')
                
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    instance.vae_model = MolecularVAE(
                        vocab_size=config['vocab_size'],
                        embedding_dim=config.get('embedding_dim', 128),
                        hidden_dim=config['hidden_dim'],
                        latent_dim=config['latent_dim'],
                        num_layers=config.get('num_layers', 2),
                        max_length=config['max_length'],
                        beta=config.get('beta', 1.0)
                    )
                    instance.vae_model.load_state_dict(checkpoint['model_state_dict'])
                    instance.vae_model.eval()
                    logger.info(f"Loaded pretrained VAE: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading pretrained VAE: {e}")
        
        return instance
    
    def generate(self, 
                n_molecules: int = 10,
                target_properties: dict = None,
                use_reservoir_guidance: bool = True,
                **kwargs) -> list:
        """Generate molecules using available methods"""
        molecules = []
        
        # Primary generation with VAE
        if self.vae_model and TORCH_AVAILABLE:
            try:
                device = kwargs.get('device', 'cpu')
                vae_molecules = self.vae_model.generate(n_molecules, device=device)
                molecules.extend(vae_molecules)
                logger.info(f"Generated {len(vae_molecules)} molecules with VAE")
            except Exception as e:
                logger.error(f"VAE generation error: {e}")
        
        # Enhancement with reservoir computing
        if use_reservoir_guidance and self.reservoir_engine and molecules:
            try:
                enhanced_molecules = []
                for smiles in molecules:
                    # Analyze with reservoir
                    properties = self.reservoir_engine.predict_molecular_properties(smiles)
                    
                    # Filter by target properties if specified
                    if target_properties:
                        if self._meets_targets(properties, target_properties):
                            enhanced_molecules.append(smiles)
                    else:
                        enhanced_molecules.append(smiles)
                
                molecules = enhanced_molecules
                logger.info(f"Reservoir guidance resulted in {len(molecules)} molecules")
                
            except Exception as e:
                logger.error(f"Reservoir guidance error: {e}")
        
        # Fallback: simple generation if no advanced methods available
        if not molecules:
            molecules = self._fallback_generation(n_molecules)
            logger.warning("Using fallback generation method")
        
        return molecules
    
    def _meets_targets(self, properties: dict, targets: dict) -> bool:
        """Check if molecule meets target properties"""
        for prop, target_value in targets.items():
            if prop in properties:
                actual_value = properties[prop]
                
                # Allow 20% tolerance
                if isinstance(target_value, (int, float)):
                    tolerance = abs(target_value * 0.2)
                    if abs(actual_value - target_value) > tolerance:
                        return False
                elif isinstance(target_value, list) and len(target_value) == 2:
                    min_val, max_val = target_value
                    if not (min_val <= actual_value <= max_val):
                        return False
        
        return True
    
    def _fallback_generation(self, n_molecules: int) -> list:
        """Fallback molecular generation"""
        # Simple drug-like SMILES templates
        templates = [
            'c1ccccc1',  # Benzene
            'CCO',       # Ethanol
            'CC(=O)O',   # Acetic acid
            'c1ccncc1',  # Pyridine
            'C1CCCCC1',  # Cyclohexane
            'CC(C)C',    # Isobutane
            'c1ccc2ccccc2c1',  # Naphthalene
            'CCN',       # Ethylamine
            'CCCCO',     # Butanol
            'c1ccc(cc1)O'  # Phenol
        ]
        
        import random
        molecules = []
        for _ in range(n_molecules):
            # Simple random selection and modification
            base = random.choice(templates)
            molecules.append(base)
        
        return molecules

__all__ = [
    "MolecularVAE",
    "SMILESTokenizer", 
    "MolecularReservoirEngine",
    "MolecularReservoirConfig",
    "ChemicalCoherenceDimensions",
    "MoleculeGenerator",
    "create_molecular_reservoir_engine"
]
