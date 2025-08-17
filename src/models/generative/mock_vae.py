"""Mock VAE for when PyTorch is not available."""

import random
import numpy as np
from typing import List, Optional

# Common drug-like SMILES patterns
DRUG_FRAGMENTS = [
    "c1ccccc1",  # benzene
    "C1CCCCC1",  # cyclohexane
    "C(=O)",     # carbonyl
    "C(=O)O",    # carboxylic acid
    "C(=O)N",    # amide
    "N",         # amine
    "O",         # ether/alcohol
    "S",         # sulfur
    "F",         # fluorine
    "Cl",        # chlorine
    "Br",        # bromine
    "C#N",       # nitrile
    "C=C",       # alkene
    "CC",        # ethyl
    "CCC",       # propyl
    "C(C)C",     # isopropyl
    "C(C)(C)C",  # tert-butyl
]

DRUG_TEMPLATES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CCO",  # Ethanol
    "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
    "CN1CCC(C(C1)O)(C2=CC=CC=C2)C3=CC=CC=C3",  # Similar to antihistamine
    "C1=CC=C(C=C1)C(C(=O)O)N",  # Phenylalanine
    "CC(C)(C)NCC(C1=CC(=CC=C1)O)O",  # Similar to beta blocker
]


class MockVAE:
    """Mock VAE that generates reasonable SMILES without PyTorch."""
    
    def __init__(self, vocab_size=100, embedding_dim=128, hidden_dim=256, 
                 latent_dim=56, num_layers=2, max_length=100, beta=1.0):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
    def generate(self, n_molecules: int, z: Optional[np.ndarray] = None) -> List[str]:
        """Generate mock molecules."""
        molecules = []
        
        for i in range(n_molecules):
            if random.random() < 0.7:  # 70% chance to use template
                # Start with a template and modify it
                template = random.choice(DRUG_TEMPLATES)
                molecule = self._modify_template(template)
            else:
                # Generate from fragments
                molecule = self._generate_from_fragments()
            
            molecules.append(molecule)
        
        return molecules
    
    def _modify_template(self, template: str) -> str:
        """Slightly modify a template molecule."""
        modifications = []
        
        # Random modifications
        if random.random() < 0.3:
            # Add a methyl group
            modifications.append(lambda s: s.replace("C", "C(C)", 1))
        
        if random.random() < 0.2:
            # Replace H with F
            modifications.append(lambda s: s.replace("C1", "C(F)1", 1))
        
        if random.random() < 0.2:
            # Add an amine
            modifications.append(lambda s: s + "N")
        
        # Apply modifications
        result = template
        for mod in modifications:
            try:
                result = mod(result)
            except:
                pass
        
        return result
    
    def _generate_from_fragments(self) -> str:
        """Generate a molecule from fragments."""
        n_fragments = random.randint(2, 5)
        fragments = [random.choice(DRUG_FRAGMENTS) for _ in range(n_fragments)]
        
        # Join fragments
        molecule = ""
        for i, frag in enumerate(fragments):
            if i > 0 and random.random() < 0.5:
                # Add a linker
                molecule += random.choice(["", "C", "CC", "O", "N"])
            molecule += frag
        
        return molecule
    
    def encode_smiles(self, smiles_list: List[str]) -> np.ndarray:
        """Mock encoding - returns random latent vectors."""
        n = len(smiles_list)
        return np.random.randn(n, self.latent_dim)
    
    def eval(self):
        """Mock eval mode."""
        pass


class MockSMILESTokenizer:
    """Mock tokenizer."""
    
    def __init__(self):
        self.vocab_size = 100
    
    def encode(self, smiles: str) -> List[int]:
        """Mock encoding."""
        return [ord(c) % self.vocab_size for c in smiles]
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decoding."""
        return "".join([chr(t + 65) for t in tokens])


# Create a mock that looks like the real VAE
class MolecularVAE(MockVAE):
    """Mock MolecularVAE that mimics the real one."""
    
    def load_state_dict(self, state_dict):
        """Mock loading."""
        pass


class SMILESTokenizer(MockSMILESTokenizer):
    """Mock SMILESTokenizer that mimics the real one."""
    pass