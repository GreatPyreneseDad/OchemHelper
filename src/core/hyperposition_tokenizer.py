"""
Hyperposition Molecular Tokenizer - Advanced Chemical Representation
Adapted from HyperPosition Neural Network for molecular tokenization

Implements hyperposition tokens for molecular representation with chemical
coherence dimensions and skip-trace analysis for molecular patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Molecular analysis will be limited.")

logger = logging.getLogger(__name__)

class ChemicalTokenType(Enum):
    """Chemical token types for universal classification"""
    ATOM = "ATOM"
    BOND = "BOND"
    RING = "RING"
    FUNCTIONAL_GROUP = "FUNCTIONAL_GROUP"
    STEREOCHEMISTRY = "STEREOCHEMISTRY"
    CHARGE = "CHARGE"
    AROMATIC = "AROMATIC"
    HETEROATOM = "HETEROATOM"

class ChemicalTransform(Enum):
    """Chemical transforms for molecular modifications"""
    IDENTITY = "IDENTITY"
    OXIDATION = "OXIDATION" 
    REDUCTION = "REDUCTION"
    SUBSTITUTION = "SUBSTITUTION"
    ADDITION = "ADDITION"
    ELIMINATION = "ELIMINATION"
    CYCLIZATION = "CYCLIZATION"
    RING_OPENING = "RING_OPENING"

@dataclass
class ChemicalHyperDimensions:
    """Enhanced hyperposition dimensions for chemistry"""
    electronic: float = 0.5      # Electronic properties (electronegativity, etc.)
    steric: float = 0.5          # Steric properties (size, bulk)
    reactivity: float = 0.5      # Chemical reactivity
    stability: float = 0.5       # Thermodynamic stability
    polarity: float = 0.5        # Molecular polarity
    hydrophobicity: float = 0.5  # Hydrophobic character
    aromaticity: float = 0.5     # Aromatic character
    chirality: float = 0.5       # Stereochemical properties
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.electronic, self.steric, self.reactivity, self.stability,
            self.polarity, self.hydrophobicity, self.aromaticity, self.chirality
        ])
    
    def from_array(self, arr: np.ndarray):
        """Update from numpy array"""
        if len(arr) >= 8:
            self.electronic = arr[0]
            self.steric = arr[1]
            self.reactivity = arr[2]
            self.stability = arr[3]
            self.polarity = arr[4]
            self.hydrophobicity = arr[5]
            self.aromaticity = arr[6]
            self.chirality = arr[7]

class MolecularHyperToken:
    """Hyperposition token for molecular fragments"""
    
    def __init__(self, 
                 smiles_fragment: str,
                 token_type: ChemicalTokenType,
                 transform: ChemicalTransform = ChemicalTransform.IDENTITY):
        self.smiles_fragment = smiles_fragment
        self.token_type = token_type
        self.transform = transform
        
        # Hyperposition dimensions
        self.dimensions = ChemicalHyperDimensions()
        
        # Quantum superposition state
        self.is_collapsed = False
        self.superposition_weights = {}
        
        # Chemical connections
        self.connections: Dict['MolecularHyperToken', float] = {}
        
        # Initialize dimensions based on fragment
        self._initialize_dimensions()
    
    def _initialize_dimensions(self):
        """Initialize hyperposition dimensions based on chemical fragment"""
        if not RDKIT_AVAILABLE:
            return
        
        try:
            # Parse SMILES fragment
            mol = Chem.MolFromSmiles(self.smiles_fragment)
            if mol is None:
                # Use atom/bond-level analysis
                self._initialize_from_string()
                return
            
            # Calculate electronic properties
            if mol.GetNumAtoms() > 0:
                atom = mol.GetAtomWithIdx(0)
                atomic_num = atom.GetAtomicNum()
                
                # Electronic dimension based on electronegativity
                electronegativity_map = {
                    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
                }
                electronegativity = electronegativity_map.get(atomic_num, 2.0)
                self.dimensions.electronic = min(1.0, electronegativity / 4.0)
                
                # Steric dimension based on atomic radius
                atomic_radius_map = {
                    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                    15: 1.07, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39
                }
                radius = atomic_radius_map.get(atomic_num, 1.0)
                self.dimensions.steric = min(1.0, radius / 1.5)
                
                # Aromaticity
                self.dimensions.aromaticity = 1.0 if atom.GetIsAromatic() else 0.0
                
                # Charge influence on polarity
                formal_charge = atom.GetFormalCharge()
                self.dimensions.polarity = min(1.0, abs(formal_charge) / 2.0 + 0.3)
            
            # Reactivity based on unsaturation and heteroatoms
            unsaturation = Descriptors.BertzCT(mol) / max(1, mol.GetNumAtoms())
            self.dimensions.reactivity = min(1.0, unsaturation / 2.0)
            
            # Stability (inverse of reactivity for simple approximation)
            self.dimensions.stability = 1.0 - self.dimensions.reactivity * 0.5
            
            # Hydrophobicity estimation
            if mol.GetNumAtoms() > 0:
                try:
                    logp_contrib = Crippen.MolLogP(mol) / mol.GetNumAtoms()
                    self.dimensions.hydrophobicity = (logp_contrib + 2.0) / 4.0
                    self.dimensions.hydrophobicity = np.clip(self.dimensions.hydrophobicity, 0.0, 1.0)
                except:
                    self.dimensions.hydrophobicity = 0.5
            
        except Exception as e:
            logger.warning(f"Error initializing dimensions for {self.smiles_fragment}: {e}")
            self._initialize_default_dimensions()
    
    def _initialize_from_string(self):
        """Initialize dimensions from SMILES string analysis"""
        fragment = self.smiles_fragment.lower()
        
        # Electronic properties based on atoms
        if 'f' in fragment or 'cl' in fragment or 'br' in fragment or 'i' in fragment:
            self.dimensions.electronic = 0.8  # Electronegative halogens
        elif 'o' in fragment or 'n' in fragment:
            self.dimensions.electronic = 0.7  # Electronegative heteroatoms
        elif 'c' in fragment:
            self.dimensions.electronic = 0.5  # Carbon baseline
        
        # Aromaticity
        if 'c' in fragment and fragment != fragment.upper():
            self.dimensions.aromaticity = 1.0  # Lowercase = aromatic
        
        # Polarity
        if any(char in fragment for char in ['o', 'n', 'f', 'cl', 'br']):
            self.dimensions.polarity = 0.8
        
        # Reactivity based on multiple bonds and functional groups
        if '=' in fragment or '#' in fragment:
            self.dimensions.reactivity = 0.7
        elif any(group in fragment for group in ['[o]', '[n]', 'c=o']):
            self.dimensions.reactivity = 0.8
        
        # Default other dimensions
        self.dimensions.steric = 0.5
        self.dimensions.stability = 1.0 - self.dimensions.reactivity * 0.3
        self.dimensions.hydrophobicity = 0.3 if self.dimensions.polarity > 0.6 else 0.7
        self.dimensions.chirality = 0.1 if '@' in self.smiles_fragment else 0.0
    
    def _initialize_default_dimensions(self):
        """Initialize with default values"""
        self.dimensions = ChemicalHyperDimensions()
        
        # Set based on token type
        if self.token_type == ChemicalTokenType.AROMATIC:
            self.dimensions.aromaticity = 1.0
            self.dimensions.stability = 0.8
        elif self.token_type == ChemicalTokenType.HETEROATOM:
            self.dimensions.electronic = 0.8
            self.dimensions.polarity = 0.7
        elif self.token_type == ChemicalTokenType.FUNCTIONAL_GROUP:
            self.dimensions.reactivity = 0.8
            self.dimensions.polarity = 0.6
    
    def normalize_dimensions(self):
        """Normalize hyperposition dimensions"""
        dims = self.dimensions.to_array()
        norm = np.linalg.norm(dims)
        if norm > 0:
            dims = dims / norm
            self.dimensions.from_array(dims)
    
    def resonance_with(self, other: 'MolecularHyperToken') -> float:
        """Calculate chemical resonance with another token"""
        if not isinstance(other, MolecularHyperToken):
            return 0.0
        
        self_dims = self.dimensions.to_array()
        other_dims = other.dimensions.to_array()
        
        # Cosine similarity in hyperspace
        dot_product = np.dot(self_dims, other_dims)
        norms = np.linalg.norm(self_dims) * np.linalg.norm(other_dims)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        
        # Chemical compatibility bonus
        compatibility_bonus = self._chemical_compatibility(other)
        
        return max(0.0, min(1.0, similarity + compatibility_bonus))
    
    def _chemical_compatibility(self, other: 'MolecularHyperToken') -> float:
        """Calculate chemical compatibility bonus"""
        bonus = 0.0
        
        # Electrophile-nucleophile attraction
        if (self.dimensions.electronic > 0.7 and other.dimensions.electronic < 0.3) or \
           (self.dimensions.electronic < 0.3 and other.dimensions.electronic > 0.7):
            bonus += 0.2
        
        # Aromatic stacking
        if self.dimensions.aromaticity > 0.8 and other.dimensions.aromaticity > 0.8:
            bonus += 0.15
        
        # Hydrophobic interactions
        if self.dimensions.hydrophobicity > 0.7 and other.dimensions.hydrophobicity > 0.7:
            bonus += 0.1
        
        # Polar interactions
        if self.dimensions.polarity > 0.7 and other.dimensions.polarity > 0.7:
            bonus += 0.1
        
        return bonus
    
    def add_connection(self, other: 'MolecularHyperToken', strength: float):
        """Add connection to another token"""
        self.connections[other] = strength
    
    def collapse(self, context: Dict[str, Any]):
        """Collapse quantum superposition based on chemical context"""
        if self.is_collapsed:
            return
        
        # Analyze chemical context
        ph = context.get('ph', 7.0)
        temperature = context.get('temperature', 298.15)
        solvent_polarity = context.get('solvent_polarity', 0.5)
        
        # Adjust dimensions based on context
        context_adjustments = self._calculate_context_adjustments(ph, temperature, solvent_polarity)
        
        # Apply adjustments
        current_dims = self.dimensions.to_array()
        adjusted_dims = current_dims + context_adjustments
        adjusted_dims = np.clip(adjusted_dims, 0.0, 1.0)
        self.dimensions.from_array(adjusted_dims)
        
        self.is_collapsed = True
    
    def _calculate_context_adjustments(self, ph: float, temperature: float, solvent_polarity: float) -> np.ndarray:
        """Calculate context-based dimension adjustments"""
        adjustments = np.zeros(8)
        
        # pH effects on ionizable groups
        if self.token_type == ChemicalTokenType.FUNCTIONAL_GROUP:
            if ph < 3:  # Acidic conditions
                adjustments[4] += 0.1  # Increase polarity
                adjustments[2] -= 0.1  # Decrease reactivity
            elif ph > 11:  # Basic conditions
                adjustments[4] += 0.2  # Increase polarity
                adjustments[2] += 0.1  # Increase reactivity
        
        # Temperature effects
        temp_factor = (temperature - 298.15) / 298.15
        adjustments[2] += temp_factor * 0.1  # Reactivity increases with temperature
        adjustments[3] -= abs(temp_factor) * 0.05  # Stability decreases with extreme temperatures
        
        # Solvent effects
        if solvent_polarity > 0.7:  # Polar solvent
            adjustments[4] += 0.1  # Enhance polarity
            adjustments[5] -= 0.2  # Reduce hydrophobicity
        elif solvent_polarity < 0.3:  # Nonpolar solvent
            adjustments[5] += 0.1  # Enhance hydrophobicity
            adjustments[4] -= 0.1  # Reduce polarity
        
        return adjustments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'smiles_fragment': self.smiles_fragment,
            'token_type': self.token_type.value,
            'transform': self.transform.value,
            'dimensions': {
                'electronic': self.dimensions.electronic,
                'steric': self.dimensions.steric,
                'reactivity': self.dimensions.reactivity,
                'stability': self.dimensions.stability,
                'polarity': self.dimensions.polarity,
                'hydrophobicity': self.dimensions.hydrophobicity,
                'aromaticity': self.dimensions.aromaticity,
                'chirality': self.dimensions.chirality
            },
            'is_collapsed': self.is_collapsed
        }

class MolecularSkipTraceEngine:
    """Skip-trace engine for molecular pattern analysis"""
    
    def __init__(self, hyper_tokens: List[MolecularHyperToken], skip_threshold: float = 0.3):
        self.hyper_tokens = hyper_tokens
        self.skip_threshold = skip_threshold
        self.traces = []
        
    def generate_traces(self) -> List[Dict[str, Any]]:
        """Generate skip traces through molecular hyperspace"""
        self.traces = []
        
        if not self.hyper_tokens:
            return self.traces
        
        # Start from each token
        for start_idx, start_token in enumerate(self.hyper_tokens):
            trace = self._generate_trace_from_token(start_idx, start_token)
            if len(trace['path']) > 1:
                self.traces.append(trace)
        
        # Sort by coherence
        self.traces.sort(key=lambda x: x['coherence'], reverse=True)
        
        return self.traces
    
    def _generate_trace_from_token(self, start_idx: int, start_token: MolecularHyperToken) -> Dict[str, Any]:
        """Generate trace starting from a specific token"""
        path = [start_token]
        visited = {start_idx}
        coherence_scores = []
        
        current_token = start_token
        current_idx = start_idx
        
        # Follow connections with highest resonance
        max_hops = min(10, len(self.hyper_tokens))
        
        for hop in range(max_hops):
            best_next_idx = None
            best_resonance = 0.0
            
            # Find best next token
            for next_idx, next_token in enumerate(self.hyper_tokens):
                if next_idx in visited:
                    continue
                
                resonance = current_token.resonance_with(next_token)
                
                # Apply skip threshold
                if resonance > self.skip_threshold and resonance > best_resonance:
                    best_resonance = resonance
                    best_next_idx = next_idx
            
            # If no valid next token found, end trace
            if best_next_idx is None:
                break
            
            # Add to path
            next_token = self.hyper_tokens[best_next_idx]
            path.append(next_token)
            visited.add(best_next_idx)
            coherence_scores.append(best_resonance)
            
            # Update current token
            current_token = next_token
            current_idx = best_next_idx
        
        # Calculate overall coherence
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'path': path,
            'coherence': overall_coherence,
            'length': len(path),
            'resonance_scores': coherence_scores
        }
    
    def get_best_trace(self) -> Optional[Dict[str, Any]]:
        """Get the best trace by coherence"""
        if not self.traces:
            return None
        return self.traces[0]
    
    def get_coherent_traces(self, min_coherence: float = 0.5) -> List[Dict[str, Any]]:
        """Get traces above minimum coherence threshold"""
        return [trace for trace in self.traces if trace['coherence'] >= min_coherence]

class UniversalMolecularTokenLibrary:
    """Universal token library for molecular fragment classification"""
    
    def __init__(self):
        self.token_patterns = self._initialize_token_patterns()
        self.functional_groups = self._initialize_functional_groups()
        
    def _initialize_token_patterns(self) -> Dict[str, ChemicalTokenType]:
        """Initialize common molecular token patterns"""
        return {
            # Atoms
            'C': ChemicalTokenType.ATOM,
            'N': ChemicalTokenType.HETEROATOM,
            'O': ChemicalTokenType.HETEROATOM,
            'S': ChemicalTokenType.HETEROATOM,
            'P': ChemicalTokenType.HETEROATOM,
            'F': ChemicalTokenType.HETEROATOM,
            'Cl': ChemicalTokenType.HETEROATOM,
            'Br': ChemicalTokenType.HETEROATOM,
            'I': ChemicalTokenType.HETEROATOM,
            
            # Aromatic atoms
            'c': ChemicalTokenType.AROMATIC,
            'n': ChemicalTokenType.AROMATIC,
            'o': ChemicalTokenType.AROMATIC,
            's': ChemicalTokenType.AROMATIC,
            
            # Bonds
            '-': ChemicalTokenType.BOND,
            '=': ChemicalTokenType.BOND,
            '#': ChemicalTokenType.BOND,
            ':': ChemicalTokenType.BOND,
            
            # Stereochemistry
            '@': ChemicalTokenType.STEREOCHEMISTRY,
            '@@': ChemicalTokenType.STEREOCHEMISTRY,
            '/': ChemicalTokenType.STEREOCHEMISTRY,
            '\\': ChemicalTokenType.STEREOCHEMISTRY,
            
            # Charges
            '+': ChemicalTokenType.CHARGE,
            '-': ChemicalTokenType.CHARGE,
        }
    
    def _initialize_functional_groups(self) -> Dict[str, Tuple[ChemicalTokenType, ChemicalTransform]]:
        """Initialize functional group patterns"""
        return {
            'C=O': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.OXIDATION),
            'C(=O)O': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'C(=O)N': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'OH': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'NH2': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'SH': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'C#N': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.IDENTITY),
            'S(=O)(=O)': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.OXIDATION),
            'P(=O)': (ChemicalTokenType.FUNCTIONAL_GROUP, ChemicalTransform.OXIDATION),
        }
    
    def compress_smiles(self, smiles: str) -> Dict[str, Any]:
        """Compress SMILES into universal molecular tokens"""
        if not smiles:
            return {'tokens': [], 'compression_ratio': 0.0}
        
        tokens = []
        i = 0
        
        while i < len(smiles):
            # Try to match functional groups first (longer patterns)
            matched = False
            
            # Check functional groups (up to 10 characters)
            for length in range(min(10, len(smiles) - i), 0, -1):
                fragment = smiles[i:i+length]
                
                if fragment in self.functional_groups:
                    token_type, transform = self.functional_groups[fragment]
                    tokens.append({
                        'fragment': fragment,
                        'type': token_type.value,
                        'transform': transform.value,
                        'start': i,
                        'end': i + length
                    })
                    i += length
                    matched = True
                    break
            
            if matched:
                continue
            
            # Try single atom/bond patterns
            for length in range(min(3, len(smiles) - i), 0, -1):
                fragment = smiles[i:i+length]
                
                if fragment in self.token_patterns:
                    token_type = self.token_patterns[fragment]
                    tokens.append({
                        'fragment': fragment,
                        'type': token_type.value,
                        'transform': ChemicalTransform.IDENTITY.value,
                        'start': i,
                        'end': i + length
                    })
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Unknown token
                tokens.append({
                    'fragment': smiles[i],
                    'type': ChemicalTokenType.ATOM.value,
                    'transform': ChemicalTransform.IDENTITY.value,
                    'start': i,
                    'end': i + 1
                })
                i += 1
        
        # Calculate compression ratio
        original_length = len(smiles)
        compressed_length = len(tokens)
        compression_ratio = compressed_length / original_length if original_length > 0 else 0.0
        
        return {
            'tokens': tokens,
            'compression_ratio': compression_ratio,
            'original_smiles': smiles
        }

class HyperMolecularProcessor:
    """Main processor for hyperposition molecular analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'skip_threshold': 0.3,
            'max_trace_length': 10,
            'resonance_strength': 1.0
        }
        
        self.token_library = UniversalMolecularTokenLibrary()
        
        # Processing metrics
        self.metrics = {
            'molecules_processed': 0,
            'average_coherence': 0.0,
            'compression_ratio': 0.0,
            'processing_time': 0.0
        }
    
    def process_molecule(self, smiles: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process molecule through hyperposition analysis"""
        import time
        start_time = time.time()
        
        if context is None:
            context = {'ph': 7.0, 'temperature': 298.15, 'solvent_polarity': 0.5}
        
        # Step 1: Compress SMILES into universal tokens
        compression = self.token_library.compress_smiles(smiles)
        
        # Step 2: Create hyperposition tokens
        hyper_tokens = self._create_hyperposition_tokens(compression)
        
        # Step 3: Build resonance connections
        self._build_resonance_connections(hyper_tokens)
        
        # Step 4: Apply context to collapse states
        for token in hyper_tokens:
            token.collapse(context)
        
        # Step 5: Generate skip traces
        trace_engine = MolecularSkipTraceEngine(hyper_tokens, self.config['skip_threshold'])
        traces = trace_engine.generate_traces()
        
        # Step 6: Calculate molecular coherence
        molecular_coherence = self._calculate_molecular_coherence(hyper_tokens, traces)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(compression, traces, processing_time)
        
        return {
            'original_smiles': smiles,
            'compression': compression,
            'hyper_tokens': [token.to_dict() for token in hyper_tokens],
            'traces': traces,
            'best_trace': trace_engine.get_best_trace(),
            'coherent_traces': trace_engine.get_coherent_traces(),
            'molecular_coherence': molecular_coherence,
            'context': context,
            'metrics': self.metrics.copy()
        }
    
    def _create_hyperposition_tokens(self, compression: Dict[str, Any]) -> List[MolecularHyperToken]:
        """Create hyperposition tokens from compressed representation"""
        hyper_tokens = []
        
        for token_data in compression['tokens']:
            token_type = ChemicalTokenType(token_data['type'])
            transform = ChemicalTransform(token_data['transform'])
            
            hyper_token = MolecularHyperToken(
                token_data['fragment'],
                token_type,
                transform
            )
            
            # Normalize dimensions
            hyper_token.normalize_dimensions()
            
            hyper_tokens.append(hyper_token)
        
        return hyper_tokens
    
    def _build_resonance_connections(self, hyper_tokens: List[MolecularHyperToken]):
        """Build resonance connections between hyperposition tokens"""
        n_tokens = len(hyper_tokens)
        
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                resonance = hyper_tokens[i].resonance_with(hyper_tokens[j])
                
                # Apply resonance strength multiplier
                adjusted_resonance = resonance * self.config['resonance_strength']
                
                # Create bidirectional connections for strong resonance
                if adjusted_resonance > 0.4:
                    hyper_tokens[i].add_connection(hyper_tokens[j], adjusted_resonance)
                    hyper_tokens[j].add_connection(hyper_tokens[i], adjusted_resonance)
    
    def _calculate_molecular_coherence(self, 
                                     hyper_tokens: List[MolecularHyperToken], 
                                     traces: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall molecular coherence metrics"""
        if not hyper_tokens:
            return {'overall': 0.0, 'stability': 0.0, 'reactivity': 0.0}
        
        # Average dimensional coherence
        avg_dimensions = np.zeros(8)
        for token in hyper_tokens:
            avg_dimensions += token.dimensions.to_array()
        avg_dimensions /= len(hyper_tokens)
        
        # Trace coherence
        trace_coherences = [trace['coherence'] for trace in traces]
        avg_trace_coherence = np.mean(trace_coherences) if trace_coherences else 0.0
        
        # Overall molecular coherence
        stability_coherence = avg_dimensions[3]  # Stability dimension
        reactivity_coherence = 1.0 - avg_dimensions[2]  # Inverse reactivity
        
        overall_coherence = (stability_coherence + reactivity_coherence + avg_trace_coherence) / 3.0
        
        return {
            'overall': overall_coherence,
            'stability': stability_coherence,
            'reactivity': avg_dimensions[2],
            'trace_coherence': avg_trace_coherence,
            'dimensional_balance': np.std(avg_dimensions)
        }
    
    def _update_metrics(self, compression: Dict[str, Any], traces: List[Dict[str, Any]], processing_time: float):
        """Update processing metrics"""
        self.metrics['molecules_processed'] += 1
        self.metrics['compression_ratio'] = (
            (self.metrics['compression_ratio'] * (self.metrics['molecules_processed'] - 1) + 
             compression['compression_ratio']) / self.metrics['molecules_processed']
        )
        self.metrics['processing_time'] = processing_time
        
        if traces:
            trace_coherences = [trace['coherence'] for trace in traces]
            avg_coherence = np.mean(trace_coherences)
            self.metrics['average_coherence'] = (
                (self.metrics['average_coherence'] * (self.metrics['molecules_processed'] - 1) + 
                 avg_coherence) / self.metrics['molecules_processed']
            )
    
    def compare_molecules(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """Compare two molecules using hyperposition analysis"""
        result1 = self.process_molecule(smiles1)
        result2 = self.process_molecule(smiles2)
        
        # Compare compression patterns
        tokens1 = [token['type'] for token in result1['compression']['tokens']]
        tokens2 = [token['type'] for token in result2['compression']['tokens']]
        
        pattern_similarity = self._calculate_pattern_similarity(tokens1, tokens2)
        
        # Compare best traces
        trace_similarity = 0.0
        if result1['best_trace'] and result2['best_trace']:
            trace_similarity = self._calculate_trace_similarity(
                result1['best_trace'], result2['best_trace']
            )
        
        # Compare molecular coherence
        coherence1 = result1['molecular_coherence']['overall']
        coherence2 = result2['molecular_coherence']['overall']
        coherence_similarity = 1.0 - abs(coherence1 - coherence2)
        
        return {
            'smiles1': smiles1,
            'smiles2': smiles2,
            'pattern_similarity': pattern_similarity,
            'trace_similarity': trace_similarity,
            'coherence_similarity': coherence_similarity,
            'overall_similarity': (pattern_similarity + trace_similarity + coherence_similarity) / 3.0,
            'analysis1': result1,
            'analysis2': result2
        }
    
    def _calculate_pattern_similarity(self, pattern1: List[str], pattern2: List[str]) -> float:
        """Calculate similarity between token patterns"""
        if not pattern1 or not pattern2:
            return 0.0
        
        # Use Jaccard similarity
        set1 = set(pattern1)
        set2 = set(pattern2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_trace_similarity(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> float:
        """Calculate similarity between molecular traces"""
        # Compare trace lengths and coherences
        length_sim = 1.0 - abs(trace1['length'] - trace2['length']) / max(trace1['length'], trace2['length'])
        coherence_sim = 1.0 - abs(trace1['coherence'] - trace2['coherence'])
        
        return (length_sim + coherence_sim) / 2.0
    
    def visualize_analysis(self, result: Dict[str, Any]) -> str:
        """Create visualization of hyperposition analysis"""
        output = []
        
        output.append('\n=== Hyperposition Molecular Analysis ===\n')
        output.append(f"SMILES: {result['original_smiles']}")
        output.append(f"Compression: {len(result['compression']['tokens'])} tokens "
                     f"({result['compression']['compression_ratio']:.3f} ratio)")
        
        # Show hyperposition tokens
        output.append('\nHyperposition Tokens:')
        for i, token in enumerate(result['hyper_tokens']):
            dims = token['dimensions']
            output.append(f"  {i+1}. [{token['smiles_fragment']}] "
                         f"({token['token_type']}) "
                         f"E:{dims['electronic']:.2f} S:{dims['stability']:.2f} "
                         f"R:{dims['reactivity']:.2f}")
        
        # Show best trace
        if result['best_trace']:
            trace = result['best_trace']
            path = [token.smiles_fragment for token in trace['path']]
            output.append(f'\nBest Molecular Trace:')
            output.append(f"  Path: {' → '.join(path)}")
            output.append(f"  Coherence: {trace['coherence']:.3f}")
        
        # Show molecular coherence
        coherence = result['molecular_coherence']
        output.append(f'\nMolecular Coherence:')
        output.append(f"  Overall: {coherence['overall']:.3f}")
        output.append(f"  Stability: {coherence['stability']:.3f}")
        output.append(f"  Reactivity: {coherence['reactivity']:.3f}")
        
        return '\n'.join(output)

def create_molecular_hyperprocessor(config: Optional[Dict[str, Any]] = None) -> HyperMolecularProcessor:
    """Factory function to create molecular hyperposition processor"""
    return HyperMolecularProcessor(config)
