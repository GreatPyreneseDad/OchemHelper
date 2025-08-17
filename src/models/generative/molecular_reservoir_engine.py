"""
Molecular Reservoir Computing Engine - Advanced Implementation
Integrating Basal Reservoir Computing with Enhanced Chemical Coherence Dynamics

Adapted from TraderAI's Basal Reservoir Engine for molecular discovery
and chemical reaction prediction using Physarum-inspired computing.

This module implements slime mold-inspired reservoir computing with chemical
reasoning capabilities for enhanced molecular design and property prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import softmax
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import json
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

logger = logging.getLogger(__name__)

@dataclass
class ChemicalCoherenceDimensions:
    """Chemical Coherence Theory dimensions adapted from GCT"""
    stability: float      # Thermodynamic stability (0.0 - 1.0)
    reactivity: float     # Chemical reactivity potential (0.0 - 1.0) 
    selectivity: float    # Reaction selectivity (0.0 - 1.0)
    accessibility: float  # Synthetic accessibility (0.0 - 1.0)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'stability': self.stability,
            'reactivity': self.reactivity,
            'selectivity': self.selectivity,
            'accessibility': self.accessibility
        }
    
    def coherence_magnitude(self) -> float:
        """Calculate overall chemical coherence magnitude"""
        return np.sqrt(self.stability**2 + self.reactivity**2 + 
                      self.selectivity**2 + self.accessibility**2) / 2.0

@dataclass
class MolecularReservoirConfig:
    """Configuration for the Molecular Reservoir Computing system"""
    num_nodes: int = 150
    spatial_dimension: int = 3  # 3D molecular space
    learning_rate: float = 0.01
    energy_decay: float = 0.95
    connection_radius: float = 0.35
    homeodynamic_strength: float = 0.1
    coherence_coupling: float = 0.08
    adaptation_rate: float = 0.001
    prediction_horizon: int = 10
    chemical_temperature: float = 298.15  # Room temperature in K

class MolecularReservoirNode:
    """
    Individual reservoir node inspired by Physarum behavior for chemistry
    Implements chemical homeodynamic regulation and molecular energetics
    """
    
    def __init__(self, node_id: int, position: np.ndarray, config: MolecularReservoirConfig):
        self.node_id = node_id
        self.position = position
        self.config = config
        
        # Core chemical state variables
        self.energy = np.random.uniform(0.3, 0.7)
        self.target_energy = np.random.uniform(0.4, 0.6)
        self.activation = 0.0
        self.chemical_potential = np.random.uniform(-0.5, 0.5)
        
        # Chemical-specific properties
        self.electron_density = np.random.uniform(0.0, 1.0)
        self.bond_order = np.random.uniform(0.5, 3.0)
        self.electrophilicity = np.random.uniform(0.0, 1.0)
        self.nucleophilicity = np.random.uniform(0.0, 1.0)
        
        # Connection weights (populated by reservoir)
        self.incoming_weights: Dict[int, float] = {}
        self.outgoing_weights: Dict[int, float] = {}
        
        # Temporal memory for reaction pattern encoding
        self.activation_history = []
        self.energy_history = []
        self.chemical_history = []
        
    def update_chemical_state(self, 
                             molecular_inputs: Dict[int, float], 
                             neighbor_activations: Dict[int, float],
                             chemical_environment: Dict[str, float]) -> float:
        """
        Update node chemical state based on molecular inputs and environment
        Implements enhanced chemical dynamics:
        Xn(t) = σ(Σ Wn,m * Mm(t) + Σ λ * Wn,n' * Xn'(t-1) + Φ(env))
        """
        # Molecular input contribution
        molecular_sum = sum(self.incoming_weights.get(source_id, 0) * signal 
                           for source_id, signal in molecular_inputs.items())
        
        # Neighbor contribution with chemical coupling
        neighbor_sum = sum(self.incoming_weights.get(neighbor_id, 0) * activation 
                          for neighbor_id, activation in neighbor_activations.items())
        
        # Environmental contribution (pH, temperature, solvent effects)
        env_contribution = self._compute_environmental_effect(chemical_environment)
        
        # Apply chemical dynamics with homeodynamic regulation
        chemical_input = molecular_sum + 0.8 * neighbor_sum + 0.3 * env_contribution
        self.activation = np.tanh(chemical_input)
        
        # Update chemical properties based on activation
        self._update_chemical_properties()
        
        # Energy dynamics with chemical potential
        energy_error = self.energy - self.target_energy
        self.energy = (self.energy * self.config.energy_decay + 
                      0.1 * self.activation - 
                      0.05 * energy_error +
                      0.02 * self.chemical_potential)
        
        # Clamp energy and chemical potential
        self.energy = np.clip(self.energy, 0.0, 1.0)
        self.chemical_potential = np.clip(self.chemical_potential, -1.0, 1.0)
        
        # Store history for pattern encoding
        self.activation_history.append(self.activation)
        self.energy_history.append(self.energy)
        self.chemical_history.append({
            'electron_density': self.electron_density,
            'electrophilicity': self.electrophilicity,
            'nucleophilicity': self.nucleophilicity
        })
        
        # Limit history size
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            self.energy_history.pop(0)
            self.chemical_history.pop(0)
            
        return self.activation
    
    def _compute_environmental_effect(self, environment: Dict[str, float]) -> float:
        """Compute effect of chemical environment on node state"""
        ph = environment.get('ph', 7.0)
        temperature = environment.get('temperature', 298.15)
        ionic_strength = environment.get('ionic_strength', 0.0)
        
        # pH effect on protonation state
        ph_effect = np.tanh((ph - 7.0) / 3.0) * 0.3
        
        # Temperature effect on kinetics (Arrhenius-like)
        temp_factor = temperature / self.config.chemical_temperature
        temp_effect = np.log(temp_factor) * 0.2
        
        # Ionic strength effect on electrostatics
        ionic_effect = ionic_strength * 0.1
        
        return ph_effect + temp_effect + ionic_effect
    
    def _update_chemical_properties(self):
        """Update chemical properties based on current activation"""
        # Electron density follows activation
        self.electron_density = 0.7 * self.electron_density + 0.3 * abs(self.activation)
        
        # Electrophilicity vs nucleophilicity balance
        if self.activation > 0:
            self.electrophilicity = min(1.0, self.electrophilicity + 0.1 * self.activation)
            self.nucleophilicity = max(0.0, self.nucleophilicity - 0.05 * self.activation)
        else:
            self.nucleophilicity = min(1.0, self.nucleophilicity - 0.1 * self.activation)
            self.electrophilicity = max(0.0, self.electrophilicity + 0.05 * self.activation)
    
    def chemical_homeodynamic_learning(self, neighbor_states: Dict[int, float]):
        """
        Chemical homeodynamic learning rule with reaction selectivity
        Wn,n'(t+1) = Wn,n'(t) - ηW * (Xn(t) - Tn(t)) * Xn'(t) * S(n,n')
        where S(n,n') is chemical selectivity factor
        """
        if not neighbor_states:
            return
            
        # Calculate energy error
        energy_error = self.energy - self.target_energy
        
        # Calculate normalization factor with chemical selectivity
        total_weighted_input = sum(
            neighbor_states.get(nid, 0) * self.incoming_weights.get(nid, 0) * 
            self._chemical_selectivity(nid)
            for nid in neighbor_states.keys()
        )
        
        if abs(total_weighted_input) < 1e-6:
            return
        
        # Update connection weights using chemical homeodynamic rule
        for neighbor_id, neighbor_activation in neighbor_states.items():
            if neighbor_id in self.incoming_weights:
                current_weight = self.incoming_weights[neighbor_id]
                selectivity = self._chemical_selectivity(neighbor_id)
                
                # Chemical homeodynamic weight update
                weight_delta = (self.config.learning_rate * energy_error * 
                               neighbor_activation * current_weight * selectivity / 
                               total_weighted_input)
                
                new_weight = current_weight - weight_delta
                self.incoming_weights[neighbor_id] = np.clip(new_weight, -2.0, 2.0)
    
    def _chemical_selectivity(self, neighbor_id: int) -> float:
        """Calculate chemical selectivity between nodes"""
        # Simple distance-based selectivity for now
        # In advanced version, this would use chemical similarity
        return np.exp(-0.1 * abs(self.node_id - neighbor_id))
    
    def adapt_chemical_target(self):
        """Adapt target energy based on chemical environment stability"""
        if len(self.energy_history) >= 10:
            recent_energy = np.array(self.energy_history[-10:])
            energy_variance = np.var(recent_energy)
            
            # Stable chemical environments lower target, unstable raise it
            if energy_variance < 0.01:  # Very stable
                self.target_energy = max(0.2, self.target_energy - self.config.adaptation_rate)
            elif energy_variance > 0.1:  # Very unstable  
                self.target_energy = min(0.8, self.target_energy + self.config.adaptation_rate)

class MolecularReservoirEngine:
    """
    Main engine implementing Molecular Reservoir Computing with Chemical Coherence
    """
    
    def __init__(self, config: MolecularReservoirConfig):
        self.config = config
        self.nodes: List[MolecularReservoirNode] = []
        self.adjacency_matrix = None
        self.coherence_state = ChemicalCoherenceDimensions(
            stability=0.5, reactivity=0.5, selectivity=0.5, accessibility=0.5
        )
        
        # Chemical reservoir dynamics parameters
        self.gamma = 0.12   # Chemical stability regulatory constant
        self.delta = 0.18   # Reaction frequency coupling
        self.epsilon = 0.06 # Kinetic energy damping
        self.phi = config.coherence_coupling  # Chemical-Reservoir integration
        
        # Chemical environment state
        self.environment = {
            'ph': 7.0,
            'temperature': 298.15,
            'ionic_strength': 0.0,
            'solvent_polarity': 0.5
        }
        
        # Temporal state tracking
        self.coherence_history = []
        self.reactivity_history = []
        self.stability_evolution_history = []
        
        # Performance metrics
        self.prediction_accuracy = 0.0
        self.synthesis_efficiency = 0.0
        
        self._initialize_molecular_reservoir()
        
    def _initialize_molecular_reservoir(self):
        """Initialize 3D molecular reservoir network with chemical topology"""
        logger.info(f"Initializing molecular reservoir with {self.config.num_nodes} nodes")
        
        # Create nodes with 3D molecular-like distribution
        positions = self._generate_molecular_positions()
        
        for i, pos in enumerate(positions):
            node = MolecularReservoirNode(i, pos, self.config)
            self.nodes.append(node)
        
        # Build chemical connectivity and initialize weights
        self._build_chemical_connectivity()
        self._initialize_chemical_weights()
        
        logger.info("Molecular reservoir initialization complete")
    
    def _generate_molecular_positions(self) -> np.ndarray:
        """Generate 3D positions that mimic molecular arrangements"""
        positions = []
        
        # Core molecular structure (like a protein backbone)
        backbone_length = int(self.config.num_nodes * 0.4)
        for i in range(backbone_length):
            t = i / backbone_length * 2 * np.pi
            pos = np.array([
                np.cos(t) * 0.5,
                np.sin(t) * 0.5,
                i / backbone_length
            ])
            positions.append(pos)
        
        # Side chains and functional groups
        for i in range(self.config.num_nodes - backbone_length):
            # Random positions around backbone
            backbone_idx = np.random.randint(0, backbone_length)
            backbone_pos = positions[backbone_idx]
            
            # Offset from backbone
            offset = np.random.normal(0, 0.3, 3)
            pos = backbone_pos + offset
            positions.append(pos)
        
        return np.array(positions)
    
    def _build_chemical_connectivity(self):
        """Build chemical connectivity matrix based on bonding rules"""
        n_nodes = len(self.nodes)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    distance = np.linalg.norm(node_i.position - node_j.position)
                    
                    # Chemical bonding rules (simplified)
                    if distance < self.config.connection_radius:
                        # Bond strength based on chemical compatibility
                        bond_strength = self._calculate_bond_strength(node_i, node_j, distance)
                        self.adjacency_matrix[i, j] = bond_strength
    
    def _calculate_bond_strength(self, node_i: MolecularReservoirNode, 
                                node_j: MolecularReservoirNode, distance: float) -> float:
        """Calculate chemical bond strength between nodes"""
        # Distance-based component
        distance_factor = np.exp(-distance / (self.config.connection_radius / 2))
        
        # Chemical compatibility (electrophile-nucleophile attraction)
        compatibility = (node_i.electrophilicity * node_j.nucleophilicity + 
                        node_i.nucleophilicity * node_j.electrophilicity) / 2
        
        # Electron density overlap
        density_overlap = min(node_i.electron_density, node_j.electron_density)
        
        return distance_factor * compatibility * density_overlap
    
    def _initialize_chemical_weights(self):
        """Initialize connection weights based on chemical bonding"""
        for i, node in enumerate(self.nodes):
            # Initialize incoming connections
            for j, bond_strength in enumerate(self.adjacency_matrix[:, i]):
                if bond_strength > 0:
                    weight = np.random.normal(0, 0.2) * bond_strength
                    node.incoming_weights[j] = weight
            
            # Initialize outgoing connections
            for j, bond_strength in enumerate(self.adjacency_matrix[i, :]):
                if bond_strength > 0:
                    weight = np.random.normal(0, 0.2) * bond_strength
                    node.outgoing_weights[j] = weight
    
    def update_reservoir_state(self, 
                              molecular_inputs: Optional[Dict[int, float]] = None,
                              environment_update: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Update entire molecular reservoir state for one time step
        Returns current activation pattern
        """
        if molecular_inputs is None:
            molecular_inputs = {}
        
        if environment_update:
            self.environment.update(environment_update)
        
        # Get current neighbor activations for each node
        current_activations = {node.node_id: node.activation for node in self.nodes}
        
        # Update all nodes
        new_activations = []
        for node in self.nodes:
            # Get neighbor states (excluding self)
            neighbor_states = {nid: act for nid, act in current_activations.items() 
                             if nid != node.node_id and nid in node.incoming_weights}
            
            # Update node chemical state
            activation = node.update_chemical_state(
                molecular_inputs, neighbor_states, self.environment
            )
            new_activations.append(activation)
            
            # Apply chemical homeodynamic learning
            node.chemical_homeodynamic_learning(neighbor_states)
            
            # Adapt chemical target
            node.adapt_chemical_target()
        
        return np.array(new_activations)
    
    def compute_chemical_coherence(self) -> float:
        """
        Compute enhanced chemical coherence with reservoir integration
        dΨ(t)/dt = -γS(t) + δR(t) - εK(t) + φΣ(Xn(t) - Tn(t))
        """
        # Get current reservoir energy deviations
        energy_deviations = [node.energy - node.target_energy for node in self.nodes]
        reservoir_contribution = self.phi * np.sum(energy_deviations)
        
        # Enhanced chemical activities
        S_t = self._compute_stability_factor()     # Thermodynamic stability
        R_t = self._compute_reactivity_factor()    # Reaction frequency  
        K_t = self._compute_kinetic_activity()     # Kinetic energy
        
        # Chemical coherence evolution equation
        coherence_derivative = (-self.gamma * S_t + 
                               self.delta * R_t - 
                               self.epsilon * K_t + 
                               reservoir_contribution)
        
        # Update coherence using Euler integration
        new_stability = self.coherence_state.stability + 0.01 * coherence_derivative
        self.coherence_state.stability = np.clip(new_stability, 0.0, 1.0)
        
        self.coherence_history.append(self.coherence_state.stability)
        return self.coherence_state.stability
    
    def compute_reaction_anticipation(self) -> float:
        """
        Compute reaction anticipation capacity
        Anticipation(t) ≈ φΣ(En(t) - μn(t))
        where En is energy and μn is chemical potential
        """
        energy_potential_deviations = [
            node.energy - node.chemical_potential for node in self.nodes
        ]
        anticipation = self.phi * np.sum(energy_potential_deviations)
        
        self.reactivity_history.append(anticipation)
        return anticipation
    
    def _compute_stability_factor(self) -> float:
        """Compute chemical stability from reservoir consensus"""
        activations = [node.activation for node in self.nodes]
        # High consensus (low variance) increases stability
        variance = np.var(activations)
        stability = np.exp(-3 * variance)  # Lower variance = higher stability
        return stability
    
    def _compute_reactivity_factor(self) -> float:
        """Compute reaction frequency from chemical oscillations"""
        if len(self.coherence_history) < 5:
            return self.coherence_state.reactivity
        
        # Analyze frequency content of recent coherence
        recent_coherence = np.array(self.coherence_history[-20:])
        fft_power = np.abs(np.fft.fft(recent_coherence))
        dominant_frequency = np.argmax(fft_power[1:len(fft_power)//2]) + 1
        normalized_freq = dominant_frequency / len(recent_coherence)
        
        return min(1.0, normalized_freq * 6)  # Scale and clamp
    
    def _compute_kinetic_activity(self) -> float:
        """Compute kinetic energy from reservoir dynamics"""
        # Kinetic activity based on total energy change rate
        if len(self.nodes) == 0:
            return 0.0
            
        energy_changes = []
        for node in self.nodes:
            if len(node.energy_history) >= 2:
                energy_change = abs(node.energy_history[-1] - node.energy_history[-2])
                energy_changes.append(energy_change)
        
        if energy_changes:
            return np.mean(energy_changes)
        return 0.0
    
    def predict_molecular_properties(self, 
                                   smiles: str, 
                                   properties: List[str] = None) -> Dict[str, float]:
        """
        Predict molecular properties using reservoir dynamics
        """
        if properties is None:
            properties = ['logP', 'MW', 'TPSA', 'QED', 'SA_score']
        
        # Encode SMILES into reservoir inputs
        molecular_inputs = self._encode_smiles_to_reservoir(smiles)
        
        # Run reservoir forward for property prediction
        predictions = {}
        
        # Update reservoir with molecular inputs
        activations = self.update_reservoir_state(molecular_inputs)
        
        # Compute chemical coherence
        coherence = self.compute_chemical_coherence()
        anticipation = self.compute_reaction_anticipation()
        
        # Predict each property
        for prop in properties:
            prediction = self._decode_property_prediction(
                activations, coherence, anticipation, prop
            )
            predictions[prop] = prediction
        
        return predictions
    
    def predict_synthetic_route(self, target_smiles: str, max_steps: int = 5) -> List[str]:
        """
        Predict synthetic route using reservoir anticipation
        """
        route = []
        current_smiles = target_smiles
        
        for step in range(max_steps):
            # Encode current molecule
            molecular_inputs = self._encode_smiles_to_reservoir(current_smiles)
            
            # Update reservoir
            activations = self.update_reservoir_state(molecular_inputs)
            anticipation = self.compute_reaction_anticipation()
            
            # Predict previous step (retrosynthesis)
            precursor = self._decode_retrosynthetic_step(
                activations, anticipation, current_smiles
            )
            
            if precursor and precursor != current_smiles:
                route.append(precursor)
                current_smiles = precursor
            else:
                break
        
        return route[::-1]  # Reverse for forward synthesis
    
    def _encode_smiles_to_reservoir(self, smiles: str) -> Dict[int, float]:
        """Encode SMILES string into reservoir input signals"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            # Calculate molecular descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            num_atoms = mol.GetNumAtoms()
            
            # Normalize descriptors
            normalized_mw = min(1.0, mw / 500.0)
            normalized_logp = (logp + 5) / 10.0  # Range [-5, 5] -> [0, 1]
            normalized_tpsa = min(1.0, tpsa / 200.0)
            normalized_atoms = min(1.0, num_atoms / 50.0)
            
            # Distribute inputs across reservoir nodes
            inputs = {}
            num_input_nodes = min(len(self.nodes) // 4, 4)
            
            input_values = [normalized_mw, normalized_logp, normalized_tpsa, normalized_atoms]
            
            for i, value in enumerate(input_values):
                if i < num_input_nodes:
                    node_id = i * (len(self.nodes) // num_input_nodes)
                    inputs[node_id] = float(value)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error encoding SMILES {smiles}: {e}")
            return {}
    
    def _decode_property_prediction(self, 
                                   activations: np.ndarray, 
                                   coherence: float, 
                                   anticipation: float, 
                                   property_name: str) -> float:
        """Decode reservoir state into property prediction"""
        
        # Weighted combination of reservoir outputs
        weighted_activation = np.average(activations, weights=np.abs(activations) + 0.1)
        
        # Property-specific decoding
        if property_name == 'logP':
            # LogP prediction (range -5 to 5)
            prediction = (0.7 * weighted_activation + 0.2 * coherence + 0.1 * anticipation)
            return (prediction - 0.5) * 10  # Scale to [-5, 5]
        
        elif property_name == 'MW':
            # Molecular weight prediction
            prediction = (0.8 * weighted_activation + 0.2 * coherence)
            return prediction * 800  # Scale to reasonable MW range
        
        elif property_name == 'TPSA':
            # Topological polar surface area
            prediction = (0.6 * weighted_activation + 0.4 * coherence)
            return prediction * 200  # Scale to TPSA range
        
        elif property_name == 'QED':
            # Drug-likeness (0-1 range)
            prediction = (0.5 * weighted_activation + 0.3 * coherence + 0.2 * anticipation)
            return np.clip(prediction, 0.0, 1.0)
        
        elif property_name == 'SA_score':
            # Synthetic accessibility (1-10, lower is better)
            prediction = (0.4 * weighted_activation + 0.6 * (1 - coherence))
            return 1 + prediction * 9  # Scale to [1, 10]
        
        else:
            # Generic property prediction
            return (0.6 * weighted_activation + 0.3 * coherence + 0.1 * anticipation)
    
    def _decode_retrosynthetic_step(self, 
                                   activations: np.ndarray, 
                                   anticipation: float, 
                                   current_smiles: str) -> Optional[str]:
        """Decode reservoir state into retrosynthetic precursor"""
        # This is a simplified placeholder - real implementation would use
        # reaction templates and chemical knowledge
        
        try:
            mol = Chem.MolFromSmiles(current_smiles)
            if mol is None:
                return None
            
            # Use anticipation to guide retrosynthetic analysis
            if anticipation > 0.5:
                # High anticipation suggests complex disconnection
                # Try breaking largest ring or longest chain
                pass
            else:
                # Low anticipation suggests simple functional group changes
                # Try deprotection or simple transformations
                pass
            
            # Placeholder: return a modified version
            return current_smiles  # Would return actual precursor
            
        except Exception as e:
            logger.error(f"Error in retrosynthetic step: {e}")
            return None
    
    def get_reservoir_state(self) -> Dict[str, any]:
        """Get current molecular reservoir state for monitoring"""
        return {
            'num_nodes': len(self.nodes),
            'average_energy': np.mean([node.energy for node in self.nodes]),
            'energy_variance': np.var([node.energy for node in self.nodes]),
            'average_activation': np.mean([node.activation for node in self.nodes]),
            'chemical_coherence': self.coherence_state.stability,
            'reaction_anticipation': self.reactivity_history[-1] if self.reactivity_history else 0.0,
            'connection_density': np.mean([len(node.incoming_weights) for node in self.nodes]),
            'environment': self.environment.copy()
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'synthesis_efficiency': self.synthesis_efficiency,
            'coherence_stability': np.std(self.coherence_history[-20:]) if len(self.coherence_history) >= 20 else 1.0,
            'reactivity_range': np.max(self.reactivity_history) - np.min(self.reactivity_history) if self.reactivity_history else 0.0
        }
    
    def save_state(self, filepath: str):
        """Save molecular reservoir state to file"""
        state = {
            'config': {
                'num_nodes': self.config.num_nodes,
                'learning_rate': self.config.learning_rate,
                'energy_decay': self.config.energy_decay,
                'coherence_coupling': self.config.coherence_coupling,
                'chemical_temperature': self.config.chemical_temperature
            },
            'coherence_state': self.coherence_state.to_dict(),
            'environment': self.environment,
            'node_states': [
                {
                    'node_id': node.node_id,
                    'position': node.position.tolist(),
                    'energy': node.energy,
                    'target_energy': node.target_energy,
                    'activation': node.activation,
                    'chemical_potential': node.chemical_potential,
                    'electron_density': node.electron_density,
                    'electrophilicity': node.electrophilicity,
                    'nucleophilicity': node.nucleophilicity,
                    'incoming_weights': node.incoming_weights
                }
                for node in self.nodes
            ],
            'history': {
                'coherence': self.coherence_history[-100:],
                'reactivity': self.reactivity_history[-100:]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Molecular Reservoir state saved to {filepath}")

def create_molecular_reservoir_engine(config: Optional[MolecularReservoirConfig] = None) -> MolecularReservoirEngine:
    """Factory function to create configured Molecular Reservoir engine"""
    if config is None:
        config = MolecularReservoirConfig()
    
    return MolecularReservoirEngine(config)
