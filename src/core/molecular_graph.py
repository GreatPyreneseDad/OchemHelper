"""Molecular graph representation using PyTorch Geometric."""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import Optional, Tuple, List
import numpy as np


class MolecularGraph:
    """Convert molecules to graph representations for neural networks."""
    
    def __init__(self, mol: Optional[Chem.Mol] = None):
        """Initialize with an RDKit molecule object."""
        self.mol = mol
        self._graph = None
        
    @classmethod
    def from_smiles(cls, smiles: str) -> "MolecularGraph":
        """Create MolecularGraph from SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return cls(mol)
    
    @property
    def num_atoms(self) -> int:
        """Get number of atoms in the molecule."""
        return self.mol.GetNumAtoms() if self.mol else 0
    
    @property
    def num_bonds(self) -> int:
        """Get number of bonds in the molecule."""
        return self.mol.GetNumBonds() if self.mol else 0
    
    def to_graph(self) -> Data:
        """Convert molecule to PyTorch Geometric Data object."""
        if self._graph is not None:
            return self._graph
            
        if self.mol is None:
            raise ValueError("No molecule to convert")
        
        # Get atom features
        atom_features = []
        for atom in self.mol.GetAtoms():
            features = self._get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get bond indices
        edge_indices = []
        edge_attrs = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Undirected graph
            
            bond_features = self._get_bond_features(bond)
            edge_attrs.extend([bond_features, bond_features])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        self._graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self._graph
    
    def _get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """Extract features from an atom."""
        features = []
        features.append(atom.GetAtomicNum())
        features.append(atom.GetDegree())
        features.append(atom.GetFormalCharge())
        features.append(int(atom.GetHybridization()))
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetMass())
        return features
    
    def _get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """Extract features from a bond."""
        features = []
        features.append(int(bond.GetBondType()))
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        features.append(int(bond.GetStereo()))
        return features