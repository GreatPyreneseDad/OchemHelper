"""Structure converter for SMILES to 3D coordinates."""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StructureConverter:
    """Convert SMILES to 3D structures and other formats."""
    
    def __init__(self):
        self.embed_params = AllChem.ETKDGv3()
        self.embed_params.randomSeed = 42
    
    def smiles_to_3d(self, smiles: str, format: str = "sdf") -> Optional[str]:
        """Convert SMILES to 3D structure in specified format.
        
        Args:
            smiles: SMILES string
            format: Output format ('sdf', 'pdb', 'mol2', 'xyz')
            
        Returns:
            3D structure in specified format or None if conversion fails
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, self.embed_params)
            if result != 0:
                logger.error(f"Failed to embed molecule: {smiles}")
                return None
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            
            # Convert to requested format
            if format == "sdf":
                return Chem.MolToMolBlock(mol)
            elif format == "pdb":
                return Chem.MolToPDBBlock(mol)
            elif format == "mol2":
                return self._mol_to_mol2(mol)
            elif format == "xyz":
                return self._mol_to_xyz(mol)
            else:
                logger.error(f"Unsupported format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting SMILES to 3D: {e}")
            return None
    
    def _mol_to_xyz(self, mol) -> str:
        """Convert RDKit mol to XYZ format."""
        conf = mol.GetConformer()
        xyz = f"{mol.GetNumAtoms()}\n"
        xyz += f"Generated from SMILES\n"
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            xyz += f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
        
        return xyz
    
    def _mol_to_mol2(self, mol) -> str:
        """Convert RDKit mol to MOL2 format (simplified)."""
        # This is a simplified MOL2 writer
        conf = mol.GetConformer()
        mol2 = "@<TRIPOS>MOLECULE\n"
        mol2 += f"Generated molecule\n"
        mol2 += f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0\n"
        mol2 += "SMALL\nGASTEIGER\n\n"
        
        # Atoms
        mol2 += "@<TRIPOS>ATOM\n"
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            mol2 += f"{i+1} {atom.GetSymbol()}{i+1} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} {atom.GetSymbol()} 1 MOL 0.0000\n"
        
        # Bonds
        mol2 += "@<TRIPOS>BOND\n"
        for i, bond in enumerate(mol.GetBonds()):
            bond_type = "1" if bond.GetBondType() == Chem.BondType.SINGLE else "2" if bond.GetBondType() == Chem.BondType.DOUBLE else "3"
            mol2 += f"{i+1} {bond.GetBeginAtomIdx()+1} {bond.GetEndAtomIdx()+1} {bond_type}\n"
        
        return mol2
    
    def get_3d_info(self, smiles: str) -> Dict:
        """Get 3D structure information for a molecule.
        
        Returns:
            Dict with structure info including coordinates, bonds, etc.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            # Add hydrogens and generate 3D
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, self.embed_params)
            if result != 0:
                return {"error": "Failed to generate 3D structure"}
            
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()
            
            # Extract atom information
            atoms = []
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                atoms.append({
                    "element": atom.GetSymbol(),
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "z": float(pos.z),
                    "charge": atom.GetFormalCharge()
                })
            
            # Extract bond information
            bonds = []
            for bond in mol.GetBonds():
                bonds.append({
                    "start": bond.GetBeginAtomIdx(),
                    "end": bond.GetEndAtomIdx(),
                    "order": int(bond.GetBondTypeAsDouble())
                })
            
            return {
                "atoms": atoms,
                "bonds": bonds,
                "smiles": smiles,
                "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": Descriptors.MolWt(mol)
            }
            
        except Exception as e:
            logger.error(f"Error getting 3D info: {e}")
            return {"error": str(e)}


# Singleton instance
converter = StructureConverter()