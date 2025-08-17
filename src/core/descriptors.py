"""Molecular descriptor calculations."""

from typing import Dict, List, Optional, Union, Tuple
from functools import lru_cache
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)


class MolecularDescriptors:
    """Calculate molecular descriptors with caching."""
    
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        self._sa_score_cache = {}
        self._load_sa_score_data()
        
    @lru_cache(maxsize=128)
    def calculate_all(self, smiles: str) -> Dict[str, float]:
        """Calculate all molecular descriptors.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary of descriptor names and values
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
            
        descriptors = {}
        
        # Basic descriptors
        descriptors["MW"] = round(Descriptors.MolWt(mol), 2)
        descriptors["logP"] = round(Crippen.MolLogP(mol), 2)
        descriptors["logD"] = round(Crippen.MolLogP(mol), 2)  # Simplified, same as logP
        descriptors["TPSA"] = round(Descriptors.TPSA(mol), 2)
        descriptors["HBA"] = Lipinski.NumHAcceptors(mol)
        descriptors["HBD"] = Lipinski.NumHDonors(mol)
        descriptors["rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
        descriptors["num_rings"] = Lipinski.RingCount(mol)
        descriptors["num_aromatic_rings"] = Lipinski.NumAromaticRings(mol)
        descriptors["num_heteroatoms"] = Lipinski.NumHeteroatoms(mol)
        descriptors["num_heavy_atoms"] = Lipinski.HeavyAtomCount(mol)
        descriptors["num_carbons"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 6])
        descriptors["num_nitrogens"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7])
        descriptors["num_oxygens"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8])
        descriptors["num_sulfurs"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 16])
        
        # Complexity descriptors
        descriptors["fraction_sp3"] = round(Lipinski.FractionCsp3(mol), 3)
        descriptors["num_stereocenters"] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        descriptors["num_unspecified_stereo"] = self._count_unspecified_stereocenters(mol)
        
        # Additional descriptors
        descriptors["MR"] = round(Crippen.MolMR(mol), 2)  # Molar refractivity
        descriptors["formal_charge"] = Chem.rdmolops.GetFormalCharge(mol)
        descriptors["num_radical_electrons"] = Descriptors.NumRadicalElectrons(mol)
        descriptors["num_valence_electrons"] = Descriptors.NumValenceElectrons(mol)
        
        # Topological descriptors
        descriptors["balaban_j"] = round(Descriptors.BalabanJ(mol), 3)
        descriptors["bertz_complexity"] = round(Descriptors.BertzCT(mol), 2)
        descriptors["kappa1"] = round(Descriptors.Kappa1(mol), 3)
        descriptors["kappa2"] = round(Descriptors.Kappa2(mol), 3)
        descriptors["kappa3"] = round(Descriptors.Kappa3(mol), 3)
        
        # Drug-likeness scores
        descriptors["QED"] = round(self.calculate_qed(mol), 3)
        descriptors["SA_score"] = round(self.calculate_sa_score(mol), 2)
        
        # Lipinski violations
        descriptors["lipinski_violations"] = self._count_lipinski_violations(mol)
        
        # Molecular scaffold
        descriptors["murcko_scaffold"] = self._get_murcko_scaffold(mol)
        
        return descriptors
        
    def calculate_qed(self, mol: Chem.Mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness) score.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            QED score between 0 and 1
        """
        try:
            return qed(mol)
        except Exception as e:
            logger.warning(f"QED calculation failed: {e}")
            return 0.0
            
    def calculate_sa_score(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score.
        
        Based on fragment contributions and complexity.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            SA score between 1 (easy) and 10 (difficult)
        """
        # Simplified SA score calculation
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = Lipinski.RingCount(mol)
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        num_macrocycles = self._count_macrocycles(mol)
        num_bridgeheads = self._count_bridgehead_atoms(mol)
        
        # Base score from molecular size
        size_score = 1 + math.log10(num_atoms) * 2
        
        # Penalties for complexity
        ring_penalty = num_rings * 0.5
        stereo_penalty = num_stereo * 0.5
        macrocycle_penalty = num_macrocycles * 1.0
        bridgehead_penalty = num_bridgeheads * 0.75
        
        # Bonus for common fragments
        fragment_bonus = self._calculate_fragment_bonus(mol)
        
        sa_score = size_score + ring_penalty + stereo_penalty + macrocycle_penalty + bridgehead_penalty - fragment_bonus
        
        # Normalize to 1-10 scale
        sa_score = max(1.0, min(10.0, sa_score))
        
        return sa_score
        
    def calculate_lipinski_descriptors(self, mol: Chem.Mol) -> Dict[str, Union[float, int]]:
        """Calculate Lipinski rule of five descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with Lipinski descriptors
        """
        return {
            "MW": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "violations": self._count_lipinski_violations(mol)
        }
        
    def calculate_veber_descriptors(self, mol: Chem.Mol) -> Dict[str, Union[float, int]]:
        """Calculate Veber's oral bioavailability descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with Veber descriptors
        """
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        return {
            "rotatable_bonds": rotatable_bonds,
            "TPSA": tpsa,
            "passes_veber": rotatable_bonds <= 10 and tpsa <= 140
        }
        
    def calculate_ro3_descriptors(self, mol: Chem.Mol) -> Dict[str, Union[float, int]]:
        """Calculate Rule of Three descriptors (fragment-like).
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with RO3 descriptors
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        
        violations = sum([
            mw > 300,
            logp > 3,
            hba > 3,
            hbd > 3,
            rotatable > 3
        ])
        
        return {
            "MW": mw,
            "logP": logp,
            "HBA": hba,
            "HBD": hbd,
            "rotatable_bonds": rotatable,
            "violations": violations,
            "passes_ro3": violations == 0
        }
        
    def calculate_ghose_descriptors(self, mol: Chem.Mol) -> Dict[str, Union[float, int, bool]]:
        """Calculate Ghose filter descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with Ghose descriptors
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        atoms = mol.GetNumHeavyAtoms()
        mr = Crippen.MolMR(mol)
        
        passes = all([
            160 <= mw <= 480,
            -0.4 <= logp <= 5.6,
            20 <= atoms <= 70,
            40 <= mr <= 130
        ])
        
        return {
            "MW": mw,
            "logP": logp,
            "heavy_atoms": atoms,
            "molar_refractivity": mr,
            "passes_ghose": passes
        }
        
    def calculate_egan_descriptors(self, mol: Chem.Mol) -> Dict[str, Union[float, bool]]:
        """Calculate Egan filter descriptors (BBB permeation).
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with Egan descriptors
        """
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        
        passes = tpsa <= 90 and logp <= 5.88
        
        return {
            "TPSA": tpsa,
            "logP": logp,
            "passes_egan": passes,
            "bbb_permeable": passes
        }
        
    def calculate_chemical_beauty(self, mol: Chem.Mol) -> float:
        """Calculate chemical beauty score based on symmetry and simplicity.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Beauty score between 0 and 1
        """
        # Factors that contribute to chemical beauty
        num_atoms = mol.GetNumHeavyAtoms()
        
        # Symmetry score
        symmetry_score = self._calculate_symmetry_score(mol)
        
        # Simplicity score (fewer heteroatoms, rings)
        num_heteroatoms = Lipinski.NumHeteroatoms(mol)
        num_rings = Lipinski.RingCount(mol)
        simplicity_score = 1.0 / (1 + num_heteroatoms * 0.1 + num_rings * 0.2)
        
        # Aromaticity score
        aromatic_atoms = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
        aromaticity_score = aromatic_atoms / num_atoms if num_atoms > 0 else 0
        
        # Combine scores
        beauty_score = (symmetry_score * 0.4 + simplicity_score * 0.3 + aromaticity_score * 0.3)
        
        return round(beauty_score, 3)
        
    # Helper methods
    def _count_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Count Lipinski rule violations."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])
        
        return violations
        
    def _count_unspecified_stereocenters(self, mol: Chem.Mol) -> int:
        """Count unspecified stereocenters."""
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return sum(1 for center in chiral_centers if center[1] == '?')
        
    def _count_macrocycles(self, mol: Chem.Mol) -> int:
        """Count macrocycles (rings with 7+ atoms)."""
        ring_info = mol.GetRingInfo()
        return sum(1 for ring in ring_info.AtomRings() if len(ring) >= 7)
        
    def _count_bridgehead_atoms(self, mol: Chem.Mol) -> int:
        """Count bridgehead atoms (atoms in 2+ rings)."""
        ring_info = mol.GetRingInfo()
        atom_ring_counts = defaultdict(int)
        
        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                atom_ring_counts[atom_idx] += 1
                
        return sum(1 for count in atom_ring_counts.values() if count >= 2)
        
    def _calculate_fragment_bonus(self, mol: Chem.Mol) -> float:
        """Calculate bonus for common fragments."""
        # Common functional groups that are synthetically accessible
        common_fragments = [
            "c1ccccc1",  # Benzene
            "C(=O)O",    # Carboxylic acid
            "C(=O)N",    # Amide
            "C(=O)",     # Carbonyl
            "O",         # Ether/alcohol
            "N",         # Amine
            "C#N",       # Nitrile
            "S(=O)(=O)", # Sulfonyl
        ]
        
        bonus = 0.0
        mol_smiles = Chem.MolToSmiles(mol)
        
        for fragment in common_fragments:
            pattern = Chem.MolFromSmarts(fragment)
            if pattern and mol.HasSubstructMatch(pattern):
                bonus += 0.1
                
        return min(bonus, 1.0)  # Cap at 1.0
        
    def _get_murcko_scaffold(self, mol: Chem.Mol) -> str:
        """Get Murcko scaffold SMILES."""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return ""
            
    def _calculate_symmetry_score(self, mol: Chem.Mol) -> float:
        """Calculate molecular symmetry score."""
        # Simplified symmetry calculation based on atom connectivity
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        if not degrees:
            return 0.0
            
        # Check for repeated patterns
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
            
        # Higher score for more repeated patterns
        total_atoms = len(degrees)
        symmetry_score = sum(count ** 2 for count in degree_counts.values()) / (total_atoms ** 2)
        
        return symmetry_score
        
    def _load_sa_score_data(self):
        """Load synthetic accessibility score data."""
        # In a real implementation, this would load fragment scores from a database
        # For now, using a simplified version
        self._common_fragments = {
            "c1ccccc1": -0.5,  # Benzene ring
            "C(=O)O": -0.3,    # Carboxylic acid
            "C(=O)N": -0.3,    # Amide
            "CC": -0.1,        # Ethyl
            "C": -0.05,        # Methyl
        }