"""Molecular validation and filtering."""

from typing import List, Dict, Tuple, Optional, Set
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import re
import logging

logger = logging.getLogger(__name__)


class MoleculeValidator:
    """Validate molecules and check chemical rules."""
    
    def __init__(self):
        """Initialize validator with structural alert patterns."""
        self.load_structural_alerts()
        self._init_filter_catalogs()
        
    def validate_smiles(self, smiles: str) -> Tuple[bool, Optional[str]]:
        """Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not isinstance(smiles, str):
            return False, "Empty or invalid SMILES string"
            
        # Check for obviously invalid characters
        invalid_chars = set(smiles) - set("CNOPSFClBrIcnops()[]@+=#-:./%\\0123456789")
        if invalid_chars:
            return False, f"Invalid characters in SMILES: {invalid_chars}"
            
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False, "Invalid SMILES: Cannot parse structure"
                
            # Try to sanitize
            problems = Chem.DetectChemistryProblems(mol)
            if problems:
                problem_types = [p.GetType() for p in problems]
                return False, f"Chemistry problems detected: {problem_types}"
                
            Chem.SanitizeMol(mol)
            
            # Additional validation checks
            if mol.GetNumAtoms() == 0:
                return False, "Empty molecule"
                
            if mol.GetNumAtoms() > 200:
                return False, "Molecule too large (>200 heavy atoms)"
                
            # Check for disconnected structures
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                return False, f"Disconnected structure with {len(fragments)} fragments"
                
            return True, None
            
        except Exception as e:
            return False, f"SMILES validation error: {str(e)}"
            
    def check_drug_likeness(self, mol: Chem.Mol) -> Dict[str, bool]:
        """Check various drug-likeness rules.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with rule names and pass/fail status
        """
        results = {}
        
        # Lipinski's Rule of Five
        results["lipinski"] = self._check_lipinski(mol)
        
        # Veber's rules
        results["veber"] = self._check_veber(mol)
        
        # Ghose filter
        results["ghose"] = self._check_ghose(mol)
        
        # Egan filter (BBB permeation)
        results["egan"] = self._check_egan(mol)
        
        # REOS (Rapid Elimination Of Swill)
        results["reos"] = self._check_reos(mol)
        
        # Rule of Three (fragment-like)
        results["ro3"] = self._check_ro3(mol)
        
        # PAINS filter
        results["pains_free"] = not self._check_pains(mol)
        
        # Aggregator prediction
        results["non_aggregator"] = not self._is_aggregator(mol)
        
        # Overall drug-like
        results["drug_like"] = (
            results["lipinski"] and 
            results["veber"] and 
            results["pains_free"] and
            results["non_aggregator"]
        )
        
        return results
        
    def check_structural_alerts(self, mol: Chem.Mol) -> List[str]:
        """Check for problematic structural features.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of structural alerts found
        """
        alerts = []
        
        # Check PAINS patterns
        pains_hits = self._check_pains(mol)
        alerts.extend([f"PAINS:{hit}" for hit in pains_hits])
        
        # Check toxicophores
        tox_hits = self._check_toxicophores(mol)
        alerts.extend([f"Toxicophore:{hit}" for hit in tox_hits])
        
        # Check reactive groups
        reactive_hits = self._check_reactive_groups(mol)
        alerts.extend([f"Reactive:{hit}" for hit in reactive_hits])
        
        # Check unstable groups
        unstable_hits = self._check_unstable_groups(mol)
        alerts.extend([f"Unstable:{hit}" for hit in unstable_hits])
        
        # Check for problematic elements
        problem_elements = self._check_elements(mol)
        alerts.extend([f"Element:{elem}" for elem in problem_elements])
        
        return alerts
        
    def load_structural_alerts(self):
        """Load PAINS and toxicophore patterns."""
        # PAINS patterns (subset of common PAINS)
        self.pains_smarts = {
            "quinone_A": "[O,N,S]=C1C=CC(=[O,N,S])C=C1",
            "quinone_B": "[O,N,S]=C1C(=[O,N,S])C=CC=C1", 
            "azide": "[N-]=[N+]=[N-]",
            "peroxide": "OO",
            "hydroxylamine": "[OH]N",
            "hydrazine": "NN",
            "hydrazone": "C=NN",
            "nitroso": "N=O",
            "nitro": "[N+](=O)[O-]",
            "phosphate": "P(=O)(O)(O)O",
            "beta_lactam": "C1CC(=O)N1",
            "azo": "N=N",
            "diazo": "[N+]#[N]",
            "isocyanate": "N=C=O",
            "isothiocyanate": "N=C=S",
            "thiourea": "NC(=S)N",
            "michael_acceptor_1": "C=CC(=O)",
            "michael_acceptor_2": "C=CC#N",
            "epoxide": "C1OC1",
            "aziridine": "C1NC1"
        }
        
        # Toxicophore patterns
        self.toxicophore_smarts = {
            "alkyl_halide": "[C][Cl,Br,I]",
            "acyl_halide": "C(=O)[Cl,Br,I]",
            "aldehyde": "[CH]=O",
            "thiol": "[SH]",
            "cyano_pyridine": "c1ncccc1C#N",
            "primary_halide_sulfate": "COS(=O)(=O)O[Cl,Br,I]",
            "phosphonate": "P(=O)(O)O",
            "sulfonyl_halide": "S(=O)(=O)[Cl,Br]",
            "n_oxide": "[N+][O-]",
            "s_oxide": "[S+][O-]"
        }
        
        # Reactive functional groups
        self.reactive_smarts = {
            "acyl_chloride": "C(=O)Cl",
            "sulfonyl_chloride": "S(=O)(=O)Cl", 
            "anhydride": "C(=O)OC(=O)",
            "imide": "C(=O)NC(=O)",
            "activated_ester": "C(=O)ON",
            "alpha_halo_carbonyl": "[C;X4][Cl,Br,I]C(=O)",
            "dichloromethyl": "ClCCl",
            "trichloromethyl": "ClC(Cl)Cl",
            "acetal": "C(O)O",
            "hemiacetal": "C(O)[O,N,S]"
        }
        
        # Unstable groups
        self.unstable_smarts = {
            "enol": "C=C[OH]",
            "enamine": "C=CN",
            "hemi_ketal": "C(O)(O)C",
            "imine": "C=N",
            "gem_diol": "C(O)(O)",
            "ortho_ester": "C(O)(O)O",
            "peroxy": "COO",
            "n_hydroxyl": "N[OH]",
            "hydroxamic_acid": "C(=O)N[OH]"
        }
        
    def _init_filter_catalogs(self):
        """Initialize RDKit filter catalogs."""
        # PAINS catalog
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog(params)
        
        # BRENK catalog (unwanted functionalities)
        params_brenk = FilterCatalogParams()
        params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog(params_brenk)
        
    def _check_lipinski(self, mol: Chem.Mol) -> bool:
        """Check Lipinski's Rule of Five."""
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
        
        return violations <= 1  # Allow one violation
        
    def _check_veber(self, mol: Chem.Mol) -> bool:
        """Check Veber's rules for oral bioavailability."""
        rotatable = Lipinski.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        return rotatable <= 10 and tpsa <= 140
        
    def _check_ghose(self, mol: Chem.Mol) -> bool:
        """Check Ghose filter."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        atoms = mol.GetNumHeavyAtoms()
        mr = Crippen.MolMR(mol)
        
        return all([
            160 <= mw <= 480,
            -0.4 <= logp <= 5.6,
            20 <= atoms <= 70,
            40 <= mr <= 130
        ])
        
    def _check_egan(self, mol: Chem.Mol) -> bool:
        """Check Egan filter for BBB permeation."""
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        
        return tpsa <= 90 and logp <= 5.88
        
    def _check_reos(self, mol: Chem.Mol) -> bool:
        """Check REOS (Rapid Elimination Of Swill) filter."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        
        return all([
            200 <= mw <= 500,
            -5 <= logp <= 5,
            hba <= 10,
            hbd <= 5,
            abs(formal_charge) <= 2,
            rotatable <= 8
        ])
        
    def _check_ro3(self, mol: Chem.Mol) -> bool:
        """Check Rule of Three for fragments."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        
        return all([
            mw <= 300,
            logp <= 3,
            hba <= 3,
            hbd <= 3,
            rotatable <= 3
        ])
        
    def _check_pains(self, mol: Chem.Mol) -> List[str]:
        """Check for PAINS (Pan Assay INterference Structures)."""
        hits = []
        
        # Check custom PAINS patterns
        for name, smarts in self.pains_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                hits.append(name)
                
        # Check RDKit PAINS catalog
        entry = self.pains_catalog.GetFirstMatch(mol)
        if entry:
            hits.append(f"PAINS_{entry.GetDescription()}")
            
        return hits
        
    def _check_toxicophores(self, mol: Chem.Mol) -> List[str]:
        """Check for toxicophore patterns."""
        hits = []
        
        for name, smarts in self.toxicophore_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                hits.append(name)
                
        return hits
        
    def _check_reactive_groups(self, mol: Chem.Mol) -> List[str]:
        """Check for reactive functional groups."""
        hits = []
        
        for name, smarts in self.reactive_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                hits.append(name)
                
        return hits
        
    def _check_unstable_groups(self, mol: Chem.Mol) -> List[str]:
        """Check for unstable groups."""
        hits = []
        
        for name, smarts in self.unstable_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                hits.append(name)
                
        return hits
        
    def _check_elements(self, mol: Chem.Mol) -> List[str]:
        """Check for problematic elements."""
        allowed_elements = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, B, C, N, O, F, P, S, Cl, Br, I
        problem_elements = []
        
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num not in allowed_elements:
                symbol = atom.GetSymbol()
                if symbol not in problem_elements:
                    problem_elements.append(symbol)
                    
        return problem_elements
        
    def _is_aggregator(self, mol: Chem.Mol) -> bool:
        """Predict if molecule is likely to aggregate."""
        # Simplified aggregator prediction based on:
        # - High logP (hydrophobic)
        # - Large aromatic systems
        # - Lack of charged groups
        
        logp = Crippen.MolLogP(mol)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_atoms = mol.GetNumHeavyAtoms()
        aromatic_fraction = aromatic_atoms / total_atoms if total_atoms > 0 else 0
        formal_charge = abs(Chem.rdmolops.GetFormalCharge(mol))
        
        # High risk if hydrophobic, highly aromatic, and uncharged
        return (logp > 5 and aromatic_fraction > 0.5 and formal_charge == 0)
        
    def filter_molecules(self, smiles_list: List[str], 
                        filters: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, bool]]]:
        """Filter a list of molecules by specified criteria.
        
        Args:
            smiles_list: List of SMILES strings
            filters: List of filter names to apply (default: all drug-likeness filters)
            
        Returns:
            List of tuples (smiles, filter_results)
        """
        if filters is None:
            filters = ["lipinski", "veber", "pains_free", "non_aggregator"]
            
        results = []
        
        for smiles in smiles_list:
            valid, error = self.validate_smiles(smiles)
            if not valid:
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            drug_likeness = self.check_drug_likeness(mol)
            
            # Check if molecule passes all specified filters
            passes_filters = all(drug_likeness.get(f, False) for f in filters)
            
            if passes_filters:
                results.append((smiles, drug_likeness))
                
        return results