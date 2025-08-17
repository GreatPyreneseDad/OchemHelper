"""Utility functions for molecular operations."""

from typing import List, Optional, Union, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Convert SMILES to RDKit molecule.
    
    Args:
        smiles: SMILES string
        sanitize: Whether to sanitize the molecule
        
    Returns:
        RDKit molecule object or None if conversion fails
    """
    if not smiles or not isinstance(smiles, str):
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
            
        if sanitize:
            Chem.SanitizeMol(mol)
            
        return mol
    except Exception as e:
        logger.warning(f"Failed to convert SMILES '{smiles}': {e}")
        return None


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True, 
                  isomeric: bool = True) -> Optional[str]:
    """Convert RDKit molecule to SMILES.
    
    Args:
        mol: RDKit molecule object
        canonical: Whether to return canonical SMILES
        isomeric: Whether to include stereochemistry
        
    Returns:
        SMILES string or None if conversion fails
    """
    if mol is None:
        return None
        
    try:
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    except Exception as e:
        logger.warning(f"Failed to convert molecule to SMILES: {e}")
        return None


def smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert SMILES to InChI.
    
    Args:
        smiles: SMILES string
        
    Returns:
        InChI string or None if conversion fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    try:
        return Chem.MolToInchi(mol)
    except Exception as e:
        logger.warning(f"Failed to convert SMILES to InChI: {e}")
        return None


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Convert SMILES to InChIKey.
    
    Args:
        smiles: SMILES string
        
    Returns:
        InChIKey string or None if conversion fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    try:
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        logger.warning(f"Failed to convert SMILES to InChIKey: {e}")
        return None


def standardize_molecule(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Standardize molecule representation.
    
    Performs:
    - Remove salts
    - Normalize tautomers
    - Remove charges where possible
    - Standardize stereochemistry
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Standardized molecule or None if standardization fails
    """
    if mol is None:
        return None
        
    try:
        # Remove salts
        remover = SaltRemover()
        mol = remover.StripMol(mol)
        
        # Uncharge molecule
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # Normalize
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        
        # Choose canonical tautomer
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
        
        return mol
    except Exception as e:
        logger.warning(f"Failed to standardize molecule: {e}")
        return None


def standardize_smiles(smiles: str) -> Optional[str]:
    """Standardize SMILES string.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized SMILES or None if standardization fails
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    mol = standardize_molecule(mol)
    if mol is None:
        return None
        
    return mol_to_smiles(mol)


def generate_2d_coords(mol: Chem.Mol) -> Chem.Mol:
    """Generate 2D coordinates for molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Molecule with 2D coordinates
    """
    if mol is None:
        return None
        
    try:
        mol_copy = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_copy)
        return mol_copy
    except Exception as e:
        logger.warning(f"Failed to generate 2D coordinates: {e}")
        return mol


def generate_3d_coords(mol: Chem.Mol, num_conformers: int = 1,
                      random_seed: int = 42) -> Chem.Mol:
    """Generate 3D coordinates for molecule.
    
    Args:
        mol: RDKit molecule object
        num_conformers: Number of conformers to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Molecule with 3D coordinates
    """
    if mol is None:
        return None
        
    try:
        mol_copy = Chem.AddHs(Chem.Mol(mol))
        
        # Generate conformers
        AllChem.EmbedMultipleConfs(
            mol_copy, 
            numConfs=num_conformers,
            randomSeed=random_seed,
            useRandomCoords=True
        )
        
        # Optimize with MMFF
        AllChem.MMFFOptimizeMoleculeConfs(mol_copy)
        
        # Remove hydrogens
        mol_copy = Chem.RemoveHs(mol_copy)
        
        return mol_copy
    except Exception as e:
        logger.warning(f"Failed to generate 3D coordinates: {e}")
        return mol


def calculate_fingerprint(mol: Chem.Mol, fp_type: str = "morgan",
                         radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Calculate molecular fingerprint.
    
    Args:
        mol: RDKit molecule object
        fp_type: Type of fingerprint ('morgan', 'rdkit', 'maccs', 'atom_pair', 'torsion')
        radius: Radius for Morgan fingerprint
        n_bits: Number of bits for fingerprint
        
    Returns:
        Numpy array of fingerprint or None if calculation fails
    """
    if mol is None:
        return None
        
    try:
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        elif fp_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif fp_type == "atom_pair":
            fp = AllChem.GetAtomPairFingerprint(mol)
        elif fp_type == "torsion":
            fp = AllChem.GetTopologicalTorsionFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
        # Convert to numpy array
        arr = np.zeros(len(fp))
        for i in range(len(fp)):
            arr[i] = fp[i]
            
        return arr
    except Exception as e:
        logger.warning(f"Failed to calculate fingerprint: {e}")
        return None


def calculate_similarity(mol1: Chem.Mol, mol2: Chem.Mol,
                        fp_type: str = "morgan") -> float:
    """Calculate Tanimoto similarity between two molecules.
    
    Args:
        mol1: First molecule
        mol2: Second molecule
        fp_type: Type of fingerprint to use
        
    Returns:
        Tanimoto similarity score (0-1)
    """
    if mol1 is None or mol2 is None:
        return 0.0
        
    try:
        if fp_type == "morgan":
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        elif fp_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
        else:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            
        return TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        logger.warning(f"Failed to calculate similarity: {e}")
        return 0.0


def get_scaffold(mol: Chem.Mol, generic: bool = False) -> Optional[str]:
    """Get Bemis-Murcko scaffold.
    
    Args:
        mol: RDKit molecule object
        generic: Whether to return generic scaffold (all atoms as carbons)
        
    Returns:
        Scaffold SMILES or None
    """
    if mol is None:
        return None
        
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
        else:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            
        return mol_to_smiles(scaffold)
    except Exception as e:
        logger.warning(f"Failed to get scaffold: {e}")
        return None


def fragment_molecule(mol: Chem.Mol, 
                     mode: str = "BRICS") -> List[Chem.Mol]:
    """Fragment molecule using BRICS or RECAP.
    
    Args:
        mol: RDKit molecule object
        mode: Fragmentation mode ('BRICS' or 'RECAP')
        
    Returns:
        List of fragment molecules
    """
    if mol is None:
        return []
        
    try:
        if mode == "BRICS":
            from rdkit.Chem import BRICS
            fragments = BRICS.BRICSDecompose(mol)
            return [smiles_to_mol(frag) for frag in fragments]
        elif mode == "RECAP":
            from rdkit.Chem import Recap
            recap = Recap.RecapDecompose(mol)
            fragments = recap.GetAllChildren()
            return [child.mol for child in fragments.values()]
        else:
            raise ValueError(f"Unknown fragmentation mode: {mode}")
    except Exception as e:
        logger.warning(f"Failed to fragment molecule: {e}")
        return []


def draw_molecule(mol: Chem.Mol, size: Tuple[int, int] = (300, 300),
                 highlight_atoms: Optional[List[int]] = None) -> Optional[str]:
    """Draw molecule and return as base64-encoded PNG.
    
    Args:
        mol: RDKit molecule object
        size: Image size (width, height)
        highlight_atoms: List of atom indices to highlight
        
    Returns:
        Base64-encoded PNG string or None
    """
    if mol is None:
        return None
        
    try:
        # Generate 2D coords if needed
        if not mol.GetNumConformers():
            mol = generate_2d_coords(mol)
            
        # Draw molecule
        drawer = Draw.MolDraw2DCairo(size[0], size[1])
        
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)
            
        drawer.FinishDrawing()
        
        # Convert to base64
        img_data = drawer.GetDrawingText()
        return base64.b64encode(img_data).decode()
    except Exception as e:
        logger.warning(f"Failed to draw molecule: {e}")
        return None


def draw_molecules_grid(mols: List[Chem.Mol], 
                       mols_per_row: int = 4,
                       img_size: Tuple[int, int] = (200, 200),
                       legends: Optional[List[str]] = None) -> Optional[str]:
    """Draw multiple molecules in a grid.
    
    Args:
        mols: List of RDKit molecule objects
        mols_per_row: Number of molecules per row
        img_size: Size of each molecule image
        legends: Optional legends for each molecule
        
    Returns:
        Base64-encoded PNG string or None
    """
    if not mols:
        return None
        
    try:
        # Filter out None molecules
        valid_mols = []
        valid_legends = []
        
        for i, mol in enumerate(mols):
            if mol is not None:
                valid_mols.append(mol)
                if legends and i < len(legends):
                    valid_legends.append(legends[i])
                else:
                    valid_legends.append("")
                    
        if not valid_mols:
            return None
            
        # Generate image
        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=mols_per_row,
            subImgSize=img_size,
            legends=valid_legends if legends else None
        )
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode()
    except Exception as e:
        logger.warning(f"Failed to draw molecule grid: {e}")
        return None


def enumerate_stereoisomers(mol: Chem.Mol, 
                           max_isomers: int = 32) -> List[Chem.Mol]:
    """Enumerate stereoisomers of a molecule.
    
    Args:
        mol: RDKit molecule object
        max_isomers: Maximum number of stereoisomers to generate
        
    Returns:
        List of stereoisomer molecules
    """
    if mol is None:
        return []
        
    try:
        from rdkit.Chem import EnumerateStereoisomers
        
        opts = EnumerateStereoisomers.StereoEnumerationOptions()
        opts.maxIsomers = max_isomers
        
        isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol, opts))
        return isomers
    except Exception as e:
        logger.warning(f"Failed to enumerate stereoisomers: {e}")
        return []


def enumerate_tautomers(mol: Chem.Mol, 
                       max_tautomers: int = 100) -> List[Chem.Mol]:
    """Enumerate tautomers of a molecule.
    
    Args:
        mol: RDKit molecule object
        max_tautomers: Maximum number of tautomers to generate
        
    Returns:
        List of tautomer molecules
    """
    if mol is None:
        return []
        
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(max_tautomers)
        
        tautomers = list(enumerator.Enumerate(mol))
        return tautomers
    except Exception as e:
        logger.warning(f"Failed to enumerate tautomers: {e}")
        return []


def get_atom_mapping(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[List[Tuple[int, int]]]:
    """Get atom mapping between two molecules.
    
    Args:
        mol1: First molecule
        mol2: Second molecule
        
    Returns:
        List of atom index pairs or None
    """
    if mol1 is None or mol2 is None:
        return None
        
    try:
        # Find maximum common substructure
        from rdkit.Chem import rdFMCS
        
        mcs = rdFMCS.FindMCS([mol1, mol2])
        if not mcs.smartsString:
            return []
            
        # Get atom mappings
        pattern = Chem.MolFromSmarts(mcs.smartsString)
        match1 = mol1.GetSubstructMatch(pattern)
        match2 = mol2.GetSubstructMatch(pattern)
        
        if len(match1) != len(match2):
            return []
            
        return list(zip(match1, match2))
    except Exception as e:
        logger.warning(f"Failed to get atom mapping: {e}")
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES or None
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
        
    return mol_to_smiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid.
    
    Args:
        smiles: SMILES string to check
        
    Returns:
        True if valid, False otherwise
    """
    return smiles_to_mol(smiles) is not None


def remove_stereochemistry(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove all stereochemistry from molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Molecule without stereochemistry
    """
    if mol is None:
        return None
        
    try:
        mol_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_copy)
        return mol_copy
    except Exception as e:
        logger.warning(f"Failed to remove stereochemistry: {e}")
        return mol