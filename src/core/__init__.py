"""Core functionality for molecular representation and processing with advanced ML."""

from .molecular_graph import MolecularGraph
from .hyperposition_tokenizer import (
    HyperMolecularProcessor,
    MolecularHyperToken,
    ChemicalTokenType,
    ChemicalTransform,
    ChemicalHyperDimensions,
    create_molecular_hyperprocessor
)

# Import error handling for optional dependencies
import logging
logger = logging.getLogger(__name__)

# Create missing modules with basic implementations
class MolecularDescriptors:
    """Basic molecular descriptors calculator"""
    
    def __init__(self):
        self.descriptors = {}
    
    def calculate(self, smiles: str) -> dict:
        """Calculate basic molecular descriptors"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logP': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'qed': Descriptors.qed(mol)
            }
        except ImportError:
            logger.warning("RDKit not available for descriptor calculation")
            return {}
        except Exception as e:
            logger.error(f"Error calculating descriptors: {e}")
            return {}

class MoleculeValidator:
    """Molecular validation and chemical rule checking"""
    
    def __init__(self):
        self.validation_rules = []
    
    def validate_smiles(self, smiles: str) -> dict:
        """Validate SMILES string"""
        result = {
            'valid': False,
            'canonical_smiles': '',
            'errors': [],
            'warnings': []
        }
        
        if not smiles:
            result['errors'].append('Empty SMILES string')
            return result
        
        try:
            from rdkit import Chem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['errors'].append('Invalid SMILES syntax')
                return result
            
            # Successful validation
            result['valid'] = True
            result['canonical_smiles'] = Chem.MolToSmiles(mol)
            
            # Check for common issues
            if mol.GetNumAtoms() == 0:
                result['warnings'].append('Empty molecule')
            
            if mol.GetNumAtoms() > 200:
                result['warnings'].append('Very large molecule (>200 atoms)')
            
            # Check valence
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                result['warnings'].append(f'Valence warning: {str(e)}')
            
        except ImportError:
            result['errors'].append('RDKit not available for validation')
        except Exception as e:
            result['errors'].append(f'Validation error: {str(e)}')
        
        return result
    
    def check_drug_likeness(self, smiles: str) -> dict:
        """Check drug-likeness rules"""
        result = {
            'lipinski_compliant': False,
            'veber_compliant': False,
            'violations': []
        }
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['violations'].append('Invalid SMILES')
                return result
            
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            violations = 0
            if mw > 500:
                violations += 1
                result['violations'].append('MW > 500')
            if logp > 5:
                violations += 1
                result['violations'].append('LogP > 5')
            if hbd > 5:
                violations += 1
                result['violations'].append('HBD > 5')
            if hba > 10:
                violations += 1
                result['violations'].append('HBA > 10')
            
            result['lipinski_compliant'] = violations <= 1
            
            # Veber rules
            rotb = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            
            result['veber_compliant'] = rotb <= 10 and tpsa <= 140
            if rotb > 10:
                result['violations'].append('Rotatable bonds > 10')
            if tpsa > 140:
                result['violations'].append('TPSA > 140')
            
        except ImportError:
            result['violations'].append('RDKit not available')
        except Exception as e:
            result['violations'].append(f'Error: {str(e)}')
        
        return result

def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit molecule object"""
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except ImportError:
        logger.warning("RDKit not available")
        return None

def mol_to_smiles(mol) -> str:
    """Convert RDKit molecule to SMILES string"""
    try:
        from rdkit import Chem
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except ImportError:
        logger.warning("RDKit not available")
        return ""

def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string"""
    mol = smiles_to_mol(smiles)
    return mol_to_smiles(mol) if mol else smiles

__all__ = [
    "MolecularGraph",
    "MolecularDescriptors",
    "MoleculeValidator", 
    "HyperMolecularProcessor",
    "MolecularHyperToken",
    "ChemicalTokenType",
    "ChemicalTransform", 
    "ChemicalHyperDimensions",
    "create_molecular_hyperprocessor",
    "smiles_to_mol",
    "mol_to_smiles",
    "canonicalize_smiles"
]
