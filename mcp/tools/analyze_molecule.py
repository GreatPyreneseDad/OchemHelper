"""Molecular analysis tools for MCP."""

from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import numpy as np


async def analyze(smiles: str, properties: Optional[List[str]] = None) -> Dict:
    """Analyze molecule structure and calculate properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}
        
        # Default properties to calculate
        if properties is None:
            properties = ["MW", "logP", "TPSA", "HBD", "HBA", "rotatable_bonds", "rings"]
        
        results = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "properties": {}
        }
        
        # Calculate requested properties
        property_calculators = {
            "MW": lambda m: Descriptors.MolWt(m),
            "logP": lambda m: Crippen.MolLogP(m),
            "TPSA": lambda m: Descriptors.TPSA(m),
            "HBD": lambda m: Lipinski.NumHDonors(m),
            "HBA": lambda m: Lipinski.NumHAcceptors(m),
            "rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
            "rings": lambda m: Lipinski.RingCount(m),
            "aromatic_rings": lambda m: Lipinski.NumAromaticRings(m),
            "heavy_atoms": lambda m: Lipinski.HeavyAtomCount(m),
            "formal_charge": lambda m: Chem.rdmolops.GetFormalCharge(m),
            "num_stereocenters": lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
            "QED": lambda m: calculate_qed(m),
            "SA_score": lambda m: calculate_sa_score(m)
        }
        
        for prop in properties:
            if prop in property_calculators:
                try:
                    results["properties"][prop] = round(property_calculators[prop](mol), 3)
                except Exception as e:
                    results["properties"][prop] = f"Error: {str(e)}"
        
        # Add structural alerts
        results["structural_alerts"] = check_structural_alerts(mol)
        
        # Add drug-likeness rules
        results["drug_likeness"] = {
            "lipinski": check_lipinski(mol),
            "veber": check_veber(mol),
            "ghose": check_ghose(mol),
            "egan": check_egan(mol)
        }
        
        # Add functional groups
        results["functional_groups"] = identify_functional_groups(mol)
        
        return results
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def calculate_qed(mol) -> float:
    """Calculate quantitative estimate of drug-likeness (QED)."""
    # Simplified QED calculation
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    psa = Descriptors.TPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    arom = Lipinski.NumAromaticRings(mol)
    alerts = len(check_structural_alerts(mol))
    
    # Normalize properties (simplified)
    qed_properties = {
        'MW': np.exp(-((mw - 250) / 200) ** 2),
        'LOGP': np.exp(-((logp - 2.5) / 2) ** 2),
        'HBA': np.exp(-((hba - 5) / 3) ** 2),
        'HBD': np.exp(-((hbd - 2.5) / 2) ** 2),
        'PSA': np.exp(-((psa - 60) / 40) ** 2),
        'ROTB': np.exp(-((rotb - 4) / 3) ** 2),
        'AROM': np.exp(-((arom - 2) / 1.5) ** 2),
        'ALERTS': np.exp(-alerts)
    }
    
    # Calculate weighted average
    weights = [0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95]
    qed = np.prod([qed_properties[k] ** w for k, w in zip(qed_properties.keys(), weights)]) ** (1/sum(weights))
    
    return qed


def calculate_sa_score(mol) -> float:
    """Calculate synthetic accessibility score."""
    # Simplified SA score based on molecular complexity
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = Lipinski.RingCount(mol)
    num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    # Simple heuristic
    sa_score = 1 + (num_atoms / 20) + (num_rings / 3) + (num_stereo / 2)
    return min(sa_score, 10)  # Cap at 10


def check_structural_alerts(mol) -> List[str]:
    """Check for problematic structural features."""
    alerts = []
    
    # PAINS patterns (simplified)
    pains_smarts = [
        "[#6]1:[#6]:[#6]:[#6](:[#6]:[#6]:1)-[#6](=[#8])-[#6]",  # Quinone
        "[#7]=[#7+]=[#7-]",  # Azide
        "[#8]=[#8]",  # Peroxide
        "[#7+]([#8-])=O",  # Nitro
        "[#16](=[#8])(=[#8])",  # Sulfone
    ]
    
    for i, smarts in enumerate(pains_smarts):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(f"PAINS_{i+1}")
    
    # Toxicity alerts (simplified)
    tox_smarts = {
        "Epoxide": "C1OC1",
        "Aldehyde": "[CH]=O",
        "Hydrazine": "NN",
        "Isocyanate": "N=C=O",
        "Thiourea": "NC(=S)N"
    }
    
    for name, smarts in tox_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    
    return alerts


def check_lipinski(mol) -> Dict[str, bool]:
    """Check Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    return {
        "MW <= 500": mw <= 500,
        "LogP <= 5": logp <= 5,
        "HBD <= 5": hbd <= 5,
        "HBA <= 10": hba <= 10,
        "passes": all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    }


def check_veber(mol) -> Dict[str, bool]:
    """Check Veber's rules for oral bioavailability."""
    rotb = Lipinski.NumRotatableBonds(mol)
    psa = Descriptors.TPSA(mol)
    
    return {
        "RotatableBonds <= 10": rotb <= 10,
        "TPSA <= 140": psa <= 140,
        "passes": rotb <= 10 and psa <= 140
    }


def check_ghose(mol) -> Dict[str, bool]:
    """Check Ghose filter."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    atoms = mol.GetNumHeavyAtoms()
    mr = Crippen.MolMR(mol)
    
    return {
        "160 <= MW <= 480": 160 <= mw <= 480,
        "-0.4 <= LogP <= 5.6": -0.4 <= logp <= 5.6,
        "20 <= Atoms <= 70": 20 <= atoms <= 70,
        "40 <= MR <= 130": 40 <= mr <= 130,
        "passes": all([160 <= mw <= 480, -0.4 <= logp <= 5.6, 20 <= atoms <= 70, 40 <= mr <= 130])
    }


def check_egan(mol) -> Dict[str, bool]:
    """Check Egan filter (BBB permeation)."""
    psa = Descriptors.TPSA(mol)
    logp = Crippen.MolLogP(mol)
    
    return {
        "TPSA <= 90": psa <= 90,
        "LogP <= 5.88": logp <= 5.88,
        "passes": psa <= 90 and logp <= 5.88
    }


def identify_functional_groups(mol) -> List[str]:
    """Identify common functional groups."""
    functional_groups = {
        "Alcohol": "[OH]",
        "Amine": "[NX3;H2,H1;!$(NC=O)]",
        "Carboxylic_acid": "C(=O)[OH]",
        "Ester": "C(=O)O[#6]",
        "Amide": "C(=O)N",
        "Ketone": "[#6]C(=O)[#6]",
        "Ether": "[#6]O[#6]",
        "Nitrile": "C#N",
        "Sulfide": "[#6]S[#6]",
        "Halide": "[F,Cl,Br,I]",
        "Aromatic": "a",
        "Alkene": "C=C",
        "Alkyne": "C#C"
    }
    
    found_groups = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            count = len(mol.GetSubstructMatches(pattern))
            found_groups.append(f"{name}({count})")
    
    return found_groups