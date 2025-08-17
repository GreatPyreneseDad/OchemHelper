"""Property prediction tools for MCP."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from functools import lru_cache

logger = logging.getLogger(__name__)


async def predict_activity(smiles: str, targets: Optional[List[str]] = None) -> Dict:
    """Predict biological activity and ADMET properties.
    
    Args:
        smiles: SMILES string of the molecule
        targets: List of targets to predict (default: all)
        
    Returns:
        Dictionary with predictions
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
            
        results = {
            "smiles": smiles,
            "predictions": {}
        }
        
        # Default targets if none specified
        if targets is None:
            targets = ["bioavailability", "toxicity", "solubility", "permeability", 
                      "clearance", "half_life", "protein_binding", "bbb_penetration"]
        
        # Basic ADMET predictions
        if "bioavailability" in targets:
            results["predictions"]["bioavailability"] = predict_bioavailability(mol)
            
        if "toxicity" in targets:
            results["predictions"]["toxicity"] = predict_toxicity(mol)
            
        if "solubility" in targets:
            results["predictions"]["solubility"] = predict_solubility(mol)
            
        if "permeability" in targets:
            results["predictions"]["permeability"] = predict_permeability(mol)
            
        if "clearance" in targets:
            results["predictions"]["clearance"] = predict_clearance(mol)
            
        if "half_life" in targets:
            results["predictions"]["half_life"] = predict_half_life(mol)
            
        if "protein_binding" in targets:
            results["predictions"]["protein_binding"] = predict_protein_binding(mol)
            
        if "bbb_penetration" in targets:
            results["predictions"]["bbb_penetration"] = predict_bbb_penetration(mol)
            
        # Add confidence scores
        results["confidence"] = calculate_prediction_confidence(mol)
        
        return results
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


def predict_bioavailability(mol: Chem.Mol) -> Dict:
    """Predict oral bioavailability.
    
    Uses multiple rules and models to predict bioavailability.
    """
    # Lipinski's Rule of Five
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    # Veber's rules
    rotatable = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    veber_pass = rotatable <= 10 and tpsa <= 140
    
    # Egan filter (for absorption)
    egan_pass = tpsa <= 131.6 and logp <= 5.88
    
    # Simple bioavailability score
    score = 1.0
    score -= lipinski_violations * 0.2
    score -= 0.3 if not veber_pass else 0
    score -= 0.2 if not egan_pass else 0
    
    # Ensure score is between 0 and 1
    score = max(0, min(1, score))
    
    return {
        "score": round(score, 3),
        "category": "High" if score > 0.7 else "Medium" if score > 0.3 else "Low",
        "lipinski_violations": lipinski_violations,
        "veber_compliant": veber_pass,
        "egan_compliant": egan_pass,
        "details": {
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "rotatable_bonds": rotatable,
            "TPSA": round(tpsa, 2)
        }
    }


def predict_toxicity(mol: Chem.Mol) -> Dict:
    """Predict various toxicity endpoints.
    
    Includes acute toxicity, mutagenicity, and organ toxicity predictions.
    """
    # Structural alerts for toxicity
    toxic_patterns = {
        "nitroaromatic": "c1ccccc1[N+](=O)[O-]",
        "epoxide": "C1OC1",
        "michael_acceptor": "C=CC(=O)",
        "alkyl_halide": "C[Cl,Br,I]",
        "aromatic_amine": "c1ccccc1N",
        "hydrazine": "NN",
        "azide": "[N-]=[N+]=[N-]",
        "nitroso": "N=O",
        "quinone": "O=C1C=CC(=O)C=C1",
        "polycyclic_aromatic": "c1ccc2cc3ccccc3cc2c1"
    }
    
    alerts = []
    for name, smarts in toxic_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    
    # Calculate toxicity risk score
    risk_score = len(alerts) * 0.15
    
    # Additional factors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    
    # High lipophilicity can indicate toxicity
    if logp > 5:
        risk_score += 0.1
    
    # Very high MW might indicate poor clearance
    if mw > 600:
        risk_score += 0.1
    
    risk_score = min(risk_score, 1.0)
    
    return {
        "acute_toxicity": {
            "risk_score": round(risk_score, 3),
            "category": "High" if risk_score > 0.5 else "Medium" if risk_score > 0.2 else "Low",
            "structural_alerts": alerts
        },
        "mutagenicity": {
            "risk": "High" if any(alert in ["nitroaromatic", "aromatic_amine", "epoxide"] for alert in alerts) else "Low",
            "alerts": [a for a in alerts if a in ["nitroaromatic", "aromatic_amine", "epoxide"]]
        },
        "hepatotoxicity": {
            "risk": "Medium" if logp > 4 and mw > 400 else "Low",
            "factors": ["high_lipophilicity"] if logp > 4 else []
        },
        "cardiotoxicity": {
            "hERG_risk": "Medium" if logp > 3.5 and mw > 350 else "Low",
            "QT_prolongation_risk": "Check" if logp > 3.5 else "Low"
        }
    }


def predict_solubility(mol: Chem.Mol) -> Dict:
    """Predict aqueous solubility (log S).
    
    Uses ESOL method (Delaney, 2004) with modifications.
    """
    # Calculate descriptors for ESOL
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)
    aromatic_proportion = len([a for a in mol.GetAtoms() if a.GetIsAromatic()]) / mol.GetNumHeavyAtoms()
    
    # ESOL equation: log S = 0.16 - 0.63 logP - 0.0062 MW + 0.066 RB - 0.74 AP
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable - 0.74 * aromatic_proportion
    
    # Convert to mg/mL
    solubility_mg_ml = 10 ** log_s * mw
    
    # Categorize
    if log_s > -1:
        category = "Very soluble"
    elif log_s > -2:
        category = "Soluble"
    elif log_s > -3:
        category = "Slightly soluble"
    elif log_s > -4:
        category = "Poorly soluble"
    else:
        category = "Very poorly soluble"
    
    return {
        "log_s": round(log_s, 2),
        "solubility_mg_ml": round(solubility_mg_ml, 3),
        "category": category,
        "factors": {
            "logP": round(logp, 2),
            "MW": round(mw, 2),
            "rotatable_bonds": rotatable,
            "aromatic_proportion": round(aromatic_proportion, 2)
        }
    }


def predict_permeability(mol: Chem.Mol) -> Dict:
    """Predict membrane permeability.
    
    Includes Caco-2 and PAMPA permeability predictions.
    """
    # Calculate relevant descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    
    # Caco-2 permeability prediction (simplified)
    # High permeability if: MW < 500, 0 < logP < 5, TPSA < 90, HBD < 5
    high_perm_score = 0
    if mw < 500:
        high_perm_score += 0.25
    if 0 < logp < 5:
        high_perm_score += 0.25
    if tpsa < 90:
        high_perm_score += 0.25
    if hbd < 5:
        high_perm_score += 0.25
    
    # Estimate log Papp
    log_papp = -4.5 + 0.4 * logp - 0.01 * tpsa
    papp = 10 ** log_papp
    
    # PAMPA prediction
    pampa_permeable = tpsa < 120 and logp > -0.5 and logp < 5
    
    return {
        "caco2": {
            "permeability_score": round(high_perm_score, 2),
            "category": "High" if high_perm_score > 0.75 else "Medium" if high_perm_score > 0.5 else "Low",
            "log_papp_cm_s": round(log_papp, 2),
            "papp_10_6_cm_s": round(papp * 1e6, 2)
        },
        "pampa": {
            "permeable": pampa_permeable,
            "log_pe": round(-4.2 + 0.35 * logp - 0.008 * tpsa, 2) if pampa_permeable else None
        },
        "factors": {
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd
        }
    }


def predict_clearance(mol: Chem.Mol) -> Dict:
    """Predict metabolic clearance.
    
    Estimates hepatic clearance and identifies potential metabolic liabilities.
    """
    # Calculate descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Identify metabolic hotspots
    metabolic_sites = identify_metabolic_sites(mol)
    
    # Estimate intrinsic clearance (simplified model)
    # Higher lipophilicity and lower polarity generally increase clearance
    cl_score = 0.5 + 0.1 * min(logp, 5) - 0.005 * tpsa
    cl_score = max(0.1, min(1.0, cl_score))
    
    # Categorize clearance
    if cl_score > 0.7:
        category = "High"
        half_life_est = "< 2 hours"
    elif cl_score > 0.4:
        category = "Moderate"
        half_life_est = "2-5 hours"
    else:
        category = "Low"
        half_life_est = "> 5 hours"
    
    return {
        "hepatic_clearance": {
            "score": round(cl_score, 2),
            "category": category,
            "estimated_half_life": half_life_est
        },
        "metabolic_stability": {
            "stable": len(metabolic_sites) < 3,
            "liability_sites": len(metabolic_sites),
            "metabolic_hotspots": metabolic_sites
        },
        "cyp_interactions": {
            "cyp3a4_substrate": logp > 2 and mw > 300,
            "cyp2d6_substrate": has_basic_nitrogen(mol),
            "cyp2c9_substrate": has_acidic_group(mol)
        }
    }


def predict_half_life(mol: Chem.Mol) -> Dict:
    """Predict plasma half-life.
    
    Estimates based on clearance and volume of distribution.
    """
    # Get clearance prediction
    clearance = predict_clearance(mol)
    cl_score = clearance["hepatic_clearance"]["score"]
    
    # Estimate volume of distribution based on lipophilicity
    logp = Crippen.MolLogP(mol)
    vd_score = 0.3 + 0.1 * min(logp, 5)
    
    # Estimate half-life (simplified)
    # t1/2 = 0.693 * Vd / CL
    if cl_score > 0:
        t_half = 0.693 * vd_score / cl_score * 5  # Scaled to hours
    else:
        t_half = 24  # Default max
    
    # Categorize
    if t_half < 2:
        category = "Short"
    elif t_half < 6:
        category = "Medium"
    else:
        category = "Long"
    
    return {
        "t_half_hours": round(t_half, 1),
        "category": category,
        "clearance_score": round(cl_score, 2),
        "vd_score": round(vd_score, 2),
        "factors": {
            "lipophilicity": "High" if logp > 3 else "Medium" if logp > 1 else "Low",
            "metabolic_stability": clearance["metabolic_stability"]["stable"]
        }
    }


def predict_protein_binding(mol: Chem.Mol) -> Dict:
    """Predict plasma protein binding.
    
    Estimates fraction bound to plasma proteins.
    """
    # Calculate descriptors
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    formal_charge = Chem.rdmolops.GetFormalCharge(mol)
    
    # Protein binding correlates with lipophilicity and aromaticity
    # Basic drugs tend to bind to alpha-1-acid glycoprotein
    # Acidic drugs tend to bind to albumin
    
    # Base score from lipophilicity
    if logp > 4:
        base_score = 0.95
    elif logp > 2:
        base_score = 0.7 + 0.1 * (logp - 2)
    elif logp > 0:
        base_score = 0.3 + 0.2 * logp
    else:
        base_score = 0.3
    
    # Adjust for aromaticity
    if aromatic_atoms > 6:
        base_score += 0.1
    
    # Adjust for charge
    if formal_charge != 0:
        base_score += 0.05
    
    # Cap at realistic values
    fraction_bound = min(0.99, max(0.1, base_score))
    
    # Calculate free fraction
    fraction_unbound = 1 - fraction_bound
    
    return {
        "fraction_bound": round(fraction_bound, 3),
        "fraction_unbound": round(fraction_unbound, 3),
        "percent_bound": round(fraction_bound * 100, 1),
        "category": "High" if fraction_bound > 0.95 else "Medium" if fraction_bound > 0.7 else "Low",
        "primary_protein": "Albumin" if formal_charge < 0 else "Alpha-1-acid glycoprotein" if formal_charge > 0 else "Albumin",
        "factors": {
            "logP": round(logp, 2),
            "aromatic_atoms": aromatic_atoms,
            "charge": formal_charge
        }
    }


def predict_bbb_penetration(mol: Chem.Mol) -> Dict:
    """Predict blood-brain barrier penetration.
    
    Uses multiple models and rules for BBB permeability.
    """
    # Calculate descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    # CNS MPO (Central Nervous System Multiparameter Optimization)
    cns_mpo_score = calculate_cns_mpo(mol)
    
    # Simple BBB rules
    # Penetrant if: MW < 450, 1 < logP < 4, TPSA < 90, HBD < 5
    rules_passed = 0
    if mw < 450:
        rules_passed += 1
    if 1 < logp < 4:
        rules_passed += 1
    if tpsa < 90:
        rules_passed += 1
    if hbd < 5:
        rules_passed += 1
    
    bbb_score = rules_passed / 4
    
    # P-glycoprotein substrate prediction (simplified)
    pgp_substrate_likely = mw > 400 and logp > 2 and hba > 5
    
    return {
        "bbb_permeable": bbb_score > 0.75,
        "bbb_score": round(bbb_score, 2),
        "cns_mpo_score": round(cns_mpo_score, 2),
        "category": "High" if bbb_score > 0.75 else "Medium" if bbb_score > 0.5 else "Low",
        "pgp_substrate": pgp_substrate_likely,
        "factors": {
            "MW": round(mw, 2),
            "logP": round(logp, 2),
            "TPSA": round(tpsa, 2),
            "HBD": hbd,
            "rules_passed": rules_passed
        }
    }


# Helper functions
def identify_metabolic_sites(mol: Chem.Mol) -> List[str]:
    """Identify potential metabolic hotspots."""
    sites = []
    
    # Common metabolic transformations
    metabolic_patterns = {
        "N-dealkylation": "[CH3]N",
        "O-dealkylation": "[CH3]O",
        "aromatic_hydroxylation": "c1ccccc1",
        "aliphatic_hydroxylation": "[CH2]",
        "N-oxidation": "N",
        "S-oxidation": "S",
        "alcohol_oxidation": "[CH2]O",
        "aldehyde_oxidation": "[CH]=O"
    }
    
    for name, smarts in metabolic_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            sites.append(name)
    
    return sites


def has_basic_nitrogen(mol: Chem.Mol) -> bool:
    """Check if molecule has basic nitrogen."""
    # Simplified check for basic nitrogen
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Nitrogen
            if atom.GetFormalCharge() > 0:
                return True
            # Check for aliphatic amine
            if not atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
                return True
    return False


def has_acidic_group(mol: Chem.Mol) -> bool:
    """Check if molecule has acidic group."""
    acidic_patterns = ["C(=O)[OH]", "S(=O)(=O)[OH]", "P(=O)([OH])[OH]"]
    
    for smarts in acidic_patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    return False


def calculate_cns_mpo(mol: Chem.Mol) -> float:
    """Calculate CNS MPO score."""
    score = 0
    
    # MW: optimal 250-350
    mw = Descriptors.MolWt(mol)
    if 250 <= mw <= 350:
        score += 1
    elif 200 <= mw <= 400:
        score += 0.5
    
    # logP: optimal 1-3
    logp = Crippen.MolLogP(mol)
    if 1 <= logp <= 3:
        score += 1
    elif 0 <= logp <= 4:
        score += 0.5
    
    # TPSA: optimal < 75
    tpsa = Descriptors.TPSA(mol)
    if tpsa <= 75:
        score += 1
    elif tpsa <= 90:
        score += 0.5
    
    # HBD: optimal <= 2
    hbd = Lipinski.NumHDonors(mol)
    if hbd <= 2:
        score += 1
    elif hbd <= 3:
        score += 0.5
    
    # pKa: optimal 7.5-9.5 (simplified - just check for basic N)
    if has_basic_nitrogen(mol):
        score += 0.5
    
    # Normalize to 0-1 scale
    return score / 5


def calculate_prediction_confidence(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate confidence scores for predictions."""
    # Base confidence on molecular complexity and drug-likeness
    mw = Descriptors.MolWt(mol)
    atoms = mol.GetNumHeavyAtoms()
    
    # Confidence is higher for drug-like molecules
    if 200 <= mw <= 500 and 10 <= atoms <= 50:
        base_confidence = 0.8
    elif 150 <= mw <= 600 and 5 <= atoms <= 70:
        base_confidence = 0.6
    else:
        base_confidence = 0.4
    
    return {
        "overall": round(base_confidence, 2),
        "admet": round(base_confidence * 0.9, 2),
        "bioactivity": round(base_confidence * 0.7, 2),
        "notes": "Predictions are computational estimates and should be validated experimentally"
    }