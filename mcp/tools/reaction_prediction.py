"""Reaction prediction and feasibility tools for MCP."""

from typing import Dict, List, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import numpy as np
import logging

logger = logging.getLogger(__name__)


async def check_feasibility(
    reactants: List[str],
    products: List[str],
    conditions: Optional[Dict[str, str]] = None
) -> Dict:
    """Check if a chemical reaction is feasible.
    
    Args:
        reactants: List of SMILES strings for reactants
        products: List of SMILES strings for expected products
        conditions: Optional reaction conditions (solvent, temperature, etc.)
        
    Returns:
        Dictionary with feasibility analysis
    """
    try:
        # Parse molecules
        reactant_mols = []
        for r_smiles in reactants:
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is None:
                return {"error": f"Invalid reactant SMILES: {r_smiles}"}
            reactant_mols.append(mol)
            
        product_mols = []
        for p_smiles in products:
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is None:
                return {"error": f"Invalid product SMILES: {p_smiles}"}
            product_mols.append(mol)
        
        results = {
            "reactants": reactants,
            "products": products,
            "feasible": False,
            "confidence": 0.0,
            "reaction_type": None,
            "mechanism": None,
            "issues": [],
            "suggestions": []
        }
        
        # Analyze reaction
        reaction_analysis = analyze_reaction(reactant_mols, product_mols)
        results.update(reaction_analysis)
        
        # Check atom balance
        atom_balance = check_atom_balance(reactant_mols, product_mols)
        if not atom_balance["balanced"]:
            results["issues"].append(f"Atom imbalance: {atom_balance['message']}")
            results["feasible"] = False
        
        # Identify reaction type
        reaction_type = identify_reaction_type(reactant_mols, product_mols)
        results["reaction_type"] = reaction_type
        
        # Check feasibility based on reaction type
        if reaction_type:
            feasibility = assess_reaction_feasibility(reaction_type, reactant_mols, product_mols, conditions)
            results.update(feasibility)
        
        # Predict mechanism
        if results["feasible"]:
            results["mechanism"] = predict_mechanism(reaction_type, reactant_mols, product_mols)
        
        # Suggest conditions if not provided
        if not conditions and reaction_type:
            results["suggested_conditions"] = suggest_conditions(reaction_type, reactant_mols)
        
        return results
        
    except Exception as e:
        return {"error": f"Reaction analysis failed: {str(e)}"}


def analyze_reaction(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Perform comprehensive reaction analysis."""
    analysis = {
        "bond_changes": analyze_bond_changes(reactants, products),
        "functional_group_changes": analyze_functional_group_changes(reactants, products),
        "stereochemistry_changes": analyze_stereochemistry_changes(reactants, products),
        "energy_estimate": estimate_reaction_energy(reactants, products)
    }
    
    return analysis


def check_atom_balance(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check if atoms are balanced between reactants and products."""
    # Count atoms in reactants
    reactant_atoms = {}
    for mol in reactants:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            reactant_atoms[symbol] = reactant_atoms.get(symbol, 0) + 1
    
    # Count atoms in products
    product_atoms = {}
    for mol in products:
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            product_atoms[symbol] = product_atoms.get(symbol, 0) + 1
    
    # Compare
    balanced = True
    message = ""
    
    all_atoms = set(reactant_atoms.keys()) | set(product_atoms.keys())
    for atom in all_atoms:
        r_count = reactant_atoms.get(atom, 0)
        p_count = product_atoms.get(atom, 0)
        
        if r_count != p_count:
            balanced = False
            message += f"{atom}: {r_count} → {p_count}; "
    
    return {
        "balanced": balanced,
        "message": message.rstrip("; ") if message else "Atoms balanced",
        "reactant_atoms": reactant_atoms,
        "product_atoms": product_atoms
    }


def identify_reaction_type(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Optional[str]:
    """Identify the type of reaction."""
    # Get functional groups in reactants and products
    reactant_fgs = set()
    for mol in reactants:
        reactant_fgs.update(identify_functional_groups_detailed(mol))
    
    product_fgs = set()
    for mol in products:
        product_fgs.update(identify_functional_groups_detailed(mol))
    
    # Lost and gained functional groups
    lost_fgs = reactant_fgs - product_fgs
    gained_fgs = product_fgs - reactant_fgs
    
    # Pattern matching for reaction types
    if "carboxylic_acid" in reactant_fgs and "alcohol" in reactant_fgs and "ester" in gained_fgs:
        return "esterification"
    
    elif "carboxylic_acid" in reactant_fgs and "amine" in reactant_fgs and "amide" in gained_fgs:
        return "amidation"
    
    elif "aldehyde" in reactant_fgs or "ketone" in reactant_fgs:
        if "alcohol" in gained_fgs:
            return "reduction"
        elif "alkene" in gained_fgs:
            return "wittig" if "phosphonium" in reactant_fgs else "aldol"
    
    elif "alcohol" in reactant_fgs and ("aldehyde" in gained_fgs or "ketone" in gained_fgs):
        return "oxidation"
    
    elif "halide" in reactant_fgs:
        if "alcohol" in gained_fgs:
            return "nucleophilic_substitution"
        elif "alkene" in gained_fgs:
            return "elimination"
    
    elif "alkene" in reactant_fgs:
        if "halide" in gained_fgs:
            return "halogenation"
        elif "alcohol" in gained_fgs:
            return "hydration"
        elif len(products) == 1 and products[0].GetNumAtoms() > sum(r.GetNumAtoms() for r in reactants):
            return "diels_alder"
    
    elif check_coupling_reaction(reactants, products):
        return identify_coupling_type(reactants, products)
    
    # Check for rearrangements
    if len(reactants) == 1 and len(products) == 1:
        if reactants[0].GetNumAtoms() == products[0].GetNumAtoms():
            return "rearrangement"
    
    return "unknown"


def assess_reaction_feasibility(reaction_type: str, reactants: List[Chem.Mol], 
                              products: List[Chem.Mol], conditions: Optional[Dict]) -> Dict:
    """Assess feasibility of identified reaction type."""
    feasibility = {
        "feasible": False,
        "confidence": 0.0,
        "issues": [],
        "suggestions": []
    }
    
    # Reaction-specific feasibility checks
    if reaction_type == "esterification":
        check = check_esterification_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type == "amidation":
        check = check_amidation_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type == "nucleophilic_substitution":
        check = check_sn_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type == "elimination":
        check = check_elimination_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type == "oxidation":
        check = check_oxidation_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type == "reduction":
        check = check_reduction_feasibility(reactants, products)
        feasibility.update(check)
        
    elif reaction_type in ["suzuki", "heck", "sonogashira", "stille"]:
        check = check_coupling_feasibility(reaction_type, reactants, products)
        feasibility.update(check)
        
    else:
        # Generic feasibility based on thermodynamics estimate
        energy = estimate_reaction_energy(reactants, products)
        if energy < 50:  # kcal/mol threshold
            feasibility["feasible"] = True
            feasibility["confidence"] = 0.6
        else:
            feasibility["issues"].append("High activation energy predicted")
    
    # Adjust confidence based on conditions
    if conditions:
        condition_check = check_condition_compatibility(reaction_type, conditions)
        feasibility["confidence"] *= condition_check["compatibility"]
        if condition_check["issues"]:
            feasibility["issues"].extend(condition_check["issues"])
    
    return feasibility


def predict_mechanism(reaction_type: str, reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Predict reaction mechanism."""
    mechanism = {
        "type": reaction_type,
        "steps": [],
        "key_intermediates": [],
        "rate_determining_step": None
    }
    
    if reaction_type == "nucleophilic_substitution":
        # Determine SN1 vs SN2
        substrate = find_substrate_with_leaving_group(reactants)
        if substrate:
            if is_tertiary_carbon(substrate):
                mechanism["subtype"] = "SN1"
                mechanism["steps"] = [
                    "Leaving group departure forming carbocation",
                    "Nucleophile attacks carbocation"
                ]
                mechanism["key_intermediates"] = ["carbocation"]
            else:
                mechanism["subtype"] = "SN2"
                mechanism["steps"] = [
                    "Concerted backside attack by nucleophile"
                ]
    
    elif reaction_type == "elimination":
        # E1 vs E2
        substrate = find_substrate_with_leaving_group(reactants)
        if substrate and is_tertiary_carbon(substrate):
            mechanism["subtype"] = "E1"
            mechanism["steps"] = [
                "Leaving group departure forming carbocation",
                "Base abstracts β-proton"
            ]
        else:
            mechanism["subtype"] = "E2"
            mechanism["steps"] = [
                "Concerted elimination with antiperiplanar geometry"
            ]
    
    elif reaction_type == "esterification":
        mechanism["steps"] = [
            "Protonation of carboxylic acid",
            "Nucleophilic attack by alcohol",
            "Proton transfer",
            "Loss of water",
            "Deprotonation"
        ]
        mechanism["key_intermediates"] = ["tetrahedral intermediate"]
        mechanism["rate_determining_step"] = "Nucleophilic attack"
    
    elif reaction_type == "aldol":
        mechanism["steps"] = [
            "Enolate formation",
            "Nucleophilic attack on carbonyl",
            "Protonation of alkoxide"
        ]
        mechanism["key_intermediates"] = ["enolate", "β-hydroxy carbonyl"]
    
    return mechanism


def suggest_conditions(reaction_type: str, reactants: List[Chem.Mol]) -> Dict:
    """Suggest reaction conditions based on reaction type."""
    conditions = {
        "esterification": {
            "catalyst": "H2SO4 or p-TsOH",
            "solvent": "Toluene with Dean-Stark trap",
            "temperature": "Reflux",
            "time": "4-8 hours",
            "notes": "Remove water to drive equilibrium"
        },
        "amidation": {
            "catalyst": "DCC/DMAP or EDC/HOBt",
            "solvent": "DCM or DMF",
            "temperature": "0°C to RT",
            "time": "2-12 hours",
            "notes": "Use coupling reagent for mild conditions"
        },
        "nucleophilic_substitution": {
            "solvent": "Polar aprotic (DMF, DMSO) for SN2",
            "temperature": "RT to 80°C",
            "time": "2-24 hours",
            "notes": "Use excess nucleophile"
        },
        "elimination": {
            "base": "Strong base (KOtBu, DBU)",
            "solvent": "Polar aprotic",
            "temperature": "RT to 100°C",
            "time": "1-4 hours"
        },
        "oxidation": {
            "oxidant": "Choose based on substrate",
            "primary_alcohol": "PCC for aldehyde, Jones for acid",
            "secondary_alcohol": "PCC, Swern, or DMP",
            "temperature": "Varies by method"
        },
        "reduction": {
            "reductant": "NaBH4 for mild, LiAlH4 for strong",
            "solvent": "MeOH for NaBH4, THF for LiAlH4",
            "temperature": "0°C to RT",
            "time": "0.5-4 hours"
        },
        "suzuki": {
            "catalyst": "Pd(PPh3)4 or Pd(OAc)2/ligand",
            "base": "K2CO3, K3PO4, or Cs2CO3",
            "solvent": "Dioxane/H2O or DMF",
            "temperature": "80-100°C",
            "time": "4-24 hours",
            "atmosphere": "Inert (N2 or Ar)"
        }
    }
    
    return conditions.get(reaction_type, {
        "general": "Consult literature for specific conditions",
        "temperature": "RT to reflux",
        "time": "Monitor by TLC"
    })


# Helper functions

def analyze_bond_changes(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Analyze bond changes in reaction."""
    # This is simplified - full implementation would use atom mapping
    reactant_bonds = count_bond_types(reactants)
    product_bonds = count_bond_types(products)
    
    formed = []
    broken = []
    
    for bond_type, count in product_bonds.items():
        diff = count - reactant_bonds.get(bond_type, 0)
        if diff > 0:
            formed.append(f"{diff} {bond_type}")
    
    for bond_type, count in reactant_bonds.items():
        diff = count - product_bonds.get(bond_type, 0)
        if diff > 0:
            broken.append(f"{diff} {bond_type}")
    
    return {
        "bonds_formed": formed,
        "bonds_broken": broken,
        "net_change": len(formed) - len(broken)
    }


def count_bond_types(molecules: List[Chem.Mol]) -> Dict[str, int]:
    """Count types of bonds in molecules."""
    bond_counts = {}
    
    for mol in molecules:
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtom().GetSymbol()
            end = bond.GetEndAtom().GetSymbol()
            bond_order = bond.GetBondTypeAsDouble()
            
            # Create canonical bond representation
            atoms = sorted([begin, end])
            bond_str = f"{atoms[0]}-{atoms[1]}"
            if bond_order == 2:
                bond_str = f"{atoms[0]}={atoms[1]}"
            elif bond_order == 3:
                bond_str = f"{atoms[0]}≡{atoms[1]}"
            
            bond_counts[bond_str] = bond_counts.get(bond_str, 0) + 1
    
    return bond_counts


def analyze_functional_group_changes(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Analyze functional group transformations."""
    reactant_fgs = []
    for mol in reactants:
        reactant_fgs.extend(identify_functional_groups_detailed(mol))
    
    product_fgs = []
    for mol in products:
        product_fgs.extend(identify_functional_groups_detailed(mol))
    
    return {
        "reactant_groups": list(set(reactant_fgs)),
        "product_groups": list(set(product_fgs)),
        "groups_lost": list(set(reactant_fgs) - set(product_fgs)),
        "groups_formed": list(set(product_fgs) - set(reactant_fgs))
    }


def analyze_stereochemistry_changes(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Analyze stereochemistry changes."""
    reactant_stereo = count_stereocenters_detailed(reactants)
    product_stereo = count_stereocenters_detailed(products)
    
    return {
        "reactant_stereocenters": reactant_stereo,
        "product_stereocenters": product_stereo,
        "stereochemistry_retained": reactant_stereo["total"] == product_stereo["total"],
        "new_stereocenters": product_stereo["total"] - reactant_stereo["total"]
    }


def count_stereocenters_detailed(molecules: List[Chem.Mol]) -> Dict:
    """Count stereocenters in molecules."""
    total = 0
    assigned = 0
    
    for mol in molecules:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        total += len(centers)
        assigned += sum(1 for c in centers if c[1] != '?')
    
    return {
        "total": total,
        "assigned": assigned,
        "unassigned": total - assigned
    }


def estimate_reaction_energy(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> float:
    """Estimate reaction energy (very simplified)."""
    # This is a placeholder - real implementation would use:
    # - Bond dissociation energies
    # - Formation enthalpies
    # - Computational chemistry
    
    # Count certain bond types as proxy
    reactant_energy = sum(estimate_molecular_energy(mol) for mol in reactants)
    product_energy = sum(estimate_molecular_energy(mol) for mol in products)
    
    return abs(product_energy - reactant_energy)


def estimate_molecular_energy(mol: Chem.Mol) -> float:
    """Estimate molecular energy (simplified)."""
    energy = 0
    
    # Aromatic stabilization
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    energy -= aromatic_atoms * 5
    
    # Ring strain
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) == 3:
            energy += 25  # 3-membered ring strain
        elif len(ring) == 4:
            energy += 15  # 4-membered ring strain
    
    # Steric strain (simplified)
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 4:
            energy += 2  # Quaternary carbon
    
    return energy


def identify_functional_groups_detailed(mol: Chem.Mol) -> List[str]:
    """Identify functional groups in molecule."""
    groups = []
    
    patterns = {
        "carboxylic_acid": "C(=O)[OH]",
        "ester": "C(=O)O[C,c]",
        "amide": "C(=O)N",
        "amine": "[NX3]",
        "alcohol": "[OH][CX4]",
        "phenol": "[OH]c",
        "ether": "[OX2]([CX4])[CX4]",
        "aldehyde": "[CX3H1](=O)",
        "ketone": "[CX3](=[OX1])[CX4]",
        "alkene": "C=C",
        "alkyne": "C#C",
        "aromatic": "a",
        "halide": "[F,Cl,Br,I]",
        "nitrile": "C#N",
        "nitro": "[N+](=O)[O-]",
        "sulfide": "S",
        "phosphate": "P(=O)(O)(O)O",
        "phosphonium": "[P+]"
    }
    
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            groups.append(name)
    
    return groups


def check_coupling_reaction(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> bool:
    """Check if reaction is a coupling reaction."""
    # Look for C-C bond formation between fragments
    if len(reactants) >= 2 and len(products) >= 1:
        # Check if product contains both reactant fragments
        r_atoms = sum(r.GetNumAtoms() for r in reactants)
        p_atoms = products[0].GetNumAtoms() if products else 0
        
        # Account for leaving groups
        if p_atoms > r_atoms * 0.7:
            return True
    
    return False


def identify_coupling_type(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> str:
    """Identify specific coupling reaction type."""
    reactant_fgs = []
    for mol in reactants:
        fgs = identify_functional_groups_detailed(mol)
        reactant_fgs.append(set(fgs))
    
    # Check for specific coupling patterns
    for fgs in reactant_fgs:
        if "boronic_acid" in fgs or "boronate" in fgs:
            return "suzuki"
        elif "alkene" in fgs and "halide" in fgs:
            return "heck"
        elif "alkyne" in fgs:
            return "sonogashira"
        elif "stannane" in fgs:
            return "stille"
    
    return "coupling"


def check_esterification_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of esterification."""
    result = {"feasible": True, "confidence": 0.9, "issues": [], "suggestions": []}
    
    # Find carboxylic acid and alcohol
    has_acid = False
    has_alcohol = False
    
    for mol in reactants:
        if has_pattern(mol, "C(=O)[OH]"):
            has_acid = True
        if has_pattern(mol, "[OH][CX4]"):
            has_alcohol = True
    
    if not has_acid:
        result["issues"].append("No carboxylic acid found")
        result["feasible"] = False
    if not has_alcohol:
        result["issues"].append("No alcohol found")
        result["feasible"] = False
    
    # Check for hindered alcohols
    for mol in reactants:
        if is_hindered_alcohol(mol):
            result["confidence"] *= 0.7
            result["suggestions"].append("Use strong acid catalyst for hindered alcohol")
    
    return result


def check_amidation_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of amidation."""
    result = {"feasible": True, "confidence": 0.85, "issues": [], "suggestions": []}
    
    has_acid = False
    has_amine = False
    
    for mol in reactants:
        if has_pattern(mol, "C(=O)[OH]"):
            has_acid = True
        if has_pattern(mol, "[NX3]"):
            has_amine = True
    
    if not has_acid:
        result["issues"].append("No carboxylic acid found")
        result["feasible"] = False
    if not has_amine:
        result["issues"].append("No amine found")
        result["feasible"] = False
    
    # Direct amidation is difficult
    if result["feasible"]:
        result["suggestions"].append("Use coupling reagent (EDC, DCC) for better yields")
    
    return result


def check_sn_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of nucleophilic substitution."""
    result = {"feasible": True, "confidence": 0.8, "issues": [], "suggestions": []}
    
    # Find substrate with leaving group
    substrate = find_substrate_with_leaving_group(reactants)
    if not substrate:
        result["issues"].append("No suitable leaving group found")
        result["feasible"] = False
        return result
    
    # Check substrate type
    if is_primary_carbon(substrate):
        result["mechanism"] = "SN2"
        result["suggestions"].append("Use polar aprotic solvent")
    elif is_tertiary_carbon(substrate):
        result["mechanism"] = "SN1"
        result["suggestions"].append("Use polar protic solvent")
    else:
        result["mechanism"] = "SN2 (secondary)"
        result["confidence"] *= 0.8
    
    return result


def check_elimination_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of elimination."""
    result = {"feasible": True, "confidence": 0.8, "issues": [], "suggestions": []}
    
    # Check for β-hydrogen
    substrate = find_substrate_with_leaving_group(reactants)
    if substrate and not has_beta_hydrogen(substrate):
        result["issues"].append("No β-hydrogen available")
        result["feasible"] = False
    
    # Check product for alkene
    has_alkene = False
    for mol in products:
        if has_pattern(mol, "C=C"):
            has_alkene = True
            break
    
    if not has_alkene:
        result["issues"].append("No alkene formed in products")
        result["confidence"] *= 0.5
    
    return result


def check_oxidation_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of oxidation."""
    result = {"feasible": True, "confidence": 0.85, "issues": [], "suggestions": []}
    
    # Check for alcohol in reactants
    alcohol_type = None
    for mol in reactants:
        if has_pattern(mol, "[OH][CX4]"):
            if is_primary_alcohol(mol):
                alcohol_type = "primary"
            elif is_secondary_alcohol(mol):
                alcohol_type = "secondary"
            else:
                alcohol_type = "tertiary"
    
    if not alcohol_type:
        result["issues"].append("No alcohol found in reactants")
        result["feasible"] = False
    elif alcohol_type == "tertiary":
        result["issues"].append("Tertiary alcohols cannot be oxidized")
        result["feasible"] = False
    
    # Suggest appropriate oxidant
    if alcohol_type == "primary":
        result["suggestions"].append("Use PCC for aldehyde, Jones/KMnO4 for acid")
    elif alcohol_type == "secondary":
        result["suggestions"].append("Use PCC, Swern, or DMP for ketone")
    
    return result


def check_reduction_feasibility(reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of reduction."""
    result = {"feasible": True, "confidence": 0.9, "issues": [], "suggestions": []}
    
    # Check for carbonyl in reactants
    has_carbonyl = False
    carbonyl_type = None
    
    for mol in reactants:
        if has_pattern(mol, "[CX3H1](=O)"):
            has_carbonyl = True
            carbonyl_type = "aldehyde"
        elif has_pattern(mol, "[CX3](=[OX1])[CX4]"):
            has_carbonyl = True
            carbonyl_type = "ketone"
        elif has_pattern(mol, "C(=O)O"):
            has_carbonyl = True
            carbonyl_type = "carboxylic_acid"
    
    if not has_carbonyl:
        result["issues"].append("No reducible group found")
        result["feasible"] = False
    
    # Suggest reducing agent
    if carbonyl_type in ["aldehyde", "ketone"]:
        result["suggestions"].append("Use NaBH4 for mild reduction")
    elif carbonyl_type == "carboxylic_acid":
        result["suggestions"].append("Use LiAlH4 for carboxylic acid reduction")
    
    return result


def check_coupling_feasibility(reaction_type: str, reactants: List[Chem.Mol], products: List[Chem.Mol]) -> Dict:
    """Check feasibility of coupling reactions."""
    result = {"feasible": True, "confidence": 0.8, "issues": [], "suggestions": []}
    
    if reaction_type == "suzuki":
        # Check for boronic acid and halide
        has_boronic = False
        has_halide = False
        
        for mol in reactants:
            if has_pattern(mol, "B(O)(O)"):
                has_boronic = True
            if has_pattern(mol, "[Cl,Br,I]"):
                has_halide = True
        
        if not has_boronic:
            result["issues"].append("No boronic acid/ester found")
            result["feasible"] = False
        if not has_halide:
            result["issues"].append("No aryl halide found")
            result["feasible"] = False
            
        result["suggestions"].append("Use Pd catalyst with phosphine ligand")
    
    return result


def check_condition_compatibility(reaction_type: str, conditions: Dict[str, str]) -> Dict:
    """Check if conditions are compatible with reaction type."""
    compatibility = {"compatibility": 1.0, "issues": []}
    
    # Check temperature
    if "temperature" in conditions:
        temp = conditions["temperature"].lower()
        if reaction_type in ["wittig", "grignard"] and "heat" in temp:
            compatibility["compatibility"] *= 0.5
            compatibility["issues"].append("High temperature may decompose reagents")
    
    # Check solvent
    if "solvent" in conditions:
        solvent = conditions["solvent"].lower()
        if reaction_type == "grignard" and "water" in solvent:
            compatibility["compatibility"] = 0
            compatibility["issues"].append("Grignard reagents incompatible with protic solvents")
    
    return compatibility


# Utility functions

def has_pattern(mol: Chem.Mol, smarts: str) -> bool:
    """Check if molecule contains SMARTS pattern."""
    pattern = Chem.MolFromSmarts(smarts)
    return pattern is not None and mol.HasSubstructMatch(pattern)


def find_substrate_with_leaving_group(reactants: List[Chem.Mol]) -> Optional[Chem.Mol]:
    """Find substrate with good leaving group."""
    leaving_groups = ["[Cl,Br,I]", "OS(=O)(=O)", "O[Ts]"]
    
    for mol in reactants:
        for lg in leaving_groups:
            if has_pattern(mol, lg):
                return mol
    
    return None


def is_primary_carbon(mol: Chem.Mol) -> bool:
    """Check if leaving group is on primary carbon."""
    # Simplified check
    return has_pattern(mol, "[CH2][Cl,Br,I]")


def is_secondary_carbon(mol: Chem.Mol) -> bool:
    """Check if leaving group is on secondary carbon."""
    return has_pattern(mol, "[CH]([C])[Cl,Br,I]")


def is_tertiary_carbon(mol: Chem.Mol) -> bool:
    """Check if leaving group is on tertiary carbon."""
    return has_pattern(mol, "[C]([C])([C])[Cl,Br,I]")


def has_beta_hydrogen(mol: Chem.Mol) -> bool:
    """Check if molecule has β-hydrogen for elimination."""
    # Simplified - check for C-C-X pattern
    return has_pattern(mol, "CC[Cl,Br,I]")


def is_primary_alcohol(mol: Chem.Mol) -> bool:
    """Check if alcohol is primary."""
    return has_pattern(mol, "[CH2]O")


def is_secondary_alcohol(mol: Chem.Mol) -> bool:
    """Check if alcohol is secondary."""
    return has_pattern(mol, "[CH]([C])O")


def is_hindered_alcohol(mol: Chem.Mol) -> bool:
    """Check if alcohol is sterically hindered."""
    # Simplified check for branching near OH
    return has_pattern(mol, "C(C)(C)CO") or has_pattern(mol, "C(C)C(C)O")