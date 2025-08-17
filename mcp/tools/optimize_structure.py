"""Structure optimization tools for MCP."""

from typing import Dict, List, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
from collections import defaultdict
import random
import logging

logger = logging.getLogger(__name__)


async def optimize_lead(
    lead_smiles: str,
    optimization_goals: Dict[str, float],
    maintain_scaffold: bool = True
) -> Dict:
    """Optimize a lead compound for better properties.
    
    Args:
        lead_smiles: SMILES of lead compound
        optimization_goals: Target property ranges (e.g., {'logP': [2, 4]})
        maintain_scaffold: Whether to keep core scaffold intact
        
    Returns:
        Dictionary with optimized molecules and analysis
    """
    try:
        lead_mol = Chem.MolFromSmiles(lead_smiles)
        if lead_mol is None:
            return {"error": "Invalid lead SMILES"}
            
        results = {
            "lead": lead_smiles,
            "lead_properties": calculate_properties(lead_mol),
            "scaffold": get_scaffold_smiles(lead_mol) if maintain_scaffold else None,
            "optimized_molecules": [],
            "optimization_strategy": determine_optimization_strategy(lead_mol, optimization_goals)
        }
        
        # Generate optimized analogs
        if maintain_scaffold:
            analogs = generate_scaffold_based_analogs(lead_mol, optimization_goals)
        else:
            analogs = generate_free_optimization(lead_mol, optimization_goals)
        
        # Score and rank analogs
        scored_analogs = score_optimized_molecules(analogs, optimization_goals, lead_mol)
        
        # Format results
        for i, (analog_mol, score, improvements) in enumerate(scored_analogs[:20]):
            analog_data = {
                "rank": i + 1,
                "smiles": Chem.MolToSmiles(analog_mol),
                "score": round(score, 3),
                "properties": calculate_properties(analog_mol),
                "improvements": improvements,
                "modifications": identify_modifications(lead_mol, analog_mol),
                "similarity": round(calculate_similarity(lead_mol, analog_mol), 3)
            }
            results["optimized_molecules"].append(analog_data)
        
        # Add optimization summary
        results["summary"] = generate_optimization_summary(results["optimized_molecules"], optimization_goals)
        
        return results
        
    except Exception as e:
        return {"error": f"Lead optimization failed: {str(e)}"}


async def generate_analogs(
    reference_smiles: str,
    num_analogs: int = 10,
    similarity_threshold: float = 0.7
) -> Dict:
    """Generate analog molecules with similar properties.
    
    Args:
        reference_smiles: SMILES of reference molecule
        num_analogs: Number of analogs to generate
        similarity_threshold: Minimum Tanimoto similarity (0-1)
        
    Returns:
        Dictionary with generated analogs
    """
    try:
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if ref_mol is None:
            return {"error": "Invalid reference SMILES"}
            
        results = {
            "reference": reference_smiles,
            "reference_properties": calculate_properties(ref_mol),
            "analogs": [],
            "scaffold": get_scaffold_smiles(ref_mol)
        }
        
        # Generate analogs using multiple strategies
        analogs = []
        
        # 1. Bioisosteric replacements
        analogs.extend(apply_bioisosteric_replacements(ref_mol))
        
        # 2. Functional group variations
        analogs.extend(vary_functional_groups(ref_mol))
        
        # 3. Ring modifications
        analogs.extend(modify_rings(ref_mol))
        
        # 4. Chain length variations
        analogs.extend(vary_chain_lengths(ref_mol))
        
        # 5. Stereochemical variations
        analogs.extend(generate_stereoisomers(ref_mol))
        
        # Filter by similarity
        similar_analogs = []
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
        
        for analog in analogs:
            if analog is None:
                continue
            analog_fp = AllChem.GetMorganFingerprintAsBitVect(analog, 2)
            similarity = TanimotoSimilarity(ref_fp, analog_fp)
            
            if similarity >= similarity_threshold:
                similar_analogs.append((analog, similarity))
        
        # Sort by similarity and diversity
        diverse_analogs = select_diverse_subset(similar_analogs, num_analogs)
        
        # Format results
        for i, (analog_mol, similarity) in enumerate(diverse_analogs):
            analog_data = {
                "rank": i + 1,
                "smiles": Chem.MolToSmiles(analog_mol),
                "similarity": round(similarity, 3),
                "properties": calculate_properties(analog_mol),
                "modifications": identify_modifications(ref_mol, analog_mol)
            }
            results["analogs"].append(analog_data)
        
        return results
        
    except Exception as e:
        return {"error": f"Analog generation failed: {str(e)}"}


def determine_optimization_strategy(mol: Chem.Mol, goals: Dict[str, float]) -> Dict:
    """Determine optimization strategy based on current properties and goals."""
    current_props = calculate_properties(mol)
    strategy = {
        "primary_objectives": [],
        "suggested_modifications": []
    }
    
    # Analyze each goal
    for prop, target in goals.items():
        current_val = current_props.get(prop)
        if current_val is None:
            continue
            
        # Handle range targets
        if isinstance(target, list):
            if current_val < target[0]:
                strategy["primary_objectives"].append(f"Increase {prop}")
                strategy["suggested_modifications"].extend(
                    get_modifications_to_increase(prop)
                )
            elif current_val > target[1]:
                strategy["primary_objectives"].append(f"Decrease {prop}")
                strategy["suggested_modifications"].extend(
                    get_modifications_to_decrease(prop)
                )
        else:
            # Single target value
            if abs(current_val - target) > 0.1:
                direction = "Increase" if current_val < target else "Decrease"
                strategy["primary_objectives"].append(f"{direction} {prop}")
    
    return strategy


def get_modifications_to_increase(prop: str) -> List[str]:
    """Get suggested modifications to increase a property."""
    modifications = {
        "logP": ["Add lipophilic groups", "Remove polar groups", "Add aromatic rings"],
        "MW": ["Add substituents", "Extend chains", "Add rings"],
        "HBA": ["Add ethers", "Add esters", "Add tertiary amines"],
        "HBD": ["Add alcohols", "Add amines", "Add amides"],
        "TPSA": ["Add polar groups", "Add heteroatoms", "Add H-bond donors/acceptors"]
    }
    return modifications.get(prop, ["Modify structure"])


def get_modifications_to_decrease(prop: str) -> List[str]:
    """Get suggested modifications to decrease a property."""
    modifications = {
        "logP": ["Add polar groups", "Remove lipophilic groups", "Add charged groups"],
        "MW": ["Remove substituents", "Shorten chains", "Simplify structure"],
        "HBA": ["Remove ethers", "Remove esters", "Replace with carbons"],
        "HBD": ["Remove alcohols", "Protect amines", "Remove NH groups"],
        "TPSA": ["Remove polar groups", "Replace heteroatoms", "Add lipophilic groups"]
    }
    return modifications.get(prop, ["Modify structure"])


def generate_scaffold_based_analogs(mol: Chem.Mol, goals: Dict[str, float]) -> List[Chem.Mol]:
    """Generate analogs maintaining core scaffold."""
    analogs = []
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    # Identify substitution points
    sub_points = identify_substitution_points(mol, scaffold)
    
    # Generate substitutions
    for point in sub_points:
        # Try different substituents based on optimization goals
        substituents = select_substituents_for_goals(goals)
        
        for sub in substituents:
            analog = make_substitution(mol, point, sub)
            if analog and is_valid_molecule(analog):
                analogs.append(analog)
    
    # Also try combinations of substitutions
    if len(sub_points) > 1:
        for i in range(min(len(sub_points), 3)):
            for j in range(i+1, min(len(sub_points), 3)):
                analog = make_double_substitution(mol, sub_points[i], sub_points[j], goals)
                if analog and is_valid_molecule(analog):
                    analogs.append(analog)
    
    return analogs


def generate_free_optimization(mol: Chem.Mol, goals: Dict[str, float]) -> List[Chem.Mol]:
    """Generate optimized molecules without scaffold constraint."""
    analogs = []
    
    # More aggressive modifications allowed
    strategies = [
        apply_bioisosteric_replacements,
        vary_functional_groups,
        modify_rings,
        vary_chain_lengths,
        fuse_rings,
        break_rings,
        add_heterocycles
    ]
    
    for strategy in strategies:
        analogs.extend(strategy(mol))
    
    # Filter based on goals
    filtered_analogs = []
    for analog in analogs:
        if analog and is_moving_toward_goals(analog, mol, goals):
            filtered_analogs.append(analog)
    
    return filtered_analogs


def apply_bioisosteric_replacements(mol: Chem.Mol) -> List[Chem.Mol]:
    """Apply bioisosteric replacements to molecule."""
    analogs = []
    
    # Common bioisosteric replacements
    replacements = {
        "C(=O)OH": ["C(=O)NHOH", "S(=O)(=O)OH", "P(=O)(OH)OH", "[nH]1nnnc1"],  # Carboxylic acid
        "C(=O)N": ["C(=S)N", "S(=O)(=O)N"],  # Amide
        "c1ccccc1": ["c1ccncc1", "c1cncnc1", "c1ccsc1"],  # Benzene
        "C=O": ["C=S", "C=N"],  # Carbonyl
        "O": ["S", "NH"],  # Ether
        "Cl": ["F", "CF3", "CN"],  # Chloro
        "C": ["N", "O", "S"]  # Methylene in rings
    }
    
    for smarts, replacements_list in replacements.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            for replacement in replacements_list:
                analog = replace_substructure(mol, smarts, replacement)
                if analog:
                    analogs.append(analog)
    
    return analogs


def vary_functional_groups(mol: Chem.Mol) -> List[Chem.Mol]:
    """Generate variations of functional groups."""
    analogs = []
    
    # Add/remove/modify common functional groups
    fg_variations = {
        "add": ["OH", "NH2", "OCH3", "N(CH3)2", "F", "Cl", "CN", "CF3", "NO2"],
        "remove": ["OH", "NH2", "OCH3", "F", "Cl"],
        "modify": {
            "OH": ["OCH3", "OC(=O)CH3"],
            "NH2": ["NHCH3", "N(CH3)2", "NHC(=O)CH3"],
            "OCH3": ["OH", "OC2H5", "OCF3"],
            "Cl": ["F", "Br", "I", "CF3"]
        }
    }
    
    # Try additions at available positions
    for fg in fg_variations["add"]:
        positions = find_substitutable_positions(mol)
        for pos in positions[:3]:  # Limit to avoid explosion
            analog = add_functional_group(mol, pos, fg)
            if analog:
                analogs.append(analog)
    
    # Try modifications of existing groups
    for fg_from, fg_to_list in fg_variations["modify"].items():
        pattern = Chem.MolFromSmarts(fg_from)
        if pattern and mol.HasSubstructMatch(pattern):
            for fg_to in fg_to_list:
                analog = replace_functional_group(mol, fg_from, fg_to)
                if analog:
                    analogs.append(analog)
    
    return analogs


def modify_rings(mol: Chem.Mol) -> List[Chem.Mol]:
    """Modify ring systems in molecule."""
    analogs = []
    
    ring_info = mol.GetRingInfo()
    
    # Ring expansion/contraction
    for ring in ring_info.AtomRings():
        if 5 <= len(ring) <= 7:
            # Try expansion
            expanded = expand_ring(mol, ring)
            if expanded:
                analogs.append(expanded)
            
            # Try contraction
            if len(ring) > 5:
                contracted = contract_ring(mol, ring)
                if contracted:
                    analogs.append(contracted)
    
    # Heteroatom substitution in rings
    for ring in ring_info.AtomRings():
        for i, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 6:  # Carbon
                # Try N, O, S substitution
                for heteroatom in ["N", "O", "S"]:
                    analog = substitute_ring_atom(mol, atom_idx, heteroatom)
                    if analog:
                        analogs.append(analog)
    
    return analogs


def vary_chain_lengths(mol: Chem.Mol) -> List[Chem.Mol]:
    """Vary alkyl chain lengths."""
    analogs = []
    
    # Find alkyl chains
    alkyl_patterns = ["CC", "CCC", "CCCC", "C(C)C"]
    
    for pattern in alkyl_patterns:
        smarts = Chem.MolFromSmarts(pattern)
        if smarts and mol.HasSubstructMatch(smarts):
            # Try shorter and longer versions
            if len(pattern) > 2:
                shorter = pattern[:-1]
                analog = replace_substructure(mol, pattern, shorter)
                if analog:
                    analogs.append(analog)
            
            longer = pattern + "C"
            analog = replace_substructure(mol, pattern, longer)
            if analog:
                analogs.append(analog)
    
    return analogs


def generate_stereoisomers(mol: Chem.Mol) -> List[Chem.Mol]:
    """Generate stereoisomers of molecule."""
    from rdkit.Chem import EnumerateStereoisomers
    
    opts = EnumerateStereoisomers.StereoEnumerationOptions()
    opts.maxIsomers = 20
    
    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol, opts))
    
    # Remove the original if present
    original_smiles = Chem.MolToSmiles(mol)
    return [iso for iso in isomers if Chem.MolToSmiles(iso) != original_smiles]


def score_optimized_molecules(analogs: List[Chem.Mol], goals: Dict[str, float], 
                            lead_mol: Chem.Mol) -> List[Tuple[Chem.Mol, float, Dict]]:
    """Score and rank optimized molecules."""
    scored = []
    
    lead_props = calculate_properties(lead_mol)
    
    for analog in analogs:
        if analog is None:
            continue
            
        analog_props = calculate_properties(analog)
        score = 0
        improvements = {}
        
        # Score based on how well goals are met
        for prop, target in goals.items():
            if prop not in analog_props:
                continue
                
            current = analog_props[prop]
            lead_val = lead_props.get(prop, current)
            
            # Handle range targets
            if isinstance(target, list):
                if target[0] <= current <= target[1]:
                    score += 1.0
                    improvements[prop] = "Within target range"
                else:
                    # Partial score based on improvement
                    if current < target[0]:
                        distance = target[0] - current
                        lead_distance = target[0] - lead_val
                        if lead_distance > 0 and distance < lead_distance:
                            score += 0.5 * (1 - distance/lead_distance)
                            improvements[prop] = f"Improved ({round(current, 2)})"
                    else:
                        distance = current - target[1]
                        lead_distance = lead_val - target[1]
                        if lead_distance > 0 and distance < lead_distance:
                            score += 0.5 * (1 - distance/lead_distance)
                            improvements[prop] = f"Improved ({round(current, 2)})"
            else:
                # Single target
                distance = abs(current - target)
                lead_distance = abs(lead_val - target)
                
                if distance < lead_distance:
                    score += 1 - distance/max(lead_distance, 1)
                    improvements[prop] = f"Closer to target ({round(current, 2)})"
        
        # Bonus for maintaining drug-likeness
        if passes_drug_likeness_filters(analog):
            score += 0.5
            improvements["drug_likeness"] = "Maintained"
        
        # Penalty for too low similarity
        similarity = calculate_similarity(lead_mol, analog)
        if similarity < 0.4:
            score *= 0.5
        
        scored.append((analog, score, improvements))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored


def select_diverse_subset(molecules: List[Tuple[Chem.Mol, float]], 
                         num_molecules: int) -> List[Tuple[Chem.Mol, float]]:
    """Select diverse subset of molecules."""
    if len(molecules) <= num_molecules:
        return molecules
    
    # Start with highest similarity molecule
    selected = [molecules[0]]
    remaining = molecules[1:]
    
    # Iteratively add most diverse molecules
    while len(selected) < num_molecules and remaining:
        max_min_distance = -1
        most_diverse_idx = -1
        
        for i, (mol, sim) in enumerate(remaining):
            # Calculate minimum distance to selected molecules
            min_distance = min(
                1 - calculate_similarity(mol, selected_mol[0])
                for selected_mol in selected
            )
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                most_diverse_idx = i
        
        if most_diverse_idx >= 0:
            selected.append(remaining.pop(most_diverse_idx))
    
    return selected


def calculate_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate molecular properties."""
    if mol is None:
        return {}
        
    props = {
        "MW": round(Descriptors.MolWt(mol), 2),
        "logP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "QED": round(Descriptors.qed(mol), 3),
        "SA_score": round(calculate_sa_score(mol), 2)
    }
    
    return props


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Calculate synthetic accessibility score."""
    # Simplified SA score
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = Lipinski.RingCount(mol)
    num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    score = 1 + num_atoms * 0.05 + num_rings * 0.5 + num_stereo * 0.5
    
    return min(score, 10)


def calculate_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Calculate Tanimoto similarity between molecules."""
    if mol1 is None or mol2 is None:
        return 0.0
        
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    
    return TanimotoSimilarity(fp1, fp2)


def get_scaffold_smiles(mol: Chem.Mol) -> str:
    """Get Murcko scaffold SMILES."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def identify_modifications(ref_mol: Chem.Mol, analog_mol: Chem.Mol) -> List[str]:
    """Identify modifications between reference and analog."""
    modifications = []
    
    # Compare molecular formulas
    ref_formula = rdMolDescriptors.CalcMolFormula(ref_mol)
    analog_formula = rdMolDescriptors.CalcMolFormula(analog_mol)
    
    if ref_formula != analog_formula:
        modifications.append(f"Formula: {ref_formula} → {analog_formula}")
    
    # Compare key properties
    ref_props = calculate_properties(ref_mol)
    analog_props = calculate_properties(analog_mol)
    
    for prop in ["MW", "logP", "TPSA"]:
        if abs(ref_props[prop] - analog_props[prop]) > 0.1:
            modifications.append(f"{prop}: {ref_props[prop]} → {analog_props[prop]}")
    
    # Try to identify specific changes
    mcs = rdFMCS.FindMCS([ref_mol, analog_mol])
    if mcs.numAtoms < ref_mol.GetNumAtoms():
        modifications.append("Structural modification detected")
    
    return modifications


def generate_optimization_summary(optimized_mols: List[Dict], goals: Dict[str, float]) -> Dict:
    """Generate summary of optimization results."""
    if not optimized_mols:
        return {"success": False, "message": "No optimized molecules generated"}
    
    summary = {
        "total_analogs": len(optimized_mols),
        "goals_achieved": 0,
        "best_overall": optimized_mols[0]["smiles"] if optimized_mols else None,
        "property_improvements": {}
    }
    
    # Check how many molecules achieve all goals
    for mol_data in optimized_mols:
        props = mol_data["properties"]
        achieves_all = True
        
        for prop, target in goals.items():
            if prop not in props:
                continue
                
            if isinstance(target, list):
                if not (target[0] <= props[prop] <= target[1]):
                    achieves_all = False
                    break
            else:
                if abs(props[prop] - target) > 0.1:
                    achieves_all = False
                    break
        
        if achieves_all:
            summary["goals_achieved"] += 1
    
    # Analyze property distributions
    for prop in goals:
        values = [mol["properties"].get(prop) for mol in optimized_mols if prop in mol["properties"]]
        if values:
            summary["property_improvements"][prop] = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "mean": round(np.mean(values), 2)
            }
    
    summary["success"] = summary["goals_achieved"] > 0
    
    return summary


# Helper functions for molecular modifications

def identify_substitution_points(mol: Chem.Mol, scaffold: Chem.Mol) -> List[int]:
    """Identify substitution points on molecule."""
    # Find atoms in molecule but not in scaffold
    scaffold_match = mol.GetSubstructMatch(scaffold)
    all_atoms = set(range(mol.GetNumAtoms()))
    scaffold_atoms = set(scaffold_match)
    
    substitution_points = []
    
    # Atoms connected to scaffold
    for atom_idx in all_atoms - scaffold_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in scaffold_atoms:
                substitution_points.append(neighbor.GetIdx())
                break
    
    return list(set(substitution_points))


def select_substituents_for_goals(goals: Dict[str, float]) -> List[str]:
    """Select appropriate substituents based on optimization goals."""
    substituents = []
    
    # Basic substituent library
    all_substituents = {
        "increase_logP": ["C", "CC", "CCC", "c1ccccc1", "C(C)C", "C(C)(C)C"],
        "decrease_logP": ["O", "OH", "N", "NH2", "C(=O)O", "S(=O)(=O)O"],
        "increase_MW": ["c1ccccc1", "C(=O)N", "S(=O)(=O)N", "C(F)(F)F"],
        "decrease_MW": ["H", "F"],
        "increase_HBA": ["O", "OC", "N", "NC", "C(=O)O"],
        "increase_HBD": ["OH", "NH2", "NH", "C(=O)N"],
        "increase_flexibility": ["CC", "CCC", "OC", "NC"],
        "increase_rigidity": ["c1ccccc1", "C1CC1", "C1CCC1"]
    }
    
    # Select based on goals
    for prop, target in goals.items():
        if prop == "logP":
            if isinstance(target, list):
                # Assume we need to increase if below range
                if target[0] > 0:
                    substituents.extend(all_substituents["increase_logP"])
                else:
                    substituents.extend(all_substituents["decrease_logP"])
            elif target > 2:
                substituents.extend(all_substituents["increase_logP"])
            else:
                substituents.extend(all_substituents["decrease_logP"])
    
    # Add some general substituents
    substituents.extend(["F", "Cl", "CN", "OCH3", "N(C)C"])
    
    return list(set(substituents))


def make_substitution(mol: Chem.Mol, position: int, substituent: str) -> Optional[Chem.Mol]:
    """Make substitution at specified position."""
    try:
        # Create editable molecule
        emol = Chem.EditableMol(mol)
        
        # Add substituent
        sub_mol = Chem.MolFromSmiles(substituent)
        if sub_mol is None:
            return None
            
        # Find hydrogen to replace
        atom = mol.GetAtomWithIdx(position)
        h_idx = None
        
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                h_idx = neighbor.GetIdx()
                break
        
        if h_idx is not None:
            # Remove hydrogen
            emol.RemoveBond(position, h_idx)
            emol.RemoveAtom(h_idx)
            
            # Add substituent (simplified - would need proper attachment)
            # This is a placeholder - real implementation would be more complex
            
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        
        return new_mol
    except:
        return None


def make_double_substitution(mol: Chem.Mol, pos1: int, pos2: int, goals: Dict[str, float]) -> Optional[Chem.Mol]:
    """Make substitutions at two positions."""
    substituents = select_substituents_for_goals(goals)
    if len(substituents) < 2:
        return None
        
    # Try first substitution
    mol1 = make_substitution(mol, pos1, substituents[0])
    if mol1 is None:
        return None
        
    # Try second substitution
    mol2 = make_substitution(mol1, pos2, substituents[1])
    
    return mol2


def is_valid_molecule(mol: Chem.Mol) -> bool:
    """Check if molecule is valid."""
    if mol is None:
        return False
        
    try:
        Chem.SanitizeMol(mol)
        
        # Basic validity checks
        if mol.GetNumAtoms() == 0:
            return False
            
        # Check for disconnected structures
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
            
        return True
    except:
        return False


def is_moving_toward_goals(analog: Chem.Mol, original: Chem.Mol, goals: Dict[str, float]) -> bool:
    """Check if analog is moving toward optimization goals."""
    original_props = calculate_properties(original)
    analog_props = calculate_properties(analog)
    
    improvements = 0
    total_goals = 0
    
    for prop, target in goals.items():
        if prop not in original_props or prop not in analog_props:
            continue
            
        total_goals += 1
        orig_val = original_props[prop]
        analog_val = analog_props[prop]
        
        if isinstance(target, list):
            # Range target
            orig_dist = min(abs(orig_val - target[0]), abs(orig_val - target[1]))
            analog_dist = min(abs(analog_val - target[0]), abs(analog_val - target[1]))
            
            if target[0] <= analog_val <= target[1]:
                improvements += 1
            elif analog_dist < orig_dist:
                improvements += 0.5
        else:
            # Single target
            if abs(analog_val - target) < abs(orig_val - target):
                improvements += 1
    
    return improvements > 0


def replace_substructure(mol: Chem.Mol, pattern: str, replacement: str) -> Optional[Chem.Mol]:
    """Replace substructure in molecule."""
    try:
        rxn_smarts = f"[{pattern}]>>[{replacement}]"
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        products = rxn.RunReactants([mol])
        
        if products:
            return products[0][0]
        return None
    except:
        return None


def find_substitutable_positions(mol: Chem.Mol) -> List[int]:
    """Find positions suitable for substitution."""
    positions = []
    
    for atom in mol.GetAtoms():
        # Look for carbons with hydrogens
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            # Prefer non-aromatic positions
            if not atom.GetIsAromatic():
                positions.append(atom.GetIdx())
            elif len(positions) < 3:  # Include aromatic if needed
                positions.append(atom.GetIdx())
    
    return positions


def add_functional_group(mol: Chem.Mol, position: int, fg: str) -> Optional[Chem.Mol]:
    """Add functional group at position."""
    return make_substitution(mol, position, fg)


def replace_functional_group(mol: Chem.Mol, old_fg: str, new_fg: str) -> Optional[Chem.Mol]:
    """Replace functional group."""
    return replace_substructure(mol, old_fg, new_fg)


def expand_ring(mol: Chem.Mol, ring: Tuple) -> Optional[Chem.Mol]:
    """Expand ring by one atom."""
    # Placeholder - real implementation would be complex
    return None


def contract_ring(mol: Chem.Mol, ring: Tuple) -> Optional[Chem.Mol]:
    """Contract ring by one atom."""
    # Placeholder - real implementation would be complex
    return None


def substitute_ring_atom(mol: Chem.Mol, atom_idx: int, new_atom: str) -> Optional[Chem.Mol]:
    """Substitute atom in ring."""
    try:
        emol = Chem.EditableMol(mol)
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # This is simplified - real implementation would handle valence, aromaticity, etc.
        new_atom_num = {"N": 7, "O": 8, "S": 16}.get(new_atom)
        if new_atom_num:
            atom.SetAtomicNum(new_atom_num)
            
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        
        return new_mol
    except:
        return None


def fuse_rings(mol: Chem.Mol) -> List[Chem.Mol]:
    """Generate ring-fused analogs."""
    # Placeholder - would implement ring fusion logic
    return []


def break_rings(mol: Chem.Mol) -> List[Chem.Mol]:
    """Generate ring-opened analogs."""
    # Placeholder - would implement ring opening logic
    return []


def add_heterocycles(mol: Chem.Mol) -> List[Chem.Mol]:
    """Add heterocyclic rings."""
    # Placeholder - would implement heterocycle addition
    return []


def passes_drug_likeness_filters(mol: Chem.Mol) -> bool:
    """Check if molecule passes basic drug-likeness filters."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    # Lipinski's Rule of Five
    if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
        return False
        
    # Additional filters
    rotatable = Lipinski.NumRotatableBonds(mol)
    if rotatable > 10:
        return False
        
    tpsa = Descriptors.TPSA(mol)
    if tpsa > 140:
        return False
        
    return True