"""Synthesis planning tools for MCP."""

from typing import Dict, List, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdFMCS
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


async def retrosynthetic_analysis(
    target_smiles: str,
    max_steps: int = 5,
    starting_materials: Optional[List[str]] = None
) -> Dict:
    """Perform retrosynthetic analysis.
    
    Args:
        target_smiles: SMILES of target molecule
        max_steps: Maximum number of retrosynthetic steps
        starting_materials: Optional list of preferred starting materials
        
    Returns:
        Dictionary with synthetic routes
    """
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return {"error": "Invalid target SMILES"}
            
        results = {
            "target": target_smiles,
            "routes": [],
            "analysis": analyze_target_molecule(target_mol)
        }
        
        # Perform retrosynthetic analysis
        routes = analyze_retrosynthesis(target_mol, max_steps)
        
        # Score and rank routes
        scored_routes = score_synthetic_routes(routes, starting_materials)
        
        # Format results
        for i, (route, score) in enumerate(scored_routes[:5]):  # Top 5 routes
            formatted_route = {
                "rank": i + 1,
                "score": round(score, 3),
                "steps": format_route(route),
                "key_transformations": identify_key_transformations(route),
                "estimated_yield": estimate_overall_yield(route),
                "complexity": assess_route_complexity(route)
            }
            results["routes"].append(formatted_route)
            
        # Add strategic bonds analysis
        results["strategic_bonds"] = identify_strategic_bonds(target_mol)
        
        return results
        
    except Exception as e:
        return {"error": f"Retrosynthesis failed: {str(e)}"}


def analyze_target_molecule(mol: Chem.Mol) -> Dict:
    """Analyze target molecule for retrosynthetic planning."""
    analysis = {
        "molecular_weight": round(rdMolDescriptors.CalcExactMolWt(mol), 2),
        "complexity": calculate_molecular_complexity(mol),
        "functional_groups": identify_functional_groups(mol),
        "rings": analyze_ring_systems(mol),
        "stereocenters": count_stereocenters(mol),
        "synthetic_accessibility": calculate_sa_score_simple(mol)
    }
    
    return analysis


def analyze_retrosynthesis(target: Chem.Mol, max_steps: int) -> List[List[Dict]]:
    """Perform retrosynthetic analysis using reaction templates."""
    routes = []
    
    # Get available reaction templates
    templates = get_reaction_templates()
    
    # BFS for retrosynthetic routes
    queue = [(target, [], 0)]
    visited = set()
    
    while queue and len(routes) < 10:  # Limit to 10 routes
        current_mol, path, depth = queue.pop(0)
        
        if depth >= max_steps:
            continue
            
        mol_smiles = Chem.MolToSmiles(current_mol)
        if mol_smiles in visited:
            continue
        visited.add(mol_smiles)
        
        # Try each reaction template
        for template_name, template in templates.items():
            precursors = apply_retro_template(current_mol, template)
            
            for precursor_set in precursors:
                # Check if we've reached simple starting materials
                if all(is_simple_starting_material(p) for p in precursor_set):
                    new_path = path + [{
                        "product": mol_smiles,
                        "precursors": [Chem.MolToSmiles(p) for p in precursor_set],
                        "reaction": template_name,
                        "step": depth + 1
                    }]
                    routes.append(new_path)
                else:
                    # Continue retrosynthesis on complex precursors
                    for precursor in precursor_set:
                        if not is_simple_starting_material(precursor):
                            new_path = path + [{
                                "product": mol_smiles,
                                "precursors": [Chem.MolToSmiles(p) for p in precursor_set],
                                "reaction": template_name,
                                "step": depth + 1
                            }]
                            queue.append((precursor, new_path, depth + 1))
    
    return routes


def get_reaction_templates() -> Dict[str, str]:
    """Get common reaction templates for retrosynthesis."""
    templates = {
        # C-C bond formations
        "aldol": "[C:1]-[C:2](=[O:3])-[C:4]-[C:5]>>[C:1]-[C:2](=[O:3]).[C:4]-[C:5]=O",
        "michael": "[C:1]-[C:2]-[C:3](=[O:4])-[C:5]>>[C:1]=[C:2]-[C:3](=[O:4]).[C:5]",
        "grignard": "[C:1]-[C:2]-[OH:3]>>[C:1]-Br.[C:2]=O",
        "wittig": "[C:1]=[C:2]>>[C:1]=O.[C:2]-P(Ph)3",
        "suzuki": "[c:1]-[c:2]>>[c:1]-B(OH)2.[c:2]-Br",
        "heck": "[C:1]=[C:2]-[c:3]>>[C:1]=[C:2].[c:3]-I",
        
        # Functional group interconversions
        "esterification": "[C:1](=[O:2])-[O:3]-[C:4]>>[C:1](=[O:2])-OH.[HO]-[C:4]",
        "amidation": "[C:1](=[O:2])-[N:3]>>[C:1](=[O:2])-OH.[N:3]",
        "reduction_carbonyl": "[C:1]-[OH:2]>>[C:1]=[O:2]",
        "oxidation_alcohol": "[C:1]=[O:2]>>[C:1]-[OH:2]",
        
        # Aromatic substitutions
        "friedel_crafts_acyl": "[c:1]-[C:2](=[O:3])>>[c:1].[C:2](=[O:3])-Cl",
        "friedel_crafts_alkyl": "[c:1]-[C:2]>>[c:1].[C:2]-Cl",
        "nitration": "[c:1]-[N+](=[O])([O-])>>[c:1]",
        "halogenation": "[c:1]-[Cl,Br,I:2]>>[c:1]",
        
        # Protection/Deprotection
        "tbdms_protection": "[C:1]-[O:2]-Si(C)(C)C(C)(C)C>>[C:1]-[OH:2]",
        "boc_protection": "[N:1]-C(=O)OC(C)(C)C>>[N:1]",
        "benzyl_protection": "[O:1]-Cc1ccccc1>>[OH:1]"
    }
    
    return templates


def apply_retro_template(mol: Chem.Mol, template: str) -> List[List[Chem.Mol]]:
    """Apply a retrosynthetic template to a molecule."""
    precursor_sets = []
    
    try:
        # Parse the retrosynthetic template
        rxn = AllChem.ReactionFromSmarts(template)
        
        # Apply the template
        products = rxn.RunReactants([mol])
        
        for product_set in products:
            valid_products = []
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    valid_products.append(product)
                except:
                    pass
            
            if valid_products:
                precursor_sets.append(valid_products)
                
    except Exception as e:
        logger.debug(f"Template application failed: {e}")
    
    return precursor_sets


def is_simple_starting_material(mol: Chem.Mol) -> bool:
    """Check if molecule is a simple starting material."""
    if mol is None:
        return False
        
    # Common starting materials (simplified list)
    common_sm_smarts = [
        "C(=O)O",      # Carboxylic acid
        "C=O",         # Aldehyde/Ketone
        "CO",          # Alcohol/Ether
        "CN",          # Amine
        "C=C",         # Alkene
        "c1ccccc1",    # Benzene
        "CCl",         # Alkyl halide
        "CBr",         # Alkyl bromide
        "CI",          # Alkyl iodide
        "CS",          # Thiol/Sulfide
        "CC",          # Simple alkane
        "C#C",         # Alkyne
        "C#N",         # Nitrile
    ]
    
    # Check molecular weight
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    if mw > 200:
        return False
    
    # Check if it matches common starting material patterns
    mol_smiles = Chem.MolToSmiles(mol)
    
    # Very simple molecules
    if len(mol_smiles) < 10 and mol.GetNumHeavyAtoms() < 8:
        return True
    
    # Check against common patterns
    for pattern in common_sm_smarts:
        if Chem.MolFromSmarts(pattern).HasSubstructMatch(mol):
            return True
    
    return False


def score_synthetic_routes(routes: List[List[Dict]], 
                          starting_materials: Optional[List[str]] = None) -> List[Tuple[List[Dict], float]]:
    """Score and rank synthetic routes."""
    scored_routes = []
    
    for route in routes:
        score = 1.0
        
        # Factors affecting score:
        # 1. Number of steps (fewer is better)
        num_steps = len(route)
        score *= (1.0 / (1 + num_steps * 0.2))
        
        # 2. Reaction reliability
        for step in route:
            reaction_score = get_reaction_reliability(step["reaction"])
            score *= reaction_score
        
        # 3. Starting material availability
        if starting_materials:
            sm_bonus = check_starting_material_match(route, starting_materials)
            score *= (1 + sm_bonus)
        
        # 4. Complexity of intermediates
        complexity_penalty = assess_intermediate_complexity(route)
        score *= (1 - complexity_penalty)
        
        # 5. Protecting group usage
        pg_penalty = count_protecting_groups(route) * 0.05
        score *= (1 - pg_penalty)
        
        scored_routes.append((route, score))
    
    # Sort by score (descending)
    scored_routes.sort(key=lambda x: x[1], reverse=True)
    
    return scored_routes


def get_reaction_reliability(reaction_name: str) -> float:
    """Get reliability score for a reaction type."""
    reliability_scores = {
        "aldol": 0.8,
        "michael": 0.85,
        "grignard": 0.9,
        "wittig": 0.85,
        "suzuki": 0.95,
        "heck": 0.85,
        "esterification": 0.95,
        "amidation": 0.9,
        "reduction_carbonyl": 0.95,
        "oxidation_alcohol": 0.9,
        "friedel_crafts_acyl": 0.8,
        "friedel_crafts_alkyl": 0.75,
        "nitration": 0.85,
        "halogenation": 0.9,
        "tbdms_protection": 0.95,
        "boc_protection": 0.95,
        "benzyl_protection": 0.9
    }
    
    return reliability_scores.get(reaction_name, 0.7)


def check_starting_material_match(route: List[Dict], starting_materials: List[str]) -> float:
    """Check if route uses preferred starting materials."""
    if not route:
        return 0
    
    # Get all starting materials from the route
    route_sm = set()
    for step in route:
        if "precursors" in step:
            for precursor in step["precursors"]:
                if is_simple_starting_material(Chem.MolFromSmiles(precursor)):
                    route_sm.add(precursor)
    
    # Check matches
    matches = 0
    for sm in starting_materials:
        if sm in route_sm:
            matches += 1
    
    return matches * 0.1  # 10% bonus per match


def assess_intermediate_complexity(route: List[Dict]) -> float:
    """Assess complexity of intermediates in the route."""
    if not route:
        return 0
    
    total_complexity = 0
    for step in route:
        if "product" in step:
            mol = Chem.MolFromSmiles(step["product"])
            if mol:
                complexity = calculate_molecular_complexity(mol)
                total_complexity += complexity
    
    # Normalize
    avg_complexity = total_complexity / len(route)
    return min(avg_complexity / 100, 0.5)  # Cap at 50% penalty


def count_protecting_groups(route: List[Dict]) -> int:
    """Count protecting group operations in route."""
    pg_count = 0
    pg_reactions = ["tbdms_protection", "boc_protection", "benzyl_protection"]
    
    for step in route:
        if step.get("reaction") in pg_reactions:
            pg_count += 1
    
    return pg_count


def format_route(route: List[Dict]) -> List[Dict]:
    """Format synthetic route for output."""
    formatted_steps = []
    
    for i, step in enumerate(route):
        formatted_step = {
            "step_number": i + 1,
            "reaction_type": step["reaction"],
            "product": step["product"],
            "precursors": step["precursors"],
            "conditions": suggest_reaction_conditions(step["reaction"]),
            "expected_yield": estimate_reaction_yield(step["reaction"])
        }
        formatted_steps.append(formatted_step)
    
    return formatted_steps


def suggest_reaction_conditions(reaction_name: str) -> Dict[str, str]:
    """Suggest reaction conditions for a given reaction type."""
    conditions = {
        "aldol": {
            "catalyst": "LDA or NaOH",
            "solvent": "THF or EtOH",
            "temperature": "-78°C to RT",
            "time": "2-4 hours"
        },
        "michael": {
            "catalyst": "DBU or KOtBu",
            "solvent": "THF or DCM",
            "temperature": "0°C to RT",
            "time": "1-3 hours"
        },
        "grignard": {
            "catalyst": "Mg turnings",
            "solvent": "THF or Et2O",
            "temperature": "0°C to reflux",
            "time": "1-2 hours",
            "notes": "Dry conditions essential"
        },
        "suzuki": {
            "catalyst": "Pd(PPh3)4 or Pd(OAc)2",
            "base": "K2CO3 or Na2CO3",
            "solvent": "DMF/H2O or Dioxane/H2O",
            "temperature": "80-100°C",
            "time": "4-12 hours"
        },
        "esterification": {
            "catalyst": "H2SO4 or DCC/DMAP",
            "solvent": "MeOH or DCM",
            "temperature": "RT to reflux",
            "time": "2-8 hours"
        }
    }
    
    return conditions.get(reaction_name, {
        "catalyst": "Standard conditions",
        "solvent": "Appropriate solvent",
        "temperature": "RT",
        "time": "2-24 hours"
    })


def estimate_reaction_yield(reaction_name: str) -> int:
    """Estimate typical yield for a reaction type."""
    typical_yields = {
        "aldol": 70,
        "michael": 80,
        "grignard": 85,
        "wittig": 75,
        "suzuki": 90,
        "heck": 80,
        "esterification": 90,
        "amidation": 85,
        "reduction_carbonyl": 95,
        "oxidation_alcohol": 90,
        "friedel_crafts_acyl": 75,
        "friedel_crafts_alkyl": 70,
        "nitration": 85,
        "halogenation": 90,
        "tbdms_protection": 95,
        "boc_protection": 95,
        "benzyl_protection": 90
    }
    
    return typical_yields.get(reaction_name, 75)


def estimate_overall_yield(route: List[Dict]) -> Dict[str, float]:
    """Estimate overall yield for a synthetic route."""
    if not route:
        return {"percent": 0, "range": "0%"}
    
    overall_yield = 1.0
    for step in route:
        step_yield = estimate_reaction_yield(step.get("reaction", "")) / 100
        overall_yield *= step_yield
    
    overall_percent = overall_yield * 100
    
    # Give a range based on uncertainty
    lower = overall_percent * 0.7
    upper = min(overall_percent * 1.2, 95)
    
    return {
        "percent": round(overall_percent, 1),
        "range": f"{round(lower, 1)}-{round(upper, 1)}%"
    }


def assess_route_complexity(route: List[Dict]) -> str:
    """Assess overall complexity of synthetic route."""
    if not route:
        return "Unknown"
    
    num_steps = len(route)
    pg_count = count_protecting_groups(route)
    
    # Check for difficult reactions
    difficult_reactions = ["friedel_crafts_alkyl", "aldol", "michael"]
    difficult_count = sum(1 for step in route if step.get("reaction") in difficult_reactions)
    
    # Scoring
    complexity_score = num_steps + pg_count * 0.5 + difficult_count * 0.5
    
    if complexity_score <= 3:
        return "Simple"
    elif complexity_score <= 6:
        return "Moderate"
    else:
        return "Complex"


def identify_key_transformations(route: List[Dict]) -> List[str]:
    """Identify key transformations in route."""
    key_transforms = []
    
    transform_descriptions = {
        "aldol": "C-C bond formation (Aldol)",
        "michael": "C-C bond formation (Michael)",
        "grignard": "C-C bond formation (Grignard)",
        "wittig": "C=C bond formation (Wittig)",
        "suzuki": "C-C coupling (Suzuki)",
        "heck": "C-C coupling (Heck)",
        "esterification": "Ester formation",
        "amidation": "Amide formation",
        "reduction_carbonyl": "Carbonyl reduction",
        "oxidation_alcohol": "Alcohol oxidation",
        "friedel_crafts_acyl": "Aromatic acylation",
        "friedel_crafts_alkyl": "Aromatic alkylation"
    }
    
    for step in route:
        reaction = step.get("reaction")
        if reaction in transform_descriptions:
            key_transforms.append(transform_descriptions[reaction])
    
    return key_transforms


def identify_strategic_bonds(mol: Chem.Mol) -> List[Dict]:
    """Identify strategic bonds for disconnection."""
    strategic_bonds = []
    
    # Look for bonds adjacent to functional groups
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        # Check if bond is strategic
        if is_strategic_bond(mol, bond):
            strategic_bonds.append({
                "bond_idx": bond.GetIdx(),
                "atoms": f"{begin_atom.GetSymbol()}{begin_atom.GetIdx()}-{end_atom.GetSymbol()}{end_atom.GetIdx()}",
                "type": identify_bond_type(mol, bond),
                "priority": calculate_bond_priority(mol, bond)
            })
    
    # Sort by priority
    strategic_bonds.sort(key=lambda x: x["priority"], reverse=True)
    
    return strategic_bonds[:5]  # Top 5 strategic bonds


def is_strategic_bond(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Check if a bond is strategic for retrosynthesis."""
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    
    # C-C bonds are often strategic
    if begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 6:
        # Check if adjacent to functional groups
        for neighbor in begin_atom.GetNeighbors() + end_atom.GetNeighbors():
            if neighbor.GetAtomicNum() in [7, 8, 16]:  # N, O, S
                return True
        
        # Check if part of a ring
        if bond.IsInRing():
            return False  # Usually don't break ring bonds
        
        return True
    
    return False


def identify_bond_type(mol: Chem.Mol, bond: Chem.Bond) -> str:
    """Identify the type of strategic bond."""
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    
    # Check for specific patterns
    if has_carbonyl_alpha(mol, bond):
        return "α-Carbonyl"
    elif has_heteroatom_beta(mol, bond):
        return "β-Heteroatom"
    elif is_benzylic(mol, bond):
        return "Benzylic"
    elif is_allylic(mol, bond):
        return "Allylic"
    else:
        return "C-C"


def calculate_bond_priority(mol: Chem.Mol, bond: Chem.Bond) -> float:
    """Calculate priority score for strategic bond."""
    priority = 1.0
    
    # Higher priority for bonds that lead to simpler fragments
    fragments = break_bond(mol, bond)
    if len(fragments) == 2:
        complexity_diff = abs(calculate_molecular_complexity(fragments[0]) - 
                            calculate_molecular_complexity(fragments[1]))
        priority += (1 - complexity_diff / 100)
    
    # Higher priority for common disconnection patterns
    bond_type = identify_bond_type(mol, bond)
    type_priorities = {
        "α-Carbonyl": 1.5,
        "β-Heteroatom": 1.3,
        "Benzylic": 1.2,
        "Allylic": 1.1,
        "C-C": 1.0
    }
    priority *= type_priorities.get(bond_type, 1.0)
    
    return priority


# Helper functions for bond analysis
def has_carbonyl_alpha(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Check if bond is alpha to carbonyl."""
    for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:  # Carbon
                for nn in neighbor.GetNeighbors():
                    if nn.GetAtomicNum() == 8 and neighbor.GetBondWithAtom(nn).GetBondType() == Chem.BondType.DOUBLE:
                        return True
    return False


def has_heteroatom_beta(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Check if bond is beta to heteroatom."""
    for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                for nn in neighbor.GetNeighbors():
                    if nn.GetAtomicNum() in [7, 8, 16]:  # N, O, S
                        return True
    return False


def is_benzylic(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Check if bond is benzylic."""
    for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                return True
    return False


def is_allylic(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Check if bond is allylic."""
    for atom in [bond.GetBeginAtom(), bond.GetEndAtom()]:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                for neighbor_bond in neighbor.GetBonds():
                    if neighbor_bond.GetBondType() == Chem.BondType.DOUBLE:
                        return True
    return False


def break_bond(mol: Chem.Mol, bond: Chem.Bond) -> List[Chem.Mol]:
    """Break a bond and return resulting fragments."""
    try:
        emol = Chem.EditableMol(mol)
        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        fragments = Chem.GetMolFrags(emol.GetMol(), asMols=True)
        return list(fragments)
    except:
        return []


def calculate_molecular_complexity(mol: Chem.Mol) -> float:
    """Calculate molecular complexity score."""
    if mol is None:
        return 0
    
    # Bertz complexity
    complexity = rdMolDescriptors.BertzCT(mol)
    
    # Normalize to 0-100 scale
    return min(complexity / 10, 100)


def identify_functional_groups(mol: Chem.Mol) -> List[str]:
    """Identify functional groups in molecule."""
    functional_groups = []
    
    fg_smarts = {
        "carboxylic_acid": "C(=O)[OH]",
        "ester": "C(=O)O[C,c]",
        "amide": "C(=O)N",
        "amine": "[NX3;H2,H1,H0]",
        "alcohol": "[OH][CX4]",
        "ketone": "[CX3](=[OX1])[CX4]",
        "aldehyde": "[CX3H1](=O)",
        "ether": "[OX2]([CX4])[CX4]",
        "nitrile": "C#N",
        "nitro": "[N+](=O)[O-]",
        "halide": "[F,Cl,Br,I]",
        "sulfide": "S",
        "aromatic": "a"
    }
    
    for name, smarts in fg_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            functional_groups.append(name)
    
    return functional_groups


def analyze_ring_systems(mol: Chem.Mol) -> Dict:
    """Analyze ring systems in molecule."""
    ring_info = mol.GetRingInfo()
    
    return {
        "num_rings": ring_info.NumRings(),
        "ring_sizes": list(map(len, ring_info.AtomRings())),
        "aromatic_rings": sum(1 for ring in ring_info.AtomRings() 
                            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)),
        "fused_rings": count_fused_rings(mol),
        "spiro_centers": count_spiro_centers(mol)
    }


def count_stereocenters(mol: Chem.Mol) -> Dict:
    """Count and analyze stereocenters."""
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    
    return {
        "total": len(chiral_centers),
        "assigned": sum(1 for center in chiral_centers if center[1] != '?'),
        "unassigned": sum(1 for center in chiral_centers if center[1] == '?')
    }


def count_fused_rings(mol: Chem.Mol) -> int:
    """Count number of fused ring systems."""
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    
    if len(rings) < 2:
        return 0
    
    fused_count = 0
    for i, ring1 in enumerate(rings):
        for j, ring2 in enumerate(rings[i+1:], i+1):
            if len(set(ring1) & set(ring2)) >= 2:
                fused_count += 1
    
    return fused_count


def count_spiro_centers(mol: Chem.Mol) -> int:
    """Count spiro centers in molecule."""
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    
    spiro_atoms = set()
    for i, ring1 in enumerate(rings):
        for j, ring2 in enumerate(rings[i+1:], i+1):
            shared = set(ring1) & set(ring2)
            if len(shared) == 1:
                spiro_atoms.update(shared)
    
    return len(spiro_atoms)


def calculate_sa_score_simple(mol: Chem.Mol) -> float:
    """Simple synthetic accessibility score."""
    # Simplified version - in practice would use more sophisticated scoring
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = len(mol.GetRingInfo().AtomRings())
    num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    sa_score = 1 + num_atoms * 0.05 + num_rings * 0.5 + num_stereo * 0.5
    
    return min(sa_score, 10)