"""Functional-group identification: the answers a TA would mark correct."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.functional_groups import identify, summarize, PATTERNS  # noqa: E402


def names(smiles):
    return [g.name for g in identify(smiles, include_notes=False)]


# (name, SMILES, expected groups in order of specificity)
CASES = [
    ("acetic acid", "CC(=O)O", ["carboxylic acid"]),
    ("ethanol", "CCO", ["primary alcohol"]),
    ("isopropanol", "CC(O)C", ["secondary alcohol"]),
    ("t-butanol", "CC(C)(C)O", ["tertiary alcohol"]),
    ("acetone", "CC(=O)C", ["ketone"]),
    ("acetaldehyde", "CC=O", ["aldehyde"]),
    ("acetamide", "CC(=O)N", ["amide"]),
    ("ethyl acetate", "CCOC(=O)C", ["ester"]),
    ("diethyl ether", "CCOCC", ["ether"]),
    ("aniline", "Nc1ccccc1", ["aniline (aryl amine)", "benzene ring"]),
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1", ["nitro", "benzene ring"]),
    ("benzonitrile", "N#Cc1ccccc1", ["nitrile", "benzene ring"]),
    ("ethylene oxide", "C1CO1", ["epoxide"]),
    ("chlorobenzene", "Clc1ccccc1", ["aryl halide", "benzene ring"]),
    ("bromoethane", "CCBr", ["alkyl halide"]),
    ("vinyl chloride", "C=CCl", ["vinyl halide", "alkene"]),
    ("propyne", "CC#C", ["alkyne"]),
    ("acetic anhydride", "CC(=O)OC(=O)C", ["acid anhydride"]),
    ("acetyl chloride", "CC(=O)Cl", ["acyl halide"]),
    ("urea", "NC(N)=O", ["urea"]),
    ("DMSO", "CS(C)=O", ["sulfoxide"]),
    ("triethylamine", "CCN(CC)CC", ["tertiary amine"]),
    ("pyridine", "c1ccncc1", ["heteroaromatic ring"]),
    ("dimethyl acetal", "COC(C)OC", ["acetal"]),
    ("hexane", "CCCCCC", []),
]


@pytest.mark.parametrize("name,smiles,expected", CASES, ids=[c[0] for c in CASES])
def test_single_group_molecules(name, smiles, expected):
    assert names(smiles) == expected


def test_aspirin_is_acid_plus_ester_not_ketone_or_alcohol():
    got = names("CC(=O)Oc1ccccc1C(=O)O")
    assert got == ["carboxylic acid", "ester", "benzene ring"]
    assert "ketone" not in got and "phenol" not in got and "ether" not in got


def test_paracetamol():
    assert names("CC(=O)Nc1ccc(O)cc1") == ["amide", "phenol", "benzene ring"]


def test_caffeine_two_rings_urea_and_amide():
    got = identify("Cn1cnc2c1c(=O)n(C)c(=O)n2C", include_notes=False)
    by = {g.name: g for g in got}
    assert set(by) == {"urea", "amide", "heteroaromatic ring"}
    assert by["heteroaromatic ring"].count == 2


def test_glucose_hemiacetal_claims_anomeric_oh_and_ring_oxygen():
    got = {g.name: g for g in identify("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", include_notes=False)}
    assert set(got) == {"hemiacetal", "primary alcohol", "secondary alcohol"}
    assert got["secondary alcohol"].count == 3   # C2, C3, C4 — not the anomeric carbon
    assert "ether" not in got                     # the ring O belongs to the hemiacetal


def test_boc_glycine_is_carbamate_not_ester():
    got = names("CC(C)(C)OC(=O)NCC(=O)O")
    assert got == ["carboxylic acid", "carbamate"]


def test_cysteine():
    assert names("N[C@@H](CS)C(=O)O") == ["carboxylic acid", "primary amine", "thiol"]


def test_penicillin_g():
    got = {g.name: g.count for g in identify(
        "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", include_notes=False)}
    assert got == {"carboxylic acid": 1, "amide": 2, "thioether (sulfide)": 1, "benzene ring": 1}


def test_sulfanilamide_sulfonamide_nitrogen_is_not_an_amine():
    got = names("Nc1ccc(cc1)S(N)(=O)=O")
    assert got == ["sulfonamide", "aniline (aryl amine)", "benzene ring"]
    assert "primary amine" not in got


def test_enone_reports_both_ketone_and_alkene():
    assert names("CC(=O)C=C") == ["ketone", "alkene"]


def test_atoms_are_reported_and_in_range():
    mol_groups = identify("CC(=O)Oc1ccccc1C(=O)O")
    for g in mol_groups:
        assert g.count == len(g.atoms)
        for tup in g.atoms:
            assert all(0 <= i < 13 for i in tup)


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        identify("C1CC")


def test_all_patterns_compile_and_key_atoms_in_range():
    for gp in PATTERNS:
        assert gp.pattern is not None, gp.name
        n = gp.pattern.GetNumAtoms()
        assert all(0 <= k < n for k in gp.key_atoms), gp.name


def test_summarize():
    assert summarize("CC(=O)Oc1ccccc1C(=O)O") == "carboxylic acid, ester, benzene ring"
    assert summarize("CCCCCC").startswith("no functional groups")
