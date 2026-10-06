"""Functional-group identification for organic molecules.

The core O-chem skill: look at a structure, name what is in it. This module
does that deterministically with RDKit SMARTS — no model, no guessing. Every
hit reports the atom indices it matched so a result can be checked against
the drawing.

Precedence matters. A carboxylic acid contains a carbonyl and a hydroxyl; a
student who reports "ketone + alcohol" for acetic acid is wrong. Patterns are
therefore ordered from most to least specific, and each pattern names a KEY
ATOM (usually the carbonyl carbon or the heteroatom). Once a key atom has been
claimed by a more specific group, less specific patterns skip matches on it.

Usage:
    >>> from src.core.functional_groups import identify
    >>> [g.name for g in identify("CC(=O)Oc1ccccc1C(=O)O")]   # aspirin
    ['carboxylic acid', 'ester', 'benzene ring']
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class GroupPattern:
    name: str
    smarts: str
    family: str          # oxygen / nitrogen / sulfur / halogen / hydrocarbon / ...
    key_atoms: Tuple[int, ...]   # SMARTS indices that this group owns (claimed on match)
    note: str            # one line a TA would say
    pattern: Chem.Mol = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mol = Chem.MolFromSmarts(self.smarts)
        if mol is None:
            raise ValueError(f"bad SMARTS for {self.name}: {self.smarts}")
        object.__setattr__(self, "pattern", mol)


@dataclass
class GroupMatch:
    name: str
    family: str
    count: int
    atoms: List[Tuple[int, ...]]   # one tuple of atom indices per occurrence
    note: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "count": self.count,
            "atoms": [list(a) for a in self.atoms],
            "note": self.note,
        }


# Ordered most-specific → least-specific. Key atoms are chosen so that a
# carbonyl carbon claimed by "ester" is not re-reported as "ketone", an
# oxygen claimed by "ester" is not re-reported as "ether", etc.
PATTERNS: Sequence[GroupPattern] = (
    # --- carboxylic acid derivatives (carbonyl carbon is the key) ---------
    GroupPattern("acid anhydride", "[CX3](=O)[OX2][CX3](=O)", "oxygen", (0, 2, 3),
                 "Two acyl groups sharing one oxygen. Most reactive of the common derivatives after acyl halides."),
    GroupPattern("acyl halide", "[CX3](=O)[F,Cl,Br,I]", "oxygen", (0,),
                 "Carbonyl bonded directly to a halogen. The most electrophilic acid derivative."),
    GroupPattern("carboxylic acid", "[CX3](=O)[OX2H1]", "oxygen", (0,),
                 "C(=O)OH. pKa ≈ 4–5; deprotonates with bicarbonate."),
    GroupPattern("carboxylate", "[CX3](=O)[O-]", "oxygen", (0,),
                 "Deprotonated carboxylic acid. Resonance-stabilised anion."),
    GroupPattern("carbamate", "[NX3][CX3](=O)[OX2][#6]", "nitrogen", (1, 3),
                 "N–C(=O)–O. Urethane linkage; common protecting group (Boc, Cbz)."),
    GroupPattern("urea", "[#7X3][#6X3](=O)[#7X3]", "nitrogen", (1,),
                 "N–C(=O)–N. Carbonyl flanked by two nitrogens."),
    GroupPattern("ester", "[CX3](=O)[OX2][#6]", "oxygen", (0, 2),
                 "C(=O)O–C. Name as alkyl alkanoate; hydrolyses to acid + alcohol."),
    GroupPattern("amide", "[#6X3](=O)[#7X3]", "nitrogen", (0,),
                 "C(=O)N. Planar, resonance-stabilised; the least reactive acid derivative."),
    GroupPattern("aldehyde", "[CX3H1](=O)[#6,#1]", "oxygen", (0,),
                 "Terminal C(=O)H. Oxidises easily to the acid; Tollens / Fehling positive."),
    GroupPattern("ketone", "[#6][CX3](=O)[#6]", "oxygen", (1,),
                 "Carbonyl between two carbons. Does not oxidise under mild conditions."),
    # --- nitrogen -----------------------------------------------------------
    GroupPattern("nitrile", "[NX1]#[CX2]", "nitrogen", (1,),
                 "C≡N. Hydrolyses to carboxylic acid; reduces to primary amine."),
    GroupPattern("isocyanate", "[NX2]=[CX2]=[OX1]", "nitrogen", (1,),
                 "N=C=O. Reacts with alcohols to give carbamates."),
    GroupPattern("azide", "[NX2]=[NX2+]=[NX1-]", "nitrogen", (0,),
                 "N3. Click-chemistry partner; potentially energetic."),
    GroupPattern("nitro", "[NX3+](=O)[O-]", "nitrogen", (0,),
                 "NO2. Strong electron-withdrawing group; meta-directing on arenes."),
    GroupPattern("nitroso", "[NX2]=[OX1]", "nitrogen", (0,),
                 "N=O on carbon."),
    GroupPattern("imine", "[CX3]=[NX2][#6,#1]", "nitrogen", (1,),
                 "C=N. Schiff base; forms from aldehyde/ketone + primary amine."),
    GroupPattern("oxime", "[CX3]=[NX2][OX2H1]", "nitrogen", (1,),
                 "C=N–OH. From carbonyl + hydroxylamine."),
    GroupPattern("hydrazone", "[CX3]=[NX2][NX3]", "nitrogen", (1,),
                 "C=N–N. From carbonyl + hydrazine."),
    GroupPattern("hydrazine", "[NX3][NX3]", "nitrogen", (0,),
                 "N–N single bond."),
    GroupPattern("sulfonamide", "[SX4](=O)(=O)[NX3]", "sulfur", (0, 3),
                 "SO2N. Sulfa drugs; N–H is weakly acidic (pKa ≈ 10)."),
    GroupPattern("aniline (aryl amine)", "[NX3;H2,H1,H0;!$(NC=O);!$(N-[SX4])][c]", "nitrogen", (0,),
                 "Nitrogen on an aromatic ring. Less basic than alkyl amines (lone pair delocalised)."),
    GroupPattern("primary amine", "[NX3;H2;!$(NC=O);!$(N-[SX4]);!$(N-N);!$(N-O)][CX4]", "nitrogen", (0,),
                 "R–NH2. Basic (pKa of conjugate acid ≈ 10–11)."),
    GroupPattern("secondary amine", "[NX3;H1;!$(NC=O);!$(N-[SX4]);!$(N-N);!$(N-O)]([CX4])[CX4]", "nitrogen", (0,),
                 "R2NH."),
    GroupPattern("tertiary amine", "[NX3;H0;!$(NC=O);!$(N-[SX4]);!$(N-N);!$(N-O)]([CX4])([CX4])[CX4]", "nitrogen", (0,),
                 "R3N. No N–H; cannot hydrogen-bond donate."),
    GroupPattern("quaternary ammonium", "[NX4+]", "nitrogen", (0,),
                 "R4N+. Permanently charged."),
    # --- oxygen -------------------------------------------------------------
    GroupPattern("epoxide", "[OX2r3]1[#6r3][#6r3]1", "oxygen", (0,),
                 "Three-membered cyclic ether. Strained; opens with nucleophiles."),
    GroupPattern("hemiacetal", "[OX2H1][CX4][OX2][#6]", "oxygen", (0, 1, 2),
                 "C bearing both OH and OR. Sugar ring form."),
    GroupPattern("acetal", "[#6][OX2][CX4;!$(C=O)]([OX2][#6])", "oxygen", (1, 2, 3),
                 "C bearing two OR groups. Carbonyl protecting group; acid-labile."),
    GroupPattern("phenol", "[OX2H1][c]", "oxygen", (0,),
                 "Ar–OH. pKa ≈ 10; more acidic than alcohols."),
    GroupPattern("enol", "[OX2H1][CX3]=[CX3]", "oxygen", (0,),
                 "OH on an alkene carbon. Usually tautomerises to the carbonyl."),
    GroupPattern("primary alcohol", "[OX2H1][CX4;H2]", "oxygen", (0,),
                 "RCH2–OH. Oxidises to aldehyde then acid."),
    GroupPattern("secondary alcohol", "[OX2H1][CX4;H1]", "oxygen", (0,),
                 "R2CH–OH. Oxidises to ketone."),
    GroupPattern("tertiary alcohol", "[OX2H1][CX4;H0]", "oxygen", (0,),
                 "R3C–OH. Resists oxidation; SN1-prone."),
    GroupPattern("peroxide", "[OX2][OX2]", "oxygen", (0,),
                 "O–O. Weak bond; radical initiator."),
    GroupPattern("ether", "[OD2;!$(O-C=O);!$(O-[SX4])]([#6])[#6]", "oxygen", (0,),
                 "C–O–C. Relatively inert; good solvent."),
    # --- sulfur -------------------------------------------------------------
    GroupPattern("sulfonic acid", "[SX4](=O)(=O)[OX2H1]", "sulfur", (0, 3),
                 "SO3H. Strong acid."),
    GroupPattern("sulfone", "[#6][SX4](=O)(=O)[#6]", "sulfur", (1,),
                 "C–SO2–C."),
    GroupPattern("sulfoxide", "[#6][SX3](=O)[#6]", "sulfur", (1,),
                 "C–S(=O)–C. DMSO is the parent solvent."),
    GroupPattern("thiol", "[SX2H1][#6]", "sulfur", (0,),
                 "R–SH. pKa ≈ 10; soft nucleophile; strong odour."),
    GroupPattern("disulfide", "[SX2][SX2]", "sulfur", (0,),
                 "S–S. Cystine crosslink."),
    GroupPattern("thioether (sulfide)", "[#6][SX2][#6]", "sulfur", (1,),
                 "C–S–C."),
    # --- halogen ------------------------------------------------------------
    GroupPattern("aryl halide", "[c][F,Cl,Br,I]", "halogen", (1,),
                 "Halogen on an aromatic ring. Not SN2-reactive; cross-coupling partner."),
    GroupPattern("vinyl halide", "[CX3]=[CX3][F,Cl,Br,I]", "halogen", (2,),
                 "Halogen on an alkene carbon."),
    GroupPattern("alkyl halide", "[CX4][F,Cl,Br,I]", "halogen", (1,),
                 "sp3 C–X. SN1/SN2/E1/E2 substrate."),
    # --- hydrocarbon --------------------------------------------------------
    GroupPattern("alkyne", "[CX2]#[CX2]", "hydrocarbon", (0,),
                 "C≡C. Terminal alkyne C–H is weakly acidic (pKa ≈ 25)."),
    GroupPattern("alkene", "[CX3]=[CX3]", "hydrocarbon", (0,),
                 "C=C. Electrophilic addition; cis/trans isomerism possible."),
    GroupPattern("benzene ring", "c1ccccc1", "hydrocarbon", (0,),
                 "Six-membered carbocyclic aromatic."),
)

HETEROAROMATIC_NOTE = "Aromatic ring containing N, O or S (pyridine, furan, thiophene, imidazole, ...)."


def _ring_dedupe(matches: Iterable[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Collapse matches that are the same atom set in a different order (ring symmetry)."""
    seen = set()
    out: List[Tuple[int, ...]] = []
    for m in matches:
        key = frozenset(m)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def identify(smiles_or_mol, *, include_notes: bool = True) -> List[GroupMatch]:
    """Return the functional groups present, most specific first.

    Raises ValueError on an unparseable SMILES. Returns an empty list for a
    molecule with no recognised groups (e.g. an alkane) — that is a real
    answer, not a failure.
    """
    mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else Chem.MolFromSmiles(smiles_or_mol)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles_or_mol!r}")

    claimed: set = set()
    results: List[GroupMatch] = []

    for gp in PATTERNS:
        raw = mol.GetSubstructMatches(gp.pattern, uniquify=True)
        if not raw:
            continue
        hits: List[Tuple[int, ...]] = []
        for m in raw:
            if any(m[k] in claimed for k in gp.key_atoms):
                continue
            hits.append(m)
        hits = _ring_dedupe(hits)
        if not hits:
            continue
        # Claim key atoms so less specific patterns skip them. Alkene and the
        # ring patterns claim nothing: a C=C inside an enone is still an alkene.
        if gp.name not in ("alkene", "benzene ring"):
            for m in hits:
                claimed.update(m[k] for k in gp.key_atoms)
        results.append(GroupMatch(
            name=gp.name, family=gp.family, count=len(hits), atoms=hits,
            note=gp.note if include_notes else "",
        ))
    hetero_rings = _heteroaromatic_rings(mol)
    if hetero_rings:
        results.append(GroupMatch(
            name="heteroaromatic ring", family="hydrocarbon", count=len(hetero_rings),
            atoms=hetero_rings, note=HETEROAROMATIC_NOTE if include_notes else "",
        ))
    return results


def _heteroaromatic_rings(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Aromatic rings (by RingInfo) that contain at least one non-carbon atom."""
    out: List[Tuple[int, ...]] = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetIsAromatic() for a in atoms) and any(a.GetAtomicNum() != 6 for a in atoms):
            out.append(tuple(ring))
    return out


def summarize(smiles: str) -> str:
    """Human-readable one-liner: '2× ester, 1× benzene ring'."""
    groups = identify(smiles, include_notes=False)
    if not groups:
        return "no functional groups recognised (saturated hydrocarbon?)"
    return ", ".join(f"{g.count}× {g.name}" if g.count > 1 else g.name for g in groups)


__all__ = ["GroupPattern", "GroupMatch", "PATTERNS", "identify", "summarize"]
