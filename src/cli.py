"""`ochem` — command-line front end for the parts of OChem Helper that run.

    ochem groups  "CC(=O)Oc1ccccc1C(=O)O"      # name the functional groups
    ochem props   "CC(=O)Oc1ccccc1C(=O)O"      # Lipinski / Veber / Egan, alerts, descriptors
    ochem groups  --json  ...                   # machine-readable
    ochem groups  --file  smiles.txt            # one SMILES per line

Everything here is deterministic RDKit. No model, no network.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable, List

from rdkit import Chem, RDLogger

from src.core.functional_groups import identify

RDLogger.DisableLog("rdApp.*")  # keep stderr for our own messages


# --- input handling ----------------------------------------------------------

def _read_inputs(args: argparse.Namespace) -> List[str]:
    smiles: List[str] = list(args.smiles or [])
    if args.file:
        for line in Path(args.file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                smiles.append(line.split()[0])  # allow "SMILES name" lines
    if not smiles and not sys.stdin.isatty():
        smiles.extend(l.strip().split()[0] for l in sys.stdin if l.strip())
    if not smiles:
        raise SystemExit("no SMILES given (positional, --file, or stdin)")
    return smiles


def _parse(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smi!r}")
    return mol


# --- groups ------------------------------------------------------------------

def cmd_groups(args: argparse.Namespace) -> int:
    rc = 0
    out = []
    for smi in _read_inputs(args):
        try:
            mol = _parse(smi)
        except ValueError as e:
            if args.json:
                out.append({"smiles": smi, "error": str(e)})
            else:
                print(f"{smi}\n  ✗ {e}", file=sys.stderr)
            rc = 1
            continue
        groups = identify(mol, include_notes=args.notes)
        if args.json:
            out.append({"smiles": Chem.MolToSmiles(mol), "groups": [g.to_dict() for g in groups]})
            continue
        print(smi)
        if not groups:
            print("  (no functional groups recognised — saturated hydrocarbon?)")
        for g in groups:
            count = f"{g.count}× " if g.count > 1 else ""
            atoms = "  atoms " + "; ".join(",".join(map(str, a)) for a in g.atoms) if args.atoms else ""
            print(f"  {count}{g.name}{atoms}")
            if args.notes and g.note:
                print(f"      {g.note}")
    if args.json:
        print(json.dumps(out, indent=2))
    return rc


# --- props -------------------------------------------------------------------

def cmd_props(args: argparse.Namespace) -> int:
    # The property rules live in the MCP tool module; import lazily so `groups`
    # never pays for sklearn/joblib.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))
    from tools import predict_properties  # type: ignore

    rc = 0
    out = []
    for smi in _read_inputs(args):
        result = asyncio.run(predict_properties.predict_activity(smi))
        if "error" in result:
            rc = 1
        if args.json:
            out.append(result)
            continue
        print(smi)
        if "error" in result:
            print(f"  ✗ {result['error']}", file=sys.stderr)
            continue
        p = result["predictions"]
        bio = p.get("bioavailability", {})
        d = bio.get("details", {})
        print(f"  MW {d.get('MW')}  logP {d.get('logP')}  TPSA {d.get('TPSA')}  "
              f"HBD {d.get('HBD')}  HBA {d.get('HBA')}  rotB {d.get('rotatable_bonds')}")
        print(f"  Lipinski violations: {bio.get('lipinski_violations')}   "
              f"Veber: {'pass' if bio.get('veber_compliant') else 'fail'}   "
              f"Egan: {'pass' if bio.get('egan_compliant') else 'fail'}   "
              f"→ oral bioavailability {bio.get('category')} ({bio.get('score')})")
        tox = p.get("toxicity", {})
        alerts = (tox.get("acute_toxicity") or {}).get("structural_alerts") or []
        mut = (tox.get("mutagenicity") or {}).get("risk")
        if alerts:
            print(f"  structural alerts: {', '.join(map(str, alerts))}   mutagenicity risk: {mut}")
        else:
            print("  structural alerts: none")
    if args.json:
        print(json.dumps(out, indent=2))
    return rc


# --- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ochem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("smiles", nargs="*", help="SMILES strings")
        p.add_argument("--file", "-f", help="file with one SMILES per line")
        p.add_argument("--json", action="store_true", help="JSON output")

    g = sub.add_parser("groups", help="identify functional groups")
    common(g)
    g.add_argument("--atoms", action="store_true", help="show matched atom indices")
    g.add_argument("--notes", action="store_true", help="show the one-line TA note per group")
    g.set_defaults(func=cmd_groups)

    p = sub.add_parser("props", help="drug-likeness rules, descriptors, structural alerts")
    common(p)
    p.set_defaults(func=cmd_props)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
