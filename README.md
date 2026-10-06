# OChem Helper

A small, honest organic-chemistry helper built on RDKit. It names the
functional groups in a molecule, checks drug-likeness rules, and flags
structural alerts — the things a TA actually checks when marking a problem set.

It is **not** a neural network. An earlier version of this README described
a VAE / GNN molecular-discovery system. That code is still in the tree, but it
was never trained and never worked; the "generated molecules" it produced were
random fragments from a template list. That is gone. What remains is
deterministic, inspectable, and runs in under a second with no model weights.

## Install

```bash
git clone https://github.com/GreatPyreneseDad/OchemHelper.git
cd OchemHelper
pip install -e .          # RDKit, numpy, scikit-learn. No PyTorch.
```

## Use

```bash
# Name the functional groups (most specific first; aspirin is an acid + ester, not a ketone + alcohol)
ochem groups "CC(=O)Oc1ccccc1C(=O)O"
#   carboxylic acid
#   ester
#   benzene ring

# With the one-line note a TA would say, and the atom indices so you can check the drawing
ochem groups --notes --atoms "CC(=O)Oc1ccccc1C(=O)O"

# Lipinski / Veber / Egan, descriptors, and toxicity structural alerts
ochem props "Nc1ccccc1[N+](=O)[O-]"
#   MW 138.13  logP 1.18  TPSA 69.16  HBD 1  HBA 3  rotB 1
#   Lipinski violations: 0   Veber: pass   Egan: pass   → oral bioavailability High (1)
#   structural alerts: nitroaromatic, aromatic_amine   mutagenicity risk: High

# Batch: one SMILES per line (a trailing name is ignored), or pipe on stdin
ochem groups --file problem_set.txt
ochem props --json "CCO" "CC(=O)O" > out.json
```

From Python:

```python
from src.core.functional_groups import identify, summarize

summarize("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
# 'hemiacetal, primary alcohol, 3× secondary alcohol'

for g in identify("CC(C)(C)OC(=O)NCC(=O)O"):
    print(g.name, g.count, g.atoms)
# carboxylic acid 1 [(9, 10, 11)]
# carbamate 1 [(7, 5, 6, 4, 1)]
```

### How group identification works

`src/core/functional_groups.py` is an ordered table of ~45 SMARTS patterns,
most specific first. Each pattern names the atoms it *owns* (the carbonyl
carbon of an ester, the two oxygens of a hemiacetal). Once a more specific
group has claimed an atom, less specific patterns skip it — so a carboxylic
acid is never also reported as a ketone and an alcohol, and the ring oxygen of
glucose is part of the hemiacetal, not an ether. Alkenes and aromatic rings
claim nothing, because a C=C inside an enone is still an alkene.

Every hit carries its atom indices. If the tool is wrong, you can see exactly
where, and fix the SMARTS. Tests in `tests/unit/test_functional_groups.py`
pin the answers for ~40 textbook molecules (aspirin, paracetamol, caffeine,
glucose, penicillin G, Boc-glycine, cysteine …).

## What else is in the tree

| path | status |
|---|---|
| `src/core/functional_groups.py`, `src/cli.py` | **works, tested** |
| `mcp/tools/predict_properties.py` | works — Lipinski/Veber/Egan, SMARTS toxicity alerts, descriptor-based ADMET heuristics. Rules, not models. |
| `mcp/tools/suggest_synthesis.py` | **broken** — template-based retrosynthesis; several reaction SMARTS do not parse and one RDKit call does not exist. Next thing to fix. |
| `mcp/server/*` | MCP server exposing the tools; not re-verified since the cleanup |
| `src/models/generative/smiles_vae.py` | untrained VAE scaffold. `/api/generate` returns 503 until a checkpoint exists at `models/checkpoints/best_model.pt`. No mock fallback. |
| `src/core/hyperposition_tokenizer.py` | experimental molecular tokenizer; not used by the CLI |
| `dashboard/` | browser dashboard from the original build; not re-verified |
| `scripts/train_vae.py`, `scripts/download_data.py` | would train the VAE if you had the data and the patience; never run |

## Running the tests

```bash
pip install pytest
pytest -q tests/unit/test_functional_groups.py tests/unit/test_cli.py
```

The older tests under `tests/unit/test_vae.py`, `test_mcp.py`, `test_api.py`
hard-code a path on the original author's laptop and import torch; they are
left in place but are not expected to pass.

## Roadmap (chipping at it)

1. Fix `suggest_synthesis.py` so retrosynthesis actually runs, with tests on
   real textbook disconnections (ester → acid + alcohol, amide → acid + amine,
   Grignard, aldol).
2. Add imide, lactone/lactam labelling, aryl ether, conjugated system detection.
3. `ochem name` — IUPAC-ish name for simple molecules, or at least the parent chain.
4. `ochem quiz` — show a structure, hide the answer, grade the response. The
   thing it was always for.

## License

MIT.
