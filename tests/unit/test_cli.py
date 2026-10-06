import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cli import main  # noqa: E402


def test_groups_json(capsys):
    rc = main(["groups", "--json", "CC(=O)Oc1ccccc1C(=O)O"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert [g["name"] for g in out[0]["groups"]] == ["carboxylic acid", "ester", "benzene ring"]


def test_groups_text_and_invalid(capsys):
    rc = main(["groups", "CCO", "not-a-smiles"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "primary alcohol" in captured.out
    assert "invalid SMILES" in captured.err


def test_groups_from_file(tmp_path, capsys):
    f = tmp_path / "s.txt"
    f.write_text("# comment\nCC(=O)O acetic acid\nCCO ethanol\n")
    assert main(["groups", "--file", str(f)]) == 0
    out = capsys.readouterr().out
    assert "carboxylic acid" in out and "primary alcohol" in out


def test_props_runs(capsys):
    rc = main(["props", "--json", "Nc1ccccc1[N+](=O)[O-]"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    alerts = out[0]["predictions"]["toxicity"]["acute_toxicity"]["structural_alerts"]
    assert "nitroaromatic" in alerts and "nitroso" not in alerts
