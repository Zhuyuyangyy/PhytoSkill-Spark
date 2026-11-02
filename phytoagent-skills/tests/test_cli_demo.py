import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from demo.run_registry_demo import run_demo
from registry.__main__ import main
from runtime.executor import SkillExecutor
from sdk.cli import run_package_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = PROJECT_ROOT / "skills"


def test_publisher_and_consumer_cli_lifecycle(package, tmp_path, capsys):
    private, public = tmp_path / "cli.private.pem", tmp_path / "cli.public.pem"
    assert main(["keygen", "--private-key", str(private), "--public-key", str(public)]) == 0
    capsys.readouterr()
    assert main(["sign", str(package), "--private-key", str(private)]) == 0
    capsys.readouterr()
    assert main(["verify", str(package), "--public-key", str(public)]) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["verification"]["signature"]["status"] == "passed"
    assert main(["tools", "--skills-dir", str(package.parent), "--public-key", str(public)]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert tools[0]["function"]["parameters"]["required"] == ["message"]
    assert main(["verify", str(package), "--public-key", str(tmp_path / "missing.pem")]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "SignatureError"


def test_demo_runs_signed_contract_loop():
    report = run_demo()
    assert report["scope"] == "sdk_registry_only"
    assert report["model_called"] is False
    assert report["verification"]["signature"]["status"] == "passed"
    assert report["instructions_loaded"] is True
    assert report["result"]["data_origin"] == "synthetic_fixture"
    assert report["result"]["echo"] == "SDK + Registry contract verified"


def test_module_entrypoint_prints_json(tmp_path):
    report_path = tmp_path / "report.json"
    process = subprocess.run([sys.executable, "-m", "demo.run_registry_demo", "--output", str(report_path)],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert report["result"]["mode"] == "fixture"


@pytest.mark.parametrize(("skill_name", "mode"), [
    ("plant_vision", "herb"),
    ("herbal_knowledge", "corpus"),
])
def test_package_cli_forwards_modes_declared_by_each_manifest(
        skill_name, mode, monkeypatch, capsys):
    package = SKILLS_DIR / skill_name
    input_path = package / "examples" / "request.json"
    forwarded = []

    def execute(_executor, name, payload, *, mode):
        forwarded.append((name, mode, payload))
        return {"skill": name, "mode": mode}

    monkeypatch.setattr(SkillExecutor, "execute", execute)
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--input", str(input_path), "--mode", mode, "--allow-unsigned",
    ])

    assert run_package_cli(package) == 0
    result = json.loads(capsys.readouterr().out)
    assert forwarded == [(skill_name, mode, json.loads(input_path.read_text(encoding="utf-8")))]
    assert result == {"skill": skill_name, "mode": mode}


def test_package_cli_rejects_mode_not_declared_by_that_manifest(monkeypatch, capsys):
    package = SKILLS_DIR / "growth_risk"
    input_path = package / "examples" / "request.json"
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--input", str(input_path), "--mode", "corpus", "--allow-unsigned",
    ])

    assert run_package_cli(package) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "UnsupportedModeError"
    assert "growth_risk does not support mode 'corpus'" in error["message"]


def test_corpus_cli_mode_runs_locally_without_a_model_request():
    package = SKILLS_DIR / "herbal_knowledge"
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUTF8": "1"}
    process = subprocess.run(
        [sys.executable, "scripts/run.py", "--input", "examples/request.json",
         "--mode", "corpus", "--allow-unsigned"],
        cwd=package, env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )

    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["provenance"]["data_origin"] == "local_corpus"
    assert result["provenance"]["model_called"] is False
