import json
from pathlib import Path
import subprocess
import sys

from demo.run_registry_demo import run_demo
from registry.__main__ import main


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
    project = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "report.json"
    process = subprocess.run([sys.executable, "-m", "demo.run_registry_demo", "--output", str(report_path)],
                             cwd=project, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert report["result"]["mode"] == "fixture"
