"""Build isolated synthetic packages; no cloud models or plant claims."""

from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from registry.signer import generate_keypair
from sdk.manifest import seal_manifest
from sdk.schema import read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package_factory(tmp_path):
    def create(name="contract_probe", *, source=None):
        package = tmp_path / "skills" / name
        shutil.copytree(PROJECT_ROOT / "examples" / "contract_probe", package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        metadata = deepcopy(read_json(PROJECT_ROOT / "examples" / "contract_probe.metadata.json"))
        metadata["name"] = name
        instructions = package / "SKILL.md"
        instructions.write_text(instructions.read_text(encoding="utf-8").replace("name: contract-probe", f"name: {name.replace('_', '-')}"), encoding="utf-8")
        if source is not None:
            (package / "skill.py").write_text(source, encoding="utf-8")
        seal_manifest(package, metadata)
        return package
    return create


@pytest.fixture
def package(package_factory):
    return package_factory()


@pytest.fixture
def keys(tmp_path):
    private, public = tmp_path / "trust" / "publisher.private.pem", tmp_path / "trust" / "publisher.public.pem"
    generate_keypair(private, public)
    return private, public
