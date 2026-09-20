"""Seal author-owned Skill sources after reviewed edits; never called by consumers."""

import json

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from sdk.manifest import seal_manifest
from sdk.schema import read_json


def main() -> None:
    sealed = {}
    for name in SKILL_NAMES:
        package = PROJECT_ROOT / "skills" / name
        manifest = seal_manifest(package, read_json(package / "metadata.json"))
        sealed[name] = manifest["manifest_sha256"]
    print(json.dumps(sealed, indent=2))


if __name__ == "__main__":
    main()
