"""Validate a Skill without importing or executing its Python entrypoint."""

from pathlib import Path

import yaml

from registry.signer import verify_signature
from sdk.exceptions import ManifestError, SignatureError
from sdk.manifest import verify_manifest
from sdk.schema import load_schema, package_file


def read_instructions(package_dir: Path, manifest: dict) -> str:
    try:
        text = package_file(package_dir, "SKILL.md").read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0] != "---":
            raise ValueError("missing YAML frontmatter")
        end = lines.index("---", 1)
        metadata = yaml.safe_load("\n".join(lines[1:end]))
        expected_name = manifest["name"].replace("_", "-")
        if (not isinstance(metadata, dict) or metadata.get("name") != expected_name
                or not isinstance(metadata.get("description"), str)
                or not metadata["description"].strip()
                or not "\n".join(lines[end + 1:]).strip()):
            raise ValueError(f"expected name {expected_name}, description and instruction body")
        return text
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise ManifestError(f"Invalid SKILL.md: {exc}") from exc


def validate_package(package_dir: str | Path, *, trusted_public_key: str | Path | None = None,
                     require_signature: bool = True) -> dict:
    package_dir = Path(package_dir)
    manifest = verify_manifest(package_dir)
    if package_dir.name not in {manifest["name"], manifest["name"].replace("_", "-")}:
        raise ManifestError("Package directory name must match manifest.name or its hyphenated Skill name")
    if require_signature and trusted_public_key is None:
        raise SignatureError("A pinned public key is required for signed discovery")
    signature = ({"status": "not_checked"} if trusted_public_key is None
                 else verify_signature(package_dir, trusted_public_key))
    read_instructions(package_dir, manifest)
    return {
        "package_dir": str(package_dir.resolve()),
        "manifest": manifest,
        "input_schema": load_schema(package_dir, manifest["input_schema"]),
        "output_schema": load_schema(package_dir, manifest["output_schema"]),
        "verification": {"integrity": "passed", "signature": signature,
                         "scanned": "not_checked", "evaluated": "not_checked"},
    }
