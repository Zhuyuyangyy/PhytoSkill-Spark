"""Manifest hashing covers metadata plus the complete non-generated inventory."""

import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator

from sdk.exceptions import ContractError, ManifestError
from sdk.schema import load_schema, package_file, read_json

MANIFEST_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "version", "description", "author", "runtime", "entrypoint",
                 "input_schema", "output_schema", "supported_modes", "files", "manifest_sha256"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
        "version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},
        "description": {"type": "string", "minLength": 1, "maxLength": 1024},
        "author": {"type": "string", "minLength": 1},
        "runtime": {"enum": ["local_python", "local_dgx"]},
        "entrypoint": {"type": "string", "pattern": "^[A-Za-z0-9_./-]+\\.py:[A-Za-z_][A-Za-z0-9_]*$"},
        "input_schema": {"type": "string", "minLength": 1},
        "output_schema": {"type": "string", "minLength": 1},
        "supported_modes": {"type": "array", "minItems": 1, "uniqueItems": True,
                            "items": {"enum": ["fixture", "corpus", "replay", "live"]}},
        "capabilities": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "permissions": {"$ref": "#/$defs/permissions"},
        "files": {"type": "object", "minProperties": 3,
                  "additionalProperties": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
        "manifest_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    },
    "$defs": {
        "permissions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["filesystem", "network"],
            "properties": {
                "filesystem": {"type": "array", "uniqueItems": True, "minItems": 1,
                               "items": {"type": "string", "minLength": 1, "maxLength": 64}},
                "network": {"enum": ["deny", "allow"]},
            },
        },
    },
}


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def manifest_digest(manifest: dict) -> str:
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(content)).hexdigest()


def _linked(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def file_inventory(package_dir: Path) -> dict[str, str]:
    if _linked(package_dir) or not package_dir.is_dir():
        raise ManifestError("Skill package must be an existing directory, not a link")
    inventory = {}

    def fail_walk(error):
        raise ManifestError(f"Cannot inspect package: {error}")

    for directory, dirs, files in os.walk(package_dir, followlinks=False, onerror=fail_walk):
        base = Path(directory)
        for name in dirs + files:
            if _linked(base / name):
                raise ManifestError(f"Linked resource is forbidden: {name}")
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(package_dir).as_posix()
            if relative in ("manifest.json", "manifest.sig") or name.endswith((".pyc", ".pyo")):
                continue
            try:
                inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ManifestError(f"Cannot hash {relative}") from exc
    return dict(sorted(inventory.items()))


def validate_manifest(manifest: dict, package_dir: Path) -> None:
    error = next(Draft202012Validator(MANIFEST_SCHEMA).iter_errors(manifest), None)
    if error:
        raise ManifestError(f"Invalid manifest: {error.message}")
    declared = ["SKILL.md", manifest["entrypoint"].split(":")[0],
                manifest["input_schema"].split("#")[0], manifest["output_schema"].split("#")[0]]
    try:
        for relative in manifest["files"]:
            package_file(package_dir, relative)
        for relative in declared:
            if relative not in manifest["files"]:
                raise ManifestError(f"Required resource missing from inventory: {relative}")
        load_schema(package_dir, manifest["input_schema"])
        load_schema(package_dir, manifest["output_schema"])
    except ContractError as exc:
        raise ManifestError(str(exc)) from exc


def seal_manifest(package_dir: Path, metadata: dict) -> dict:
    """Explicit publisher action. Never call this to repair a verification failure."""
    package_dir = Path(package_dir)
    manifest = {key: value for key, value in metadata.items() if key not in ("files", "manifest_sha256")}
    manifest["files"] = file_inventory(package_dir)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    validate_manifest(manifest, package_dir)
    target = package_dir / "manifest.json"
    if _linked(target):
        raise ManifestError("Cannot seal a linked manifest")
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(package_dir: Path) -> dict:
    package_dir = Path(package_dir)
    actual = file_inventory(package_dir)
    try:
        manifest = read_json(package_file(package_dir, "manifest.json"))
    except ContractError as exc:
        raise ManifestError(str(exc)) from exc
    validate_manifest(manifest, package_dir)
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise ManifestError("Manifest metadata hash mismatch")
    if actual != manifest["files"]:
        added = sorted(actual.keys() - manifest["files"].keys())
        removed = sorted(manifest["files"].keys() - actual.keys())
        modified = sorted(k for k in actual.keys() & manifest["files"].keys() if actual[k] != manifest["files"][k])
        raise ManifestError(f"Package hash mismatch: added={added}, removed={removed}, modified={modified}")
    return manifest
