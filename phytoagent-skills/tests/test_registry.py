import json

import pytest

from registry import SkillRegistry
from registry.signer import sign_package
from sdk.exceptions import ContractError, RegistryError
from sdk.manifest import seal_manifest, verify_manifest


def test_signed_discovery_exports_actual_input_contract(package, keys):
    sign_package(package, keys[0])
    registry = SkillRegistry(package.parent, trusted_public_key=keys[1])
    records = registry.discover()
    tool = registry.available_tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "contract_probe"
    assert tool["function"]["parameters"] == records[0]["input_schema"]
    assert records[0]["verification"]["signature"]["status"] == "passed"
    assert records[0]["verification"]["evaluated"] == "not_checked"
    assert "synthetic fixture" in registry.instructions("contract_probe")
    registry.validate_input("contract_probe", {"message": "ok"})
    with pytest.raises(ContractError):
        registry.validate_output("contract_probe", {"status": "wrong"})
    tool["function"]["parameters"].clear()
    records[0]["manifest"]["name"] = "mutated"
    assert registry.get("contract_probe")["manifest"]["name"] == "contract_probe"
    assert registry.available_tools[0]["function"]["parameters"]["type"] == "object"


def test_signed_discovery_is_default_and_unsigned_requires_explicit_policy(package, keys):
    with pytest.raises(RegistryError, match="pinned"):
        SkillRegistry(package.parent).discover()
    with pytest.raises(RegistryError, match="manifest.sig"):
        SkillRegistry(package.parent, trusted_public_key=keys[1]).discover()
    record = SkillRegistry(package.parent, require_signature=False).discover()[0]
    assert record["verification"]["signature"]["status"] == "not_checked"


def test_discovery_does_not_execute_signed_python(package_factory, keys, tmp_path):
    marker = tmp_path / "unexpected-execution.txt"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('must not import')\n"
    package = package_factory(source=source)
    sign_package(package, keys[0])
    registry = SkillRegistry(package.parent, trusted_public_key=keys[1])
    registry.discover()
    registry.instructions("contract_probe")
    assert not marker.exists()
    assert registry.available_tools[0]["function"]["name"] == "contract_probe"


def test_failed_discovery_does_not_publish_partial_snapshot(package_factory):
    package = package_factory("a_first")
    registry = SkillRegistry(package.parent, require_signature=False)
    registry.discover()
    original = registry.records
    package_factory("b_valid")
    bad = package.parent / "z_invalid"
    bad.mkdir()
    with pytest.raises(RegistryError, match="z_invalid"):
        registry.discover()
    assert registry.records == original
    assert [t["function"]["name"] for t in registry.available_tools] == ["a_first"]


def test_directory_identity_and_instruction_metadata_are_checked(package):
    manifest = verify_manifest(package)
    manifest["name"] = "renamed"
    seal_manifest(package, manifest)
    with pytest.raises(RegistryError, match="directory name"):
        SkillRegistry(package.parent, require_signature=False).discover()
    manifest["name"] = package.name
    (package / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    seal_manifest(package, manifest)
    with pytest.raises(RegistryError, match="SKILL.md"):
        SkillRegistry(package.parent, require_signature=False).discover()


def test_key_cannot_be_trusted_from_scanned_packages(package, keys):
    public = package.parent / "untrusted.public.pem"
    public.write_bytes(keys[1].read_bytes())
    with pytest.raises(RegistryError, match="outside"):
        SkillRegistry(package.parent, trusted_public_key=public).discover()


def test_consuming_instructions_rechecks_original_discovery(package):
    registry = SkillRegistry(package.parent, require_signature=False)
    registry.discover()
    manifest = verify_manifest(package)
    manifest["version"] = "0.2.0"
    seal_manifest(package, manifest)
    with pytest.raises(RegistryError, match="rediscover"):
        registry.instructions(package.name)


def test_config_resolves_paths_relative_to_file_and_rejects_unknown_fields(package, keys, tmp_path):
    sign_package(package, keys[0])
    config_path = tmp_path / "registry.json"
    config = {"version": 1, "skills_dir": "skills", "require_signature": True,
              "trusted_public_key": "trust/publisher.public.pem"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    registry = SkillRegistry.from_config(config_path)
    assert registry.discover()[0]["manifest"]["name"] == package.name
    config["silent_fallback"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError):
        SkillRegistry.from_config(config_path)


def test_empty_directory_and_unknown_skills(tmp_path):
    registry = SkillRegistry(tmp_path, require_signature=False)
    assert registry.discover() == []
    assert registry.available_tools == []
    with pytest.raises(RegistryError, match="Unknown"):
        registry.get("missing")


def test_portable_hyphenated_directory_preserves_function_identity(package):
    portable = package.with_name("contract-probe")
    package.rename(portable)
    registry = SkillRegistry(portable.parent, require_signature=False)
    registry.discover()
    assert registry.catalog[0]["name"] == "contract_probe"
    assert registry.get("contract_probe")["package_dir"] == str(portable.resolve())


def test_alias_directories_cannot_register_the_same_function_twice(package):
    import shutil
    shutil.copytree(package, package.with_name("contract-probe"))
    registry = SkillRegistry(package.parent, require_signature=False)
    with pytest.raises(RegistryError, match="Duplicate"):
        registry.discover()
    assert registry.records == []
