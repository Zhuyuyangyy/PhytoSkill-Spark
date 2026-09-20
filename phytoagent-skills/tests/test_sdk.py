import json

import pytest

from sdk import BaseSkill, ContractError, ManifestError
from sdk.exceptions import UnsupportedModeError
from sdk.manifest import seal_manifest, verify_manifest
from sdk.schema import check_schema, load_schema, package_file, read_json, validate_payload


class Probe(BaseSkill):
    def run(self, payload, *, mode):
        message = payload.pop("message")
        return {"status": "success", "mode": mode, "echo": message, "data_origin": "synthetic_fixture"}


def test_execute_validates_contract_and_does_not_mutate_callers(package):
    skill = Probe(package)
    payload = {"message": "hello"}
    result = skill.execute(payload)
    assert result["echo"] == "hello"
    assert payload == {"message": "hello"}
    metadata = skill.metadata
    assert metadata["skill_name"] == "contract_probe"
    metadata["input_schema"]["properties"].clear()
    manifest = skill.manifest
    manifest["name"] = "changed"
    assert skill.manifest["name"] == "contract_probe"
    assert "message" in skill.metadata["input_schema"]["properties"]


@pytest.mark.parametrize("payload", [None, {}, [], {"message": ""}, {"message": 2},
                                     {"message": "ok", "extra": True}, {"message": float("nan")},
                                     {"message": ("tuple",)}, {1: "not a string key"}])
def test_invalid_input_never_enters_skill(package, payload):
    class MustNotRun(Probe):
        def run(self, payload, *, mode):
            pytest.fail("Invalid input reached skill code")
    with pytest.raises(ContractError):
        MustNotRun(package).execute(payload)


def test_invalid_output_is_rejected(package):
    class BadOutput(Probe):
        def run(self, payload, *, mode):
            return {"status": "invented"}
    with pytest.raises(ContractError, match="output"):
        BadOutput(package).execute({"message": "ok"})


@pytest.mark.parametrize("mode", ["live", "replay", "anything"])
def test_unsupported_modes_do_not_fall_back_to_fixture(package, mode):
    with pytest.raises(UnsupportedModeError):
        Probe(package).execute({"message": "ok"}, mode=mode)


def test_explicitly_resealed_package_still_requires_skill_reload(package):
    skill = Probe(package)
    metadata = skill.manifest
    metadata["version"] = "0.2.0"
    seal_manifest(package, metadata)
    with pytest.raises(ManifestError, match="reload"):
        skill.execute({"message": "ok"})


def test_runtime_errors_are_not_disguised_as_success(package):
    class FailedSkill(Probe):
        def run(self, payload, *, mode):
            raise RuntimeError("adapter unavailable")
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        FailedSkill(package).execute({"message": "ok"})


def test_relative_package_path_survives_working_directory_change(package, monkeypatch, tmp_path):
    monkeypatch.chdir(package.parent)
    skill = Probe(package.name)
    monkeypatch.chdir(tmp_path)
    assert skill.execute({"message": "ok"})["echo"] == "ok"


@pytest.mark.parametrize("relative", ["../schema.json", "/schema.json", "C:/schema.json", "schema.json:stream",
                                      "sub/../../schema.json", "sub\\schema.json", "./schema.json", "missing.json"])
def test_package_paths_are_contained(package, relative):
    with pytest.raises(ContractError):
        package_file(package, relative)


@pytest.mark.parametrize("schema", [
    {"type": "array"},
    {"type": "object", "required": "not-an-array"},
    {"type": "object", "$ref": "https://example.invalid/schema.json"},
    {"type": "object", "$defs": {"a": {"$ref": "file:///private.json"}}},
    {"type": "object", "$id": "https://example.invalid/schema"},
    {"type": "object", "$dynamicRef": "#thing"},
    {"type": "object", "$ref": "#/missing"},
    {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
    {"type": "object", "description": "not a schema", "$ref": "#/description"},
    {"type": "object", "custom": {"$ref": "https://example.invalid/hidden"}, "$ref": "#/custom"},
    {"type": "object", "custom": {"required": "invalid"}, "$ref": "#/custom"},
    {"type": "object", "properties": {"x": {"$schema": "http://json-schema.org/draft-07/schema#"}}},
])
def test_invalid_and_remote_schemas_fail_during_loading(schema):
    with pytest.raises(ContractError):
        check_schema(schema)


def test_local_definitions_and_format_validation(package):
    schema = {"type": "object", "$defs": {"mail": {"type": "string", "format": "email"}},
              "required": ["email"], "properties": {"email": {"$ref": "#/$defs/mail"}, "$ref": {"type": "string"}}}
    check_schema(schema)
    validate_payload({"email": "person@example.org", "$ref": "literal property"}, schema, label="test")
    with pytest.raises(ContractError):
        validate_payload({"email": "not-email"}, schema, label="test")
    assert load_schema(package, "schema.json#/input")["required"] == ["message"]


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_ambiguous_or_non_json_documents_are_rejected(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(path)


@pytest.mark.parametrize("change", ["metadata", "modified", "added", "removed"])
def test_manifest_detects_changes_without_resealing(package, change):
    original = (package / "manifest.json").read_bytes()
    if change == "metadata":
        value = json.loads(original)
        value["description"] = "modified description"
        (package / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    elif change == "modified":
        with (package / "skill.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# modified\n")
    elif change == "added":
        (package / "unexpected.py").write_text("print('unexpected')", encoding="utf-8")
    else:
        (package / "skill.py").unlink()
    before_verification = (package / "manifest.json").read_bytes()
    with pytest.raises(ManifestError):
        verify_manifest(package)
    assert (package / "manifest.json").read_bytes() == before_verification
