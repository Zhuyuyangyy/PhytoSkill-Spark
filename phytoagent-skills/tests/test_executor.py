from pathlib import Path

import pytest

from registry import SkillRegistry
from registry.signer import sign_package
from runtime.executor import SkillExecutor
from sdk.exceptions import ContractError, ManifestError, RegistryError, UnsupportedModeError


def test_executor_validates_before_import(package_factory, tmp_path):
    marker = tmp_path / "executed.txt"
    package = package_factory(source=f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    registry = SkillRegistry(package.parent, require_signature=False)
    registry.register(package)
    executor = SkillExecutor(registry)
    with pytest.raises(ContractError):
        executor.execute(package.name, {})
    with pytest.raises(UnsupportedModeError):
        executor.execute(package.name, {"message": "ok"}, mode="live")
    assert not marker.exists()


def test_executor_rechecks_signed_package_before_import(package, keys):
    sign_package(package, keys[0])
    registry = SkillRegistry(package.parent, trusted_public_key=keys[1])
    registry.register(package)
    (package / "skill.py").write_text("raise RuntimeError('modified')", encoding="utf-8")
    with pytest.raises(ManifestError):
        SkillExecutor(registry).execute(package.name, {"message": "ok"})


def test_single_registration_does_not_visit_siblings_and_is_atomic(package):
    unrelated = package.parent / "unrelated"
    unrelated.mkdir()
    registry = SkillRegistry(package.parent, require_signature=False)
    registry.register(package)
    before = registry.records
    with pytest.raises(RegistryError, match="Duplicate"):
        registry.register(package)
    assert registry.records == before
    with pytest.raises(RegistryError, match="direct child"):
        registry.register(package.parent.parent)


def test_failure_result_preserves_id_without_leaking_arbitrary_exception(package_factory):
    package = package_factory(source="raise RuntimeError('credential=secret-test-value')\n")
    registry = SkillRegistry(package.parent, require_signature=False)
    registry.register(package)
    response = SkillExecutor(registry).call(package.name, {"message": "ok"}, mode="fixture", tool_call_id="id-123")
    assert response["tool_call_id"] == "id-123"
    assert response["status"] == "failed" and response["data"] is None
    assert response["error"]["code"] == "SkillExecutionError"
    assert "secret-test-value" not in str(response)
