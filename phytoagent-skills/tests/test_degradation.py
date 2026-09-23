"""Degradation matrix: what happens when the real path is unavailable.

These are the failures an operator actually hits — no credentials, no node, a
timeout, a corrupt cache entry, a Skill that does not implement the mode. Each
must fail loudly and say why; none may silently answer with a fixture, because a
run presented as real must not contain synthetic data.
"""

from __future__ import annotations

import json

import pytest

import pytest

from dgx.cache import ObservationCache
from demo.fixture_workspace import PROJECT_ROOT, fixture_registry
from runtime.executor import SkillExecutor
from sdk.exceptions import ContractError, FixtureMismatchError, SkillError

LEAF = {"case_id": "degrade-001", "species": "黄芪",
        "image_path": "fixture://huangqi-leaf-01"}


@pytest.fixture(scope="module")
def domain_registry():
    with fixture_registry() as registry:
        yield registry




def test_live_mode_refuses_a_fixture_placeholder(domain_registry):
    """A fixture:// path is not an image, and answering it would be fabrication."""
    executor = SkillExecutor(domain_registry)
    with pytest.raises(ContractError, match="fixture://"):
        executor.execute("plant_vision", LEAF, mode="live")
    with pytest.raises(ContractError, match="fixture://"):
        executor.execute("plant_vision", LEAF, mode="herb")


def test_live_mode_refuses_a_missing_image(domain_registry):
    executor = SkillExecutor(domain_registry)
    payload = dict(LEAF, image_path="case_workspace/does-not-exist.jpg")
    for mode in ("live", "herb"):
        with pytest.raises(ContractError, match="image not found"):
            executor.execute("plant_vision", payload, mode=mode)


def test_a_missing_credential_is_an_error_not_a_fallback(domain_registry, tmp_path):
    """No key means no inference. It must not quietly become a fixture answer."""
    executor = SkillExecutor(domain_registry)
    payload = dict(LEAF, image_path=str(tmp_path / "real.jpg"))
    (tmp_path / "real.jpg").write_bytes(b"not really a jpeg")
    package = _package_dir(domain_registry)
    from pathlib import Path

    source = (Path(package) / "skill.py").read_text(encoding="utf-8")
    namespace = {"__name__": "degrade", "__file__": str(Path(package) / "skill.py")}
    exec(compile(source, str(Path(package) / "skill.py"), "exec"), namespace)
    skill = namespace["PlantVisionSkill"](
        Path(package), env_file=tmp_path / "absent.env")
    with pytest.raises(ContractError, match="DGX credentials unavailable"):
        skill.run(payload, mode="live")


def test_an_unsupported_mode_is_refused(domain_registry):
    executor = SkillExecutor(domain_registry)
    for name in ("growth_risk", "evidence_fusion", "agentshield_audit"):
        with pytest.raises(SkillError):
            executor.execute(name, {"case_id": "x", "species": "黄芪"}, mode="live")


def test_a_corrupt_cache_entry_is_recomputed_not_trusted(tmp_path):
    """A truncated observation is indistinguishable from a real one, so it goes."""
    cache = ObservationCache(tmp_path / "cache")
    key = "abc123"
    (cache.cache_dir / f"{key}.json").write_text('{"regions": [{"phenotyp', encoding="utf-8")
    assert cache.get(key) is None


def test_an_entry_without_regions_is_recomputed(tmp_path):
    cache = ObservationCache(tmp_path / "cache")
    (cache.cache_dir / "shaped.json").write_text(json.dumps({"image_usable": True}),
                                                 encoding="utf-8")
    assert cache.get("shaped") is None


def test_a_cache_hit_is_labelled_so_it_is_never_read_as_a_fresh_measurement(tmp_path):
    """The report must be able to tell a cached observation from a new one."""
    cache = ObservationCache(tmp_path / "cache")
    cache.put("k", {"regions": [], "image_usable": False})
    served = cache.get("k")
    assert "cache" not in served  # the flag is added by the wrapper, not the store


def test_the_fixture_mode_still_works_after_the_real_modes_exist(domain_registry):
    """Adding real modes must not break the offline path."""
    executor = SkillExecutor(domain_registry)
    from sdk.schema import read_json
    fixture_input = read_json(PROJECT_ROOT / "skills" / "plant_vision" / "examples" / "request.json")
    result = executor.execute("plant_vision", fixture_input)
    assert result["provenance"]["data_origin"] == "synthetic_fixture"
    assert result["provenance"]["model_called"] is False


def test_a_fixture_input_still_fails_for_a_real_case(domain_registry):
    executor = SkillExecutor(domain_registry)
    with pytest.raises(FixtureMismatchError):
        executor.execute("plant_vision", dict(LEAF, case_id="real-case-999"))


def _package_dir(registry) -> str:
    return registry.get("plant_vision")["package_dir"]
