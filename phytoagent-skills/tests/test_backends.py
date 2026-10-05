"""Tests for the execution-backend layer.

All offline. Network is replaced by an injected poster; the SSH library is never
contacted and its absence is simulated by nulling the module attribute.

The claim under test is the whole point of this layer: **inference must not
require one particular machine.** If any of these tests needed a GPU, a
credential or a socket, the abstraction would have failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backends.base import BackendIdentity, BackendUnavailable
from backends.ollama import OllamaBackend
from backends.registry import available, normalise, resolve
from backends.replay import ReplayBackend, ReplayMiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"

LEAF_REPLY = json.dumps({
    "image_usable": True,
    "regions": [{"phenotype": "leaf_yellowing", "label": "yellow patch",
                 "bbox": [0.1, 0.2, 0.5, 0.6], "confidence": 0.8}],
}, ensure_ascii=False)

# A reply that names a cause. The parser must drop it, not record it.
DIAGNOSIS_REPLY = json.dumps({
    "image_usable": True,
    "regions": [{"phenotype": "leaf_spot", "label": "病原菌侵染导致的病斑",
                 "bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.9}],
}, ensure_ascii=False)


def poster_for(reply: str, **extra):
    def poster(url: str, payload: dict, timeout: float) -> dict:
        assert url.endswith("/api/generate"), url
        return {"response": reply, "eval_count": 17, "load_duration": 1000,
                "total_duration": 2_000_000_000, **extra}

    return poster


def make_backend(reply: str = LEAF_REPLY, **kwargs) -> OllamaBackend:
    return OllamaBackend("http://127.0.0.1:11434",
                         poster=poster_for(reply),
                         getter=lambda *a: None, **kwargs)


# ── resolution ──────────────────────────────────────────────────────────────


def test_an_unset_backend_fails_instead_of_guessing():
    """Two machines must never silently pick different hardware."""
    with pytest.raises(BackendUnavailable, match="no vision backend configured"):
        resolve(None, environ={})


def test_backend_aliases_are_accepted():
    assert normalise("dgx")[0] == "dgx_spark"
    assert normalise("dgx-spark")[0] == "dgx_spark"
    assert normalise("spark")[0] == "dgx_spark"
    assert normalise("local_cpu")[0] == "local_cpu"


def test_an_http_endpoint_routes_to_ollama_whatever_the_kind():
    backend = resolve("remote_gpu@http://10.0.0.5:11434", environ={})
    assert isinstance(backend, OllamaBackend)
    assert backend.identity.kind == "remote_gpu"
    assert backend.identity.endpoint == "http://10.0.0.5:11434"


def test_an_unknown_backend_name_is_rejected():
    with pytest.raises(BackendUnavailable, match="unknown vision backend"):
        resolve("quantum", environ={})


def test_availability_is_reported_without_connecting():
    assert available("local_cpu", environ={})["ok"] is True
    report = available(None, environ={})
    assert report["ok"] is False and "no vision backend" in report["error"]


# ── ollama backend ──────────────────────────────────────────────────────────


def test_the_ollama_backend_parses_a_reply_without_touching_a_gpu():
    backend = make_backend()
    record = backend.run_vision(image_bytes=b"bytes", species="黄芪",
                                model=MODEL, mode="live")
    assert record["image_usable"] is True
    assert len(record["regions"]) == 1
    assert record["regions"][0]["phenotype"] == "leaf_yellowing"
    assert record["cache"] == "miss"
    # No GPU was observed, so none is claimed.
    assert record["gpu"] is None
    assert record["backend"]["gpu_observed"] is False


def test_a_region_that_names_a_cause_is_dropped():
    backend = make_backend(DIAGNOSIS_REPLY)
    record = backend.run_vision(image_bytes=b"bytes", species="黄芪",
                                model=MODEL, mode="live")
    assert record["regions"] == []


def test_an_empty_reply_is_an_honest_empty_result():
    backend = make_backend("")
    record = backend.run_vision(image_bytes=b"bytes", species="黄芪",
                                model=MODEL, mode="live")
    assert record["image_usable"] is False
    assert record["regions"] == []


def test_a_transport_failure_is_backend_unavailable_not_a_guess():
    def failing(url, payload, timeout):
        raise BackendUnavailable("ollama endpoint unreachable: connection refused")

    backend = OllamaBackend("http://127.0.0.1:11434", poster=failing,
                            getter=lambda *a: None, probe_gpu=False)
    with pytest.raises(BackendUnavailable, match="unreachable"):
        backend.run_vision(image_bytes=b"bytes", species="黄芪",
                           model=MODEL, mode="live")


def test_an_unsupported_mode_is_refused():
    with pytest.raises(BackendUnavailable, match="does not support mode"):
        make_backend().run_vision(image_bytes=b"bytes", species="黄芪",
                                  model=MODEL, mode="fixture")


def test_the_backend_identity_travels_with_the_result():
    backend = make_backend()
    record = backend.run_vision(image_bytes=b"bytes", species="黄芪",
                                model=MODEL, mode="live")
    assert record["backend"] == BackendIdentity(
        name="ollama", kind="local_cpu",
        endpoint="http://127.0.0.1:11434", gpu_observed=False).to_dict()


# ── replay backend ──────────────────────────────────────────────────────────


def test_replay_returns_a_recorded_observation():
    source = PROJECT_ROOT / "artifacts/dgx/vision-runs.json"
    if not source.is_file():  # pragma: no cover - artifacts always ship
        pytest.skip("recorded runs not present")
    backend = ReplayBackend()
    image = (PROJECT_ROOT / "artifacts/dgx/synthetic-leaf-01.png").read_bytes()
    record = backend.run_vision(image_bytes=image, species="x",
                                model=MODEL, mode="live")
    assert record["cache"] == "replay"
    assert record["replay"]["matched_by"] == "image_sha256"
    # The timing belongs to the original run, and the record says so.
    assert record["replay"]["latency_is_replayed"] is True


def test_replay_refuses_an_image_it_has_never_seen():
    """A replay miss must be a hard failure, never a nearest-neighbour guess."""
    backend = ReplayBackend()
    with pytest.raises(ReplayMiss, match="will not invent"):
        backend.run_vision(image_bytes=b"never-recorded", species="x",
                           model=MODEL, mode="live")


# ── the SSH library is optional ─────────────────────────────────────────────


def test_the_ssh_library_is_only_required_when_connecting(monkeypatch):
    """Importing the DGX modules must not demand paramiko.

    The prompt and parser there are hardware-independent and are reused by
    non-SSH backends; requiring an SSH library to read them would reintroduce
    exactly the coupling this layer exists to remove.
    """
    import dgx.client as client_module

    monkeypatch.setattr(client_module, "paramiko", None)
    client = client_module.DgxClient(
        {"host": "h", "port": 22, "user": "u", "password": "p"})
    with pytest.raises(ModuleNotFoundError, match="paramiko"):
        client.connect()


def test_the_ollama_backend_imports_without_the_ssh_library(monkeypatch):
    """Simulates a machine with no paramiko installed."""
    import dgx.client as client_module

    monkeypatch.setattr(client_module, "paramiko", None)
    # Constructing and using an ollama backend must not care.
    backend = make_backend()
    record = backend.run_vision(image_bytes=b"bytes", species="黄芪",
                                model=MODEL, mode="live")
    assert record["backend"]["kind"] == "local_cpu"


# ── the Skill itself ────────────────────────────────────────────────────────


def _load_plant_vision():
    import importlib.util
    import sys

    path = PROJECT_ROOT / "skills" / "plant_vision" / "skill.py"
    spec = importlib.util.spec_from_file_location("_pv_backend_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_skill_runs_real_inference_through_an_injected_backend(tmp_path):
    """End to end, offline: no GPU, no SSH, no network."""
    module = _load_plant_vision()
    image = tmp_path / "leaf.jpg"
    image.write_bytes(b"pretend jpeg bytes")

    skill = module.PlantVisionSkill(
        PROJECT_ROOT / "skills" / "plant_vision",
        backend=make_backend(),
        cache_dir=tmp_path / "cache")

    result = skill.run({"image_path": str(image), "species": "黄芪",
                        "case_id": "c1"}, mode="live")

    assert result["status"] == "success"
    assert len(result["observations"]) == 1
    provenance = result["provenance"]
    # The hardware field is driven by what the backend actually reported.
    assert provenance["dgx_hardware_used"] is False
    assert provenance["backend"]["kind"] == "local_cpu"
    assert provenance["model_called"] is True
    assert any("后端" in item for item in result["limitations"])


def test_the_skill_does_not_need_a_backend_for_fixture_mode(tmp_path):
    """A signed package copied anywhere must still answer its own fixture case."""
    module = _load_plant_vision()
    skill = module.PlantVisionSkill(
        PROJECT_ROOT / "skills" / "plant_vision", cache_dir=tmp_path / "cache")
    payload = json.loads(
        (PROJECT_ROOT / "skills" / "plant_vision" / "examples" / "request.json"
         ).read_text(encoding="utf-8"))
    result = skill.run(payload, mode="fixture")
    assert result["status"] == "success"
