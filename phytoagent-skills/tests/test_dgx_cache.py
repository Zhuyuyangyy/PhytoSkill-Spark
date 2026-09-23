"""Tests for the content-addressed observation cache.

Offline: no GPU, no SSH. The cache's whole job is to decide correctly whether a
record can be reused, so those decisions are what get tested.
"""

from __future__ import annotations

import json

import pytest

from dgx.cache import (ObservationCache, cache_key, cached_vision, image_digest)

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


@pytest.fixture
def cache(tmp_path):
    return ObservationCache(tmp_path / "cache")


# ── keying ───────────────────────────────────────────────────────────────────


def test_the_key_follows_the_image_content_not_the_path():
    a = cache_key(image_bytes=b"same bytes", model=MODEL, mode="herb", species="黄芪")
    b = cache_key(image_bytes=b"same bytes", model=MODEL, mode="herb", species="黄芪")
    assert a == b


def test_different_bytes_get_different_keys():
    a = cache_key(image_bytes=b"first", model=MODEL, mode="herb", species="黄芪")
    b = cache_key(image_bytes=b"second", model=MODEL, mode="herb", species="黄芪")
    assert a != b


@pytest.mark.parametrize("field,value", [
    ("model", "other-model"),
    ("mode", "live"),
    ("species", "人参"),
])
def test_every_input_that_changes_the_answer_is_part_of_the_key(field, value):
    base = {"image_bytes": b"img", "model": MODEL, "mode": "herb", "species": "黄芪"}
    other = dict(base, **{field: value})
    assert cache_key(**base) != cache_key(**other)


def test_the_digest_is_a_sha256_hex():
    digest = image_digest(b"anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ── read / write ─────────────────────────────────────────────────────────────


def test_a_miss_returns_none(cache):
    assert cache.get("nonexistent") is None
    assert cache.stats()["misses"] == 1


def test_a_put_then_get_round_trips(cache):
    record = {"regions": [{"phenotype": "colour_amber"}], "image_usable": True}
    cache.put("key-1", record)
    assert cache.get("key-1") == record


def test_stats_count_hits_and_misses(cache):
    cache.get("absent")
    cache.put("k", {"regions": []})
    cache.get("k")
    stats = cache.stats()
    assert stats == {"entries": 1, "hits": 1, "misses": 1, "hit_rate": 0.5}


def test_a_corrupt_entry_is_treated_as_absent(cache):
    (cache.cache_dir / "broken.json").write_text("{not json", encoding="utf-8")
    assert cache.get("broken") is None
    assert cache.stats()["misses"] == 1


def test_an_entry_without_regions_is_treated_as_absent(cache):
    """A record with no regions field is not an observation; reusing it would be
    presenting a truncated result as a real one."""
    (cache.cache_dir / "shaped.json").write_text(json.dumps({"image_usable": True}),
                                                 encoding="utf-8")
    assert cache.get("shaped") is None


def test_no_temporary_files_are_left_behind(cache):
    cache.put("k", {"regions": []})
    assert not list(cache.cache_dir.glob(".tmp-*"))


def test_entries_are_addressable_json_files(cache):
    cache.put("abc123", {"regions": []})
    assert (cache.cache_dir / "abc123.json").is_file()


# ── the memoising wrapper ────────────────────────────────────────────────────


class FakeClient:
    """Records how many times the runner was actually invoked."""


def _fake_runner(calls: list):
    def run(client, *, image_bytes, species, model, timeout):
        calls.append({"species": species, "model": model, "timeout": timeout})
        return {"regions": [{"phenotype": "colour_amber", "label": "bark"}],
                "image_usable": True, "latency_ms": 1234.5,
                "gpu": {"name": "NVIDIA GB10"}}
    return run


def test_the_second_identical_call_does_not_re_invoke_the_runner(cache):
    calls: list = []
    runner = _fake_runner(calls)
    for _ in range(3):
        record = cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                               species="黄芪", model=MODEL, mode="herb", runner=runner)
        assert record["regions"]
    assert len(calls) == 1
    assert cache.stats()["hits"] == 2


def test_a_cache_hit_is_labelled_as_such(cache):
    """A cached observation must never be presented as a fresh measurement."""
    runner = _fake_runner([])
    first = cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                          species="黄芪", model=MODEL, mode="herb", runner=runner)
    second = cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                           species="黄芪", model=MODEL, mode="herb", runner=runner)
    assert first["cache"] == "miss"
    assert second["cache"] == "hit"


def test_a_hit_preserves_the_measured_fields(cache):
    """The point of caching is that the second caller sees the same evidence."""
    runner = _fake_runner([])
    first = cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                          species="黄芪", model=MODEL, mode="herb", runner=runner)
    second = cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                           species="黄芪", model=MODEL, mode="herb", runner=runner)
    assert second["regions"] == first["regions"]
    assert second["latency_ms"] == first["latency_ms"]
    assert second["gpu"] == first["gpu"]


def test_the_runner_receives_the_timeout(cache):
    calls: list = []
    cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                  species="黄芪", model=MODEL, mode="herb",
                  runner=_fake_runner(calls), timeout=900)
    assert calls[0]["timeout"] == 900


def test_changing_the_mode_re_runs_the_inference(cache):
    """A leaf observation and a herb observation of one image are different facts."""
    calls: list = []
    runner = _fake_runner(calls)
    for mode in ("herb", "live", "herb"):
        cached_vision(FakeClient(), cache=cache, image_bytes=b"img",
                      species="黄芪", model=MODEL, mode=mode, runner=runner)
    assert len(calls) == 2
