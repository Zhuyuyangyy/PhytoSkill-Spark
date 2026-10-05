"""Content-addressed cache for vision observations.

One inference costs seconds to minutes and may contend with other work on the
machine. A/B runs make the same call once per arm and repeat it on every rerun,
so without a cache an experiment spends most of its wall-clock time
re-deriving observations it already has.

This is hardware-independent: it caches *results*, wherever they came from. It
moved here from ``dgx.cache``, where it lived only because DGX was the sole
backend. ``dgx.cache`` re-exports these symbols, so existing imports keep
working.

Keyed by the **image bytes**, not the path: two directories holding the same
photograph must share an entry, and an edited image must not. Model, mode,
species and prompt are all part of the key too — each changes what is observed.

Entries are written atomically. A partially written file must never be read as
a result, because a truncated observation is indistinguishable from a real one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# Renamed from artifacts/dgx/cache: the cache holds observations from any
# backend, and a DGX-only path would imply otherwise. An existing cache at the
# old location can be moved here by hand; the entries are content-addressed, so
# they remain valid.
DEFAULT_CACHE_DIR = Path("artifacts/vision-cache")


def image_digest(image_bytes: bytes) -> str:
    """SHA-256 of the exact bytes, so the key follows the content."""
    return hashlib.sha256(image_bytes).hexdigest()


def cache_key(*, image_bytes: bytes, model: str, mode: str, species: str,
              prompt: str = "") -> str:
    """Everything that changes the answer, hashed into one key.

    ``prompt`` is part of the key for a reason that cost a real experiment: the
    cache originally keyed on image + model + mode + species, so editing the
    prompt and re-running returned the *old* observations verbatim. A/B on a
    prompt change would then have measured nothing. Any prompt change must miss.
    """
    payload = "|".join([image_digest(image_bytes), model, mode, species,
                        hashlib.sha256(prompt.encode("utf-8")).hexdigest()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ObservationCache:
    """A small on-disk cache of parsed vision observations."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return a cached record, or None. Corrupt entries are treated as absent."""
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self.misses += 1
            return None
        if not isinstance(record, dict) or "regions" not in record:
            self.misses += 1
            return None
        self.hits += 1
        return record

    def put(self, key: str, record: dict) -> None:
        """Write one record atomically, so a reader never sees a partial file."""
        path = self._path(key)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.cache_dir, prefix=".tmp-", delete=False)
        try:
            with handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def stats(self) -> dict:
        entries = sum(1 for path in self.cache_dir.glob("*.json"))
        return {"entries": entries, "hits": self.hits, "misses": self.misses,
                "hit_rate": (round(self.hits / (self.hits + self.misses), 3)
                             if self.hits + self.misses else None)}


def cached_observation(backend, *, cache: ObservationCache, image_bytes: bytes,
                       species: str, model: str, mode: str, timeout: int = 900,
                       prompt: str = "") -> dict:
    """Run one vision inference through ``backend``, or serve it from the cache.

    The returned record carries a ``cache`` field saying whether it was computed
    now, so a report can never present a cached observation as a fresh
    measurement. A replayed record keeps its own ``cache="replay"`` marker.

    ``prompt`` must be the exact text the backend will send; leaving it empty is
    allowed but a prompt edit will then not invalidate the cache.
    """
    key = cache_key(image_bytes=image_bytes, model=model, mode=mode,
                    species=species, prompt=prompt)
    record = cache.get(key)
    if record is not None:
        served = dict(record)
        served["cache"] = "hit"
        return served
    record = backend.run_vision(image_bytes=image_bytes, species=species,
                                model=model, mode=mode, timeout=timeout,
                                prompt=prompt)
    served = dict(record)
    # A replay backend already knows it is replaying; do not overwrite that.
    served.setdefault("cache", "miss")
    if served.get("cache") == "miss":
        cache.put(key, record)
    return served


__all__ = ["DEFAULT_CACHE_DIR", "ObservationCache", "cache_key",
           "cached_observation", "image_digest"]
