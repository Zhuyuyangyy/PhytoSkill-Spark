"""Content-addressed cache for DGX vision observations.

One inference costs 12-167 seconds and contends with other processes on the node.
An A/B run makes the same call twice (once per arm) and repeats it on every rerun,
so without a cache a single experiment would spend most of its wall-clock time
re-deriving observations it already has.

The cache is keyed by the **image bytes**, not the path: two directories holding
the same photograph must share an entry, and a edited image must not. The model,
the mode and the species are part of the key too, because each of them changes
what is observed.

Cache entries are written atomically. A partially written file must never be read
as a result, because a truncated observation is indistinguishable from a real one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_CACHE_DIR = Path("artifacts/dgx/cache")


def image_digest(image_bytes: bytes) -> str:
    """SHA-256 of the exact bytes, so the key follows the content."""
    return hashlib.sha256(image_bytes).hexdigest()


def cache_key(*, image_bytes: bytes, model: str, mode: str, species: str) -> str:
    """Everything that changes the answer, hashed into one key."""
    payload = "|".join([image_digest(image_bytes), model, mode, species])
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


def cached_vision(client, *, cache: ObservationCache, image_bytes: bytes,
                  species: str, model: str, mode: str, runner, timeout: int = 900) -> dict:
    """Run one vision inference, or serve it from the cache.

    ``runner`` is the mode-specific callable (``dgx.vision.run_vision`` or
    ``dgx.herb_vision.run_vision``). The returned record carries a ``cache``
    field saying whether it was computed now, so a report can never present a
    cached observation as a fresh measurement.
    """
    key = cache_key(image_bytes=image_bytes, model=model, mode=mode, species=species)
    record = cache.get(key)
    if record is not None:
        served = dict(record)
        served["cache"] = "hit"
        return served
    record = runner(client, image_bytes=image_bytes, species=species,
                    model=model, timeout=timeout)
    served = dict(record)
    served["cache"] = "miss"
    cache.put(key, record)
    return served
