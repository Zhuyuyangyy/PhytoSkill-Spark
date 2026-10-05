"""Compatibility shim for the observation cache.

The cache is hardware-independent — it caches *results*, wherever they came
from — so it now lives in ``backends.observation_cache``, next to the other
backend-agnostic code. It sat here originally only because DGX was the only
backend that existed.

Everything is re-exported, so ``from dgx.cache import ObservationCache`` keeps
working. ``cached_vision`` keeps its original ``runner`` signature for the
callers that still hand it a DGX client; new code should use
``backends.observation_cache.cached_observation``, which takes a backend.
"""

from __future__ import annotations

from backends.observation_cache import DEFAULT_CACHE_DIR
from backends.observation_cache import ObservationCache
from backends.observation_cache import cache_key
from backends.observation_cache import image_digest

__all__ = ["DEFAULT_CACHE_DIR", "ObservationCache", "cache_key", "cached_vision",
           "image_digest"]


def cached_vision(client, *, cache: ObservationCache, image_bytes: bytes,
                  species: str, model: str, mode: str, runner, timeout: int = 900,
                  prompt: str = "") -> dict:
    """Run one vision inference, or serve it from the cache.

    ``runner`` is the mode-specific callable (``dgx.vision.run_vision`` or
    ``dgx.herb_vision.run_vision``). The returned record carries a ``cache``
    field saying whether it was computed now, so a report can never present a
    cached observation as a fresh measurement.

    ``prompt`` must be the exact text the runner will send. Leaving it empty is
    allowed, but then a prompt edit will not invalidate the cache — which is why
    callers that care pass it in.
    """
    key = cache_key(image_bytes=image_bytes, model=model, mode=mode,
                    species=species, prompt=prompt)
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
