"""Replay backend: serve observations that were recorded on real hardware.

Why this exists: once the project is hardware-independent, its end-to-end chain
must still be runnable *without* any hardware. Replay makes that possible
honestly — it returns a measurement that was actually taken, on a named machine,
by a named model, for a named image.

Why it is strict: a replay miss is a hard failure. A backend that quietly
returned the nearest observation, or an empty one, would turn every downstream
metric into fiction. The whole point of the recorded artifacts is that they are
the only observations anyone is allowed to claim.

What it is not: a performance measurement. ``latency_ms`` in a replayed record
describes the run that produced it, on hardware that may no longer exist. Every
replayed record says so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backends.base import KIND_REPLAY, BackendIdentity, BackendUnavailable, VisionBackend

DEFAULT_SOURCE = Path("artifacts/dgx/vision-runs.json")
DEFAULT_IMAGE_DIR = Path("artifacts/dgx")


class ReplayMiss(BackendUnavailable):
    """No recorded observation matches this image, model and mode."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReplayBackend(VisionBackend):
    """Return previously recorded observations, keyed by image identity."""

    def __init__(self, records: list[dict] | None = None, *,
                 source: str | Path | None = None,
                 image_dir: str | Path | None = None,
                 name: str = "replay"):
        self._name = name
        self._source = Path(source) if source is not None else DEFAULT_SOURCE
        self._image_dir = Path(image_dir) if image_dir is not None else DEFAULT_IMAGE_DIR
        self._records = records if records is not None else self._load()
        self._by_digest: dict[tuple[str, str], dict] = {}
        self._by_name: dict[tuple[str, str], dict] = {}
        self._index()

    # ── loading ───────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self._source.is_file():
            raise BackendUnavailable(f"replay source not found: {self._source}")
        payload = json.loads(self._source.read_text(encoding="utf-8"))
        runs = payload.get("runs") if isinstance(payload, dict) else payload
        if not isinstance(runs, list):
            raise BackendUnavailable(f"replay source has no runs list: {self._source}")
        return runs

    def _index(self) -> None:
        """Key each record by image content, falling back to file name.

        Content is the reliable key. The name fallback exists because a recorded
        run may reference an image that is no longer on disk; that record stays
        reachable by name but can never be matched by content, which is the
        honest outcome.
        """
        for record in self._records:
            model = str(record.get("model") or "")
            image_file = record.get("image_file")
            if image_file:
                self._by_name[(str(image_file), model)] = record
                path = self._image_dir / str(image_file)
                if path.is_file():
                    self._by_digest[(_sha256(path.read_bytes()), model)] = record

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(name=self._name, kind=KIND_REPLAY,
                               endpoint=str(self._source), gpu_observed=False)

    @property
    def record_count(self) -> int:
        return len(self._records)

    # ── inference ─────────────────────────────────────────────────────────

    def run_vision(self, *, image_bytes: bytes, species: str, model: str,
                   mode: str, timeout: int = 600, prompt: str | None = None) -> dict:
        digest = _sha256(image_bytes)
        record = self._by_digest.get((digest, model))
        matched_by = "image_sha256"
        if record is None:
            # The caller may have passed the recorded file under a new path.
            for (name, rec_model), candidate in self._by_name.items():
                path = self._image_dir / name
                if rec_model == model and path.is_file() and _sha256(path.read_bytes()) == digest:
                    record = candidate
                    matched_by = "image_sha256"
                    break
        if record is None:
            raise ReplayMiss(
                f"no recorded observation for image sha256 {digest[:12]}… "
                f"with model {model!r}; replay will not invent one")

        served = {key: value for key, value in record.items()
                  if key not in ("image_file",)}
        served["backend"] = self.identity.to_dict()
        served["cache"] = "replay"
        served["replay"] = {
            "matched_by": matched_by,
            "source": str(self._source),
            "image_sha256": digest,
            # The timing belongs to the original run, not to this call.
            "latency_is_replayed": True,
            "recorded_gpu": record.get("gpu"),
        }
        return served
