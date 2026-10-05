"""SSH backend: run inference on a remote node that hosts an ollama server.

This wraps the existing DGX Spark path. It is deliberately a thin wrapper: the
SSH-and-script route is what produced every recorded real observation in
``artifacts/dgx/``, and reimplementing it here would risk changing behaviour
that has already been measured.

What changes is the framing. The node is no longer *the* execution environment,
it is one reachable endpoint among several. ``dgx_spark`` is a ``kind`` string
here, not the identity of the project.

Requires paramiko. Its absence raises :class:`BackendUnavailable` at call time,
not at import time, so the rest of the package stays importable without it.
"""

from __future__ import annotations

from pathlib import Path

from backends.base import (BackendIdentity, BackendUnavailable, VisionBackend,
                           annotate)
from backends.base import KIND_DGX_SPARK, KIND_REMOTE_GPU
from dgx.client import DgxClient, load_credentials


class SshOllamaBackend(VisionBackend):
    """Run vision inference by executing a script on a remote node over SSH."""

    def __init__(self, credentials: dict | None = None, *,
                 env_file: str | Path | None = None,
                 kind: str = KIND_DGX_SPARK, name: str = "ssh-ollama",
                 label: str | None = None):
        self._credentials = credentials
        self._env_file = env_file
        self._kind = kind
        self._name = name
        self._label = label

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def identity(self) -> BackendIdentity:
        host = self._label or self._describe_host()
        # A GPU is what this backend is for; it is reported as observed only
        # after nvidia-smi actually answers (see run_vision).
        return BackendIdentity(name=self._name, kind=self._kind, endpoint=host,
                               gpu_observed=False)

    def _describe_host(self) -> str:
        try:
            creds = self._resolve()
        except BackendUnavailable:
            return "unconfigured"
        return f"ssh://{creds.get('user')}@{creds.get('host')}:{creds.get('port')}"

    def _resolve(self) -> dict:
        if self._credentials is not None:
            return self._credentials
        try:
            return load_credentials(self._env_file)
        except ValueError as exc:
            raise BackendUnavailable(f"SSH credentials unavailable: {exc}") from exc
        except ModuleNotFoundError as exc:
            raise BackendUnavailable(str(exc)) from exc

    # ── inference ─────────────────────────────────────────────────────────

    def run_vision(self, *, image_bytes: bytes, species: str, model: str,
                   mode: str, timeout: int = 600, prompt: str | None = None) -> dict:
        from dgx.herb_vision import run_vision as run_herb
        from dgx.vision import run_vision as run_leaf

        if mode == "live":
            runner = run_leaf
        elif mode == "herb":
            runner = run_herb
        else:
            raise BackendUnavailable(f"ssh backend does not support mode {mode!r}")

        credentials = self._resolve()
        try:
            with DgxClient(credentials) as client:
                record = runner(client, image_bytes=image_bytes, species=species,
                                model=model, timeout=timeout)
        except ModuleNotFoundError as exc:
            raise BackendUnavailable(str(exc)) from exc

        identity = BackendIdentity(
            name=self._name, kind=self._kind,
            endpoint=self._describe_host(),
            gpu_observed=bool(record.get("gpu")))
        record = dict(record)
        record["backend"] = identity.to_dict()
        record["cache"] = "miss"
        return record


def dgx_spark_backend(env_file: str | Path | None = None) -> SshOllamaBackend:
    """Convenience constructor for the DGX Spark node specifically."""
    return SshOllamaBackend(env_file=env_file, kind=KIND_DGX_SPARK,
                            name="dgx-spark", label="dgx-spark")


def remote_gpu_backend(env_file: str | Path | None = None, *,
                       name: str = "remote-gpu") -> SshOllamaBackend:
    """Any rented or lab GPU node reachable over SSH. Same transport, no brand."""
    return SshOllamaBackend(env_file=env_file, kind=KIND_REMOTE_GPU, name=name)
