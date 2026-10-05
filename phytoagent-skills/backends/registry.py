"""Resolve a backend name into a backend instance.

One environment variable decides where inference runs, so a machine swap is a
configuration change and not a code change. That is the whole point of the
abstraction; if resolving a backend required editing a Skill, the abstraction
would have achieved nothing.

Defaults to *failing* rather than guessing. An unset backend would otherwise
silently pick whatever the developer's laptop had, and two people would get
different numbers from the same commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from backends.base import (KIND_DGX_SPARK, KIND_LOCAL_CPU, KIND_LOCAL_CUDA,
                           KIND_REMOTE_GPU, KIND_REPLAY, BackendUnavailable,
                           VisionBackend)
from backends.ollama import DEFAULT_ENDPOINT, OllamaBackend
from backends.replay import ReplayBackend
from backends.ssh import SshOllamaBackend

ENV_BACKEND = "PHYTO_VISION_BACKEND"
ENV_OLLAMA_ENDPOINT = "PHYTO_OLLAMA_ENDPOINT"

# Accepted spellings. Kept permissive on purpose: "dgx", "dgx_spark" and
# "dgx-spark" all mean the same node, and rejecting a hyphen would be pedantry
# that costs the user a debugging session.
ALIASES = {
    "local": KIND_LOCAL_CPU,
    "local_cpu": KIND_LOCAL_CPU,
    "cpu": KIND_LOCAL_CPU,
    "ollama": KIND_LOCAL_CPU,
    "local_cuda": KIND_LOCAL_CUDA,
    "cuda": KIND_LOCAL_CUDA,
    "gpu": KIND_LOCAL_CUDA,
    "remote": KIND_REMOTE_GPU,
    "remote_gpu": KIND_REMOTE_GPU,
    "ssh": KIND_REMOTE_GPU,
    "dgx": KIND_DGX_SPARK,
    "dgx_spark": KIND_DGX_SPARK,
    "dgx-spark": KIND_DGX_SPARK,
    "spark": KIND_DGX_SPARK,
    "replay": KIND_REPLAY,
    "recorded": KIND_REPLAY,
}

CHOICES = ("local_cpu", "local_cuda", "remote_gpu", "dgx_spark", "replay")


def normalise(spec: str) -> tuple[str, str | None]:
    """Split ``"kind"`` or ``"kind@target"`` into its two parts."""
    spec = spec.strip()
    if not spec:
        raise BackendUnavailable("empty backend spec")
    kind, _, target = spec.partition("@")
    kind = ALIASES.get(kind.strip().lower(), kind.strip().lower())
    return kind, (target.strip() or None)


def resolve(spec: str | None = None, *, environ: dict | None = None,
            env_file: str | Path | None = None) -> VisionBackend:
    """Build the backend named by ``spec``, or by ``PHYTO_VISION_BACKEND``.

    Raises :class:`BackendUnavailable` when nothing is configured: an inference
    that silently ran in the wrong place would be worse than one that refuses.
    """
    environ = os.environ if environ is None else environ
    raw = spec if spec is not None else environ.get(ENV_BACKEND, "")
    if not raw.strip():
        raise BackendUnavailable(
            f"no vision backend configured; set {ENV_BACKEND} to one of "
            f"{', '.join(CHOICES)} (e.g. {ENV_BACKEND}=local_cpu). "
            "Replay is the offline option: it serves recorded observations and "
            "fails loudly on anything it has not seen.")

    kind, target = normalise(raw)

    if kind == KIND_REPLAY:
        return ReplayBackend(source=target)

    if kind in (KIND_LOCAL_CPU, KIND_LOCAL_CUDA, KIND_REMOTE_GPU):
        # An explicit HTTP endpoint means "talk to ollama directly", whatever
        # the kind is: a remote GPU box that exposes its ollama port needs no
        # SSH at all. Without one, local kinds use the resident server and a
        # remote kind falls back to the SSH path.
        if target and target.startswith(("http://", "https://")):
            return OllamaBackend(target, kind=kind)
        if kind in (KIND_LOCAL_CPU, KIND_LOCAL_CUDA):
            endpoint = environ.get(ENV_OLLAMA_ENDPOINT) or DEFAULT_ENDPOINT
            return OllamaBackend(endpoint, kind=kind)
        return SshOllamaBackend(env_file=env_file, kind=KIND_REMOTE_GPU,
                                name="remote-gpu")

    if kind == KIND_DGX_SPARK:
        return SshOllamaBackend(env_file=env_file, kind=KIND_DGX_SPARK,
                                name="dgx-spark", label="dgx-spark")

    raise BackendUnavailable(
        f"unknown vision backend {raw!r}; expected one of {', '.join(CHOICES)}")


def available(spec: str | None = None, *, environ: dict | None = None) -> dict:
    """Report whether a backend can be constructed, without connecting.

    Used by preflight and by diagnostics. Constructing an SSH backend does not
    open a socket, so this stays cheap and offline-safe for every kind.
    """
    try:
        backend = resolve(spec, environ=environ)
    except BackendUnavailable as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "backend": backend.identity.to_dict()}


__all__ = ["ALIASES", "CHOICES", "ENV_BACKEND", "ENV_OLLAMA_ENDPOINT",
           "available", "normalise", "resolve"]
