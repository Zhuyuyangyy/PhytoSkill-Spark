"""Execution backends: where vision inference runs.

One sentence of intent: **hardware is configuration, not identity.** DGX Spark
was the first backend and gave the project its name; it is now one entry in a
list, and the project runs without it.

Quick start:

    from backends.registry import resolve
    backend = resolve("local_cpu")          # ollama on this machine
    backend = resolve("ollama@http://10.0.0.5:11434")   # any reachable box
    backend = resolve("dgx_spark")          # the original node, over SSH
    backend = resolve("replay")             # recorded observations, offline

Every backend returns the same record shape and carries its own provenance, so
a report can always say where an observation came from and whether a GPU was
actually observed.
"""

from backends.base import (BackendIdentity, BackendUnavailable, VisionBackend,
                           annotate)
from backends.ollama import OllamaBackend
from backends.replay import ReplayBackend, ReplayMiss
from backends.ssh import SshOllamaBackend

__all__ = ["BackendIdentity", "BackendUnavailable", "OllamaBackend",
           "ReplayBackend", "ReplayMiss", "SshOllamaBackend", "VisionBackend",
           "annotate"]
