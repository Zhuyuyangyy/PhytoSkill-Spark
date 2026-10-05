"""Execution backends: where vision inference actually happens.

This layer exists because the project was born on one specific machine. DGX Spark
was the only place inference could run, so its SSH client ended up imported
directly by the Skill package. That made hardware an identity rather than a
configuration choice — the package could not run anywhere else, and a project
with no access to that node could not run its own tests.

A backend answers exactly one question: **given image bytes and a mode, return a
parsed observation record.** Everything above it — the prompt, the parser, the
contract, the Skill — is hardware-independent and must not know or care whether
the bytes were processed by a GPU in a remote rack or by a CPU on a laptop.

Rules every backend must obey:

* **Missing dependency is an error, not a fallback.** A backend that cannot run
  raises :class:`BackendUnavailable`. It never returns a plausible-looking
  observation in place of a real one.
* **GPU state is reported, never assumed.** A backend without a GPU returns
  ``gpu=None`` and ``gpu_observed=False``. Inventing ``NVIDIA GB10`` on a laptop
  would be a fabricated measurement.
* **Provenance travels with the result.** Every record carries a ``backend``
  identity block, so a report can always say where an observation came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

# Vocabulary for the ``kind`` field. ``kind`` describes the execution *class*,
# not the product name: a rented 4090 and a DGX Spark are both ``remote_gpu``
# as far as this code is concerned.
KIND_LOCAL_CPU = "local_cpu"
KIND_LOCAL_CUDA = "local_cuda"
KIND_REMOTE_GPU = "remote_gpu"
KIND_DGX_SPARK = "dgx_spark"
KIND_REPLAY = "replay"

KINDS = (KIND_LOCAL_CPU, KIND_LOCAL_CUDA, KIND_REMOTE_GPU, KIND_DGX_SPARK,
         KIND_REPLAY)

# Every observation record a backend returns contains at least these keys. The
# shape is unchanged from the DGX-era records, so existing artifacts stay valid.
RESULT_KEYS = ("model", "image_usable", "regions", "latency_ms", "backend", "cache")


class BackendUnavailable(RuntimeError):
    """The backend cannot run.

    Raised for a missing dependency, a missing credential or an unreachable
    endpoint. Deliberately distinct from a contract error: this is an
    environment problem, not a bad request, and the two must not be conflated
    in a report.
    """


@dataclass(frozen=True)
class BackendIdentity:
    """Where a result was produced, and whether a GPU was actually observed."""

    name: str
    kind: str
    endpoint: str
    gpu_observed: bool

    def to_dict(self) -> dict:
        return asdict(self)


class VisionBackend(ABC):
    """Run one vision inference for one mode and return a parsed record."""

    @property
    @abstractmethod
    def identity(self) -> BackendIdentity:
        """Where this backend executes."""

    @abstractmethod
    def run_vision(self, *, image_bytes: bytes, species: str, model: str,
                   mode: str, timeout: int = 600, prompt: str | None = None) -> dict:
        """Return one observation record.

        ``mode`` selects the observation vocabulary: ``live`` for leaf/field
        phenotypes, ``herb`` for dried sliced material. ``prompt`` may be
        supplied by the caller so an observation cache can key on the exact
        text; when omitted the backend renders the mode's own prompt.
        """


def annotate(record: dict, backend: VisionBackend, *, cache: str = "miss") -> dict:
    """Attach provenance to a result record.

    Kept as a function rather than inlined into each backend so the identity
    block cannot drift between implementations.
    """
    record = dict(record)
    record["backend"] = backend.identity.to_dict()
    record["cache"] = cache
    return record
