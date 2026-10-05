"""Ollama backend: inference against any ollama endpoint, over plain HTTP.

This is the backend that makes the project hardware-independent. DGX Spark was
never special at the protocol level — it runs ollama like anything else, and the
old code reached it by uploading a script over SSH and running it there. Talk to
the HTTP API directly and the same code works against:

* a laptop CPU running ollama locally,
* a workstation with a CUDA GPU,
* a rented instance,
* the DGX node, if its ollama port is reachable.

No SSH, no paramiko, no shell. Configuration only.

``poster`` is injectable so tests can run offline; nothing in the default path
reaches the network unless a caller invokes it.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Callable

from backends.base import (BackendIdentity, BackendUnavailable, VisionBackend,
                           annotate)
from backends.parsing import MODE_HERB, NUM_PREDICT, build_prompt, parse_regions

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# Token budgets differ per mode because the subject does: a plate of root slices
# makes the model enumerate slice by slice, and the worst observed herb reply ran
# to 1500 tokens while still producing valid regions.
NUM_PREDICT_BY_MODE = {"live": 400, MODE_HERB: NUM_PREDICT}

Poster = Callable[[str, dict, float], dict]


def default_poster(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON to an ollama endpoint and return the decoded body."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:600]
        raise BackendUnavailable(
            f"ollama endpoint returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise BackendUnavailable(f"ollama endpoint unreachable: {error.reason}") from error


def default_getter(url: str, timeout: float):
    """GET JSON from an ollama endpoint. Returns None when unavailable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except Exception:  # noqa: BLE001 - a probe failure is not an inference failure
        return None


class OllamaBackend(VisionBackend):
    """Run vision inference through the ollama HTTP API."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, *, kind: str = "local_cpu",
                 name: str = "ollama", poster: Poster | None = None,
                 getter=None, probe_gpu: bool = True):
        if not endpoint:
            raise BackendUnavailable("ollama backend requires an endpoint")
        self._endpoint = endpoint.rstrip("/")
        self._kind = kind
        self._name = name
        self._poster = poster or default_poster
        self._getter = getter or default_getter
        self._gpu_observed = False
        self._gpu_detail: dict | None = None
        if probe_gpu and poster is None and getter is None:
            # Only a real deployment can be probed. An injected poster means
            # this is a test, and probing would be a network call the test did
            # not ask for.
            self._probe_gpu()

    def _probe_gpu(self) -> None:
        """Ask ollama what is resident and whether it is on a GPU.

        This is an observation, not an assumption: ``size_vram > 0`` means the
        model is actually resident in GPU memory right now. If the probe fails
        we record ``gpu_observed=False`` rather than guessing from the hostname.
        """
        try:
            state = self._getter(f"{self._endpoint}/api/ps", 30)
        except Exception:  # noqa: BLE001 - probe must never break construction
            state = None
        if not isinstance(state, dict):
            return
        models = state.get("models")
        if not isinstance(models, list):
            return
        for entry in models:
            if isinstance(entry, dict) and int(entry.get("size_vram") or 0) > 0:
                self._gpu_observed = True
                self._gpu_detail = {
                    "name": entry.get("name"),
                    "size_vram": entry.get("size_vram"),
                    "source": "ollama /api/ps",
                }
                return

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(name=self._name, kind=self._kind,
                               endpoint=self._endpoint,
                               gpu_observed=self._gpu_observed)

    @property
    def gpu_detail(self) -> dict | None:
        return self._gpu_detail

    def run_vision(self, *, image_bytes: bytes, species: str, model: str,
                   mode: str, timeout: int = 600, prompt: str | None = None) -> dict:
        if mode not in NUM_PREDICT_BY_MODE:
            raise BackendUnavailable(f"ollama backend does not support mode {mode!r}")
        text_prompt = prompt or build_prompt(species=species, mode=mode)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": model,
            "prompt": text_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0,
                        "num_predict": NUM_PREDICT_BY_MODE[mode]},
        }

        started = time.perf_counter()
        body = self._poster(f"{self._endpoint}/api/generate", payload, float(timeout))
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not isinstance(body, dict):
            raise BackendUnavailable("ollama returned a non-object response")
        text = body.get("response") or ""
        image_usable, regions = parse_regions(text, image_bytes=image_bytes, mode=mode)

        return annotate({
            "model": model,
            "image_usable": image_usable,
            "regions": regions,
            "latency_ms": round(elapsed_ms, 1),
            "eval_count": body.get("eval_count"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "load_duration_ns": body.get("load_duration"),
            "total_duration_ns": body.get("total_duration"),
            # Reported only when actually observed; never synthesised.
            "gpu": self._gpu_detail,
            "gpu_ps": None,
            "raw_response": text[:2000],
        }, self, cache="miss")
