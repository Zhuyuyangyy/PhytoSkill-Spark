"""plant_vision adapter with three explicit modes, never silently substituted.

* ``fixture`` — the published synthetic case. Answers exactly one input and
  refuses everything else, which is what makes the contract testable offline.
* ``live`` — real inference on the configured backend, observing **field/leaf**
  phenotypes (yellowing, spots, wilting). Requires a real image path.
* ``herb`` — real inference on the same backend, observing **dried sliced herb
  material** (cut-surface texture, colour, mould, insect damage, slice shape).

``herb`` exists because the supplied image set is 15 photographs of sliced
Astragalus root, not growing plants. Asking a leaf-chlorosis prompt about a plate
of root slices returns either nothing or a region covering the whole plate; the
instrument has to match the subject.

**Which machine runs the inference is configuration, not code.** This Skill used
to import the DGX Spark SSH client directly, which made one specific piece of
hardware a precondition for running the package at all. It now takes a backend
from ``backends/`` — a local ollama server, a CUDA workstation, a rented GPU, a
remote node over SSH, or recorded observations replayed offline. See
``backends.registry.resolve``.

Every real mode has **no fixture fallback**: an unconfigured backend, a missing
credential, an unreachable endpoint or an absent file are all hard errors.
Answering a real request with synthetic data would be a fabricated observation.
"""

from __future__ import annotations

import os
from pathlib import Path

from sdk import BaseSkill, ContractError
from sdk.exceptions import UnsupportedModeError
from sdk.fixture_skill import FixtureSkill

DEFAULT_MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"

# The backend layer is imported lazily, inside the real modes. A signed package
# may be copied anywhere and executed on its own; importing a backend at module
# scope would make even the offline fixture mode fail on a machine that has no
# backend layer installed. Fixture mode must always work; inference mode is
# allowed to require one.

# ``data_origin`` values are unchanged from the DGX era. They name the
# observation vocabulary, not the hardware; renaming them is a separate change
# because the value is asserted by the evidence_fusion schema too.
DATA_ORIGIN_BY_MODE = {"live": "dgx_live_inference", "herb": "dgx_herb_inference"}


class PlantVisionSkill(BaseSkill):
    """One narrowly scoped, contract-validated capability."""

    def __init__(self, package_dir, *, model: str | None = None,
                 backend=None,
                 cache=None,
                 cache_dir: str | Path | None = None,
                 env_file: str | Path | None = None):
        """
        ``backend`` may be injected directly. When it is omitted, one is resolved
        from ``PHYTO_VISION_BACKEND`` at first use; an unset variable is a hard
        error rather than a guess, so two machines never silently disagree about
        where an observation came from.

        ``env_file`` is accepted for compatibility. It is only meaningful to an
        SSH backend, which reads node credentials from it.
        """
        super().__init__(package_dir)
        self._model = model or DEFAULT_MODEL
        self._backend = backend
        self._env_file = Path(env_file) if env_file is not None else None
        self._cache = cache
        self._cache_dir = cache_dir

    @property
    def model(self) -> str:
        return self._model

    def run(self, payload: dict, *, mode: str) -> dict:
        if mode == "fixture":
            return self._run_fixture(payload)
        if mode in DATA_ORIGIN_BY_MODE:
            return self._run_real(payload, mode=mode)
        raise UnsupportedModeError(f"plant_vision does not support mode {mode!r}")

    # ── modes ─────────────────────────────────────────────────────────────

    def _run_fixture(self, payload: dict) -> dict:
        """Delegate to the sealed fixture; a mismatch is an explicit failure."""
        return FixtureSkill.run(self, payload, mode="fixture")

    def _run_real(self, payload: dict, *, mode: str) -> dict:
        """Run real inference on the configured backend. No fixture fallback."""
        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ContractError("real mode requires a non-empty image_path")
        species = payload.get("species")
        if not isinstance(species, str) or not species.strip():
            raise ContractError("real mode requires a species")
        if image_path.startswith("fixture://"):
            raise ContractError(
                "real mode cannot read a fixture:// placeholder; supply a real image path")
        path = Path(image_path)
        if not path.is_file():
            raise ContractError(f"image not found: {image_path}")

        try:
            from backends.base import BackendUnavailable
            from backends.observation_cache import cached_observation
            from backends.parsing import build_prompt
        except ImportError as exc:  # pragma: no cover - depends on install layout
            raise ContractError(f"vision backend layer unavailable: {exc}") from exc

        backend = self._require_backend()
        image_bytes = path.read_bytes()
        # Part of the cache key: an edited prompt must not reuse old observations.
        prompt = build_prompt(species=species, mode=mode)

        try:
            # A large photograph on a contended machine takes minutes: the
            # observed range was ~12 s to ~167 s for the same code path. 900 s
            # is not generous, it is what the slowest real image needed.
            call = cached_observation(
                backend, cache=self._observation_cache(), image_bytes=image_bytes,
                species=species, model=self._model, mode=mode, timeout=900,
                prompt=prompt)
        except BackendUnavailable as exc:
            raise ContractError(f"vision backend unavailable: {exc}") from exc

        observations = []
        for index, region in enumerate(call["regions"], start=1):
            observations.append({
                "observation_id": f"vision-region-{index:02d}",
                "phenotype": region["phenotype"],
                "region": {"label": region["label"],
                           "bbox_normalized": region["bbox_normalized"]},
                "affected_area_fraction": self._area(region["bbox_normalized"]),
                "area_basis": "visible_subject_area",
                "model_score": region["model_score"],
            })

        identity = call.get("backend") or {}
        where = f"{identity.get('name', 'unknown')}（{identity.get('kind', 'unknown')}）"
        limitations = [
            f"本次结果来自 {where} 后端上的真实视觉模型推理，不是合成 fixture。",
            "模型只描述可见性状，不判定等级、真伪或病因；model_score 是模型对自己描述的信心，不是质量分。",
            "可见区域面积比例不是有效成分变化比例；性状观察不能替代药典检验与专业人员鉴定。",
        ]
        if call.get("cache") == "hit":
            limitations.append(
                "本条观察来自观察缓存：内容与首次推理一致，耗时不代表本次推理。")
        if call.get("cache") == "replay":
            limitations.append(
                "本条观察来自已记录产物回放，内容为当时真实推理所得；"
                "latency 属于原始运行，不代表本次性能。")
        if not call["image_usable"]:
            limitations.append("模型判定图像不可用；未据此产生任何表型结论。")
        if not observations:
            limitations.append("模型未报告任何可用区域；此处不补造区域。")

        return {
            "status": "success",
            "case_id": payload.get("case_id", ""),
            "species": species,
            "provenance": {
                "data_origin": DATA_ORIGIN_BY_MODE[mode],
                "skill": "plant_vision",
                "version": self.version,
                "model_called": True,
                "model": call["model"],
                # Historical field name. It records whether a GPU was observed
                # during this call, whatever the backend was; it is driven by
                # the backend's own report and never hard-coded.
                "dgx_hardware_used": bool(identity.get("gpu_observed")),
                "backend": identity,
                "gpu": call.get("gpu"),
                "latency_ms": call["latency_ms"],
                "eval_count": call["eval_count"],
                "observation_cache": call.get("cache", "unknown"),
            },
            "image_usable": call["image_usable"],
            "observations": observations,
            "limitations": limitations,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _require_backend(self):
        """Return the backend, resolving it from configuration on first use."""
        if self._backend is not None:
            return self._backend
        try:
            from backends.base import BackendUnavailable
            from backends.registry import resolve as resolve_backend
        except ImportError as exc:  # pragma: no cover - depends on install layout
            raise ContractError(f"vision backend layer unavailable: {exc}") from exc
        try:
            self._backend = resolve_backend(environ=os.environ, env_file=self._env_file)
        except BackendUnavailable as exc:
            raise ContractError(f"vision backend unavailable: {exc}") from exc
        return self._backend

    def _observation_cache(self):
        """One inference may cost minutes; both A/B arms and every rerun would
        otherwise pay for it again. Created lazily so fixture mode needs nothing."""
        if self._cache is None:
            from backends.observation_cache import DEFAULT_CACHE_DIR, ObservationCache

            self._cache = ObservationCache(
                self._cache_dir if self._cache_dir is not None else DEFAULT_CACHE_DIR)
        return self._cache

    @staticmethod
    def _area(bbox: list[float]) -> float:
        """Fraction of the image the box covers. A description, not a leaf area."""
        left, top, right, bottom = bbox
        return round((right - left) * (bottom - top), 4)
