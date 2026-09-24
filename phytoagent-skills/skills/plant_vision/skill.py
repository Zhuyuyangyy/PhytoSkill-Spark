"""plant_vision adapter with three explicit modes, never silently substituted.

* ``fixture`` — the published synthetic case. Answers exactly one input and
  refuses everything else, which is what makes the contract testable offline.
* ``live`` — real inference on the DGX Spark node, observing **field/leaf**
  phenotypes (yellowing, spots, wilting). Requires a real image path.
* ``herb`` — real inference on the same node, observing **dried sliced herb
  material** (cut-surface texture, colour, mould, insect damage, slice shape).

``herb`` exists because the supplied image set is 15 photographs of sliced
Astragalus root, not growing plants. Asking a leaf-chlorosis prompt about a plate
of root slices returns either nothing or a region covering the whole plate; the
instrument has to match the subject.

Every real mode has **no fixture fallback**: a missing node, a missing credential,
a ``fixture://`` placeholder or an absent file are all hard errors. Answering a
real request with synthetic data would be a fabricated observation.
"""

from __future__ import annotations

from pathlib import Path

from dgx.cache import ObservationCache, cached_vision
from dgx.client import DgxClient, load_credentials
from dgx.herb_vision import build_prompt as build_herb_prompt
from dgx.herb_vision import run_vision as run_herb_vision
from dgx.vision import build_prompt as build_leaf_prompt
from dgx.vision import run_vision as run_leaf_vision
from sdk import BaseSkill, ContractError
from sdk.exceptions import UnsupportedModeError
from sdk.fixture_skill import FixtureSkill

DEFAULT_MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


def _default_env_file() -> Path:
    """Locate the credential file relative to the *source* checkout.

    ``__file__`` is wrong here: a signed copy of this package is frequently run
    from a temporary directory (the demo and the gate both do that), where no
    credential file exists. Walking up from the current working directory finds
    the checkout the run was started from, which is where the operator put it.
    """
    for candidate in (Path.cwd(), *Path.cwd().parents):
        path = candidate / ".dgx.env"
        if path.is_file():
            return path
    return Path(".dgx.env")


class PlantVisionSkill(BaseSkill):
    """One narrowly scoped, contract-validated capability."""

    def __init__(self, package_dir, *, model: str | None = None,
                 env_file: str | Path | None = None,
                 cache: ObservationCache | None = None):
        super().__init__(package_dir)
        self._model = model or DEFAULT_MODEL
        # None means "use the repository default", not "no credentials".
        self._env_file = Path(env_file) if env_file is not None else _default_env_file()
        # One inference costs 12-167 s and contends with other work on the node.
        # Both A/B arms and every rerun would otherwise pay it again.
        self._cache = cache if cache is not None else ObservationCache()

    @property
    def model(self) -> str:
        return self._model

    def run(self, payload: dict, *, mode: str) -> dict:
        if mode == "fixture":
            return self._run_fixture(payload)
        if mode == "live":
            return self._run_real(payload, mode=mode, runner=run_leaf_vision,
                                  data_origin="dgx_live_inference",
                                  build_prompt=build_leaf_prompt)
        if mode == "herb":
            return self._run_real(payload, mode=mode, runner=run_herb_vision,
                                  data_origin="dgx_herb_inference",
                                  build_prompt=build_herb_prompt)
        raise UnsupportedModeError(f"plant_vision does not support mode {mode!r}")

    # ── modes ─────────────────────────────────────────────────────────────

    def _run_fixture(self, payload: dict) -> dict:
        """Delegate to the sealed fixture; a mismatch is an explicit failure."""
        return FixtureSkill.run(self, payload, mode="fixture")

    def _run_real(self, payload: dict, *, mode: str, runner, data_origin: str,
                  build_prompt) -> dict:
        """Run real inference on the DGX Spark node. No fixture fallback."""
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
            credentials = load_credentials(self._env_file)
        except ValueError as exc:
            raise ContractError(f"DGX credentials unavailable: {exc}") from exc

        image_bytes = path.read_bytes()
        with DgxClient(credentials) as client:
            # A large photograph on a contended node takes minutes: the observed
            # range is ~12 s to ~167 s for the same code path. 600 s is not
            # generous, it is what the slowest real image needed.
            prompt = build_prompt(species=species)
            call = cached_vision(client, cache=self._cache, image_bytes=image_bytes,
                                 species=species, model=self._model, mode=mode,
                                 runner=runner, timeout=900, prompt=prompt)

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
        limitations = [
            "本次结果来自 DGX Spark 上的真实视觉模型推理，不是合成 fixture。",
            "模型只描述可见性状，不判定等级、真伪或病因；model_score 是模型对自己描述的信心，不是质量分。",
            "可见区域面积比例不是有效成分变化比例；性状观察不能替代药典检验与专业人员鉴定。",
        ]
        if not call["image_usable"]:
            limitations.append("模型判定图像不可用；未据此产生任何表型结论。")
        if not observations:
            limitations.append("模型未报告任何可用区域；此处不补造区域。")
        return {
            "status": "success",
            "case_id": payload.get("case_id", ""),
            "species": species,
            "provenance": {
                "data_origin": data_origin,
                "skill": "plant_vision",
                "version": self.version,
                "model_called": True,
                "model": call["model"],
                "dgx_hardware_used": True,
                "gpu": call["gpu"],
                "latency_ms": call["latency_ms"],
                "eval_count": call["eval_count"],
                "observation_cache": call.get("cache", "unknown"),
            },
            "image_usable": call["image_usable"],
            "observations": observations,
            "limitations": limitations,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _area(bbox: list[float]) -> float:
        """Fraction of the image the box covers. A description, not a leaf area."""
        left, top, right, bottom = bbox
        return round((right - left) * (bottom - top), 4)
