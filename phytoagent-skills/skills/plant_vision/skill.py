"""plant_vision adapter: fixture mode by default, live mode on the DGX Spark node.

The two modes are deliberately separate and never silently substituted:

* ``fixture`` — the published synthetic case. It answers exactly one input and
  refuses everything else, which is what makes the package's contract testable
  offline.
* ``live`` — real inference on the DGX Spark node. Selected explicitly by the
  caller. If the node is unreachable or the credential is missing, the call fails;
  it does not fall back to the fixture, because a fixture answer to a real
  photograph would be a fabricated observation.

What ``live`` never does: diagnose. The prompt asks for description only, and the
parser drops any region whose text names a pathogen. ``model_score`` is the
model's stated confidence in its own description, not a probability of disease.
"""

from __future__ import annotations

from pathlib import Path

from dgx.client import DgxClient, load_credentials
from dgx.vision import run_vision
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
                 env_file: str | Path | None = None):
        super().__init__(package_dir)
        self._model = model or DEFAULT_MODEL
        # None means "use the repository default", not "no credentials".
        self._env_file = Path(env_file) if env_file is not None else _default_env_file()

    @property
    def model(self) -> str:
        return self._model

    def run(self, payload: dict, *, mode: str) -> dict:
        if mode == "fixture":
            return self._run_fixture(payload)
        if mode == "live":
            return self._run_live(payload)
        raise UnsupportedModeError(f"plant_vision does not support mode {mode!r}")

    # ── modes ─────────────────────────────────────────────────────────────

    def _run_fixture(self, payload: dict) -> dict:
        """Delegate to the sealed fixture; a mismatch is an explicit failure."""
        return FixtureSkill.run(self, payload, mode="fixture")

    def _run_live(self, payload: dict) -> dict:
        """Run real inference on the DGX Spark node. No fixture fallback."""
        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ContractError("live mode requires a non-empty image_path")
        species = payload.get("species")
        if not isinstance(species, str) or not species.strip():
            raise ContractError("live mode requires a species")
        if image_path.startswith("fixture://"):
            raise ContractError(
                "live mode cannot read a fixture:// placeholder; supply a real image path")
        path = Path(image_path)
        if not path.is_file():
            raise ContractError(f"image not found: {image_path}")

        try:
            credentials = load_credentials(self._env_file)
        except ValueError as exc:
            raise ContractError(f"DGX credentials unavailable: {exc}") from exc

        with DgxClient(credentials) as client:
            call = run_vision(client, image_bytes=path.read_bytes(),
                              species=species, model=self._model)

        observations = []
        for index, region in enumerate(call["regions"], start=1):
            observations.append({
                "observation_id": f"vision-region-{index:02d}",
                "phenotype": region["phenotype"],
                "region": {"label": region["label"],
                           "bbox_normalized": region["bbox_normalized"]},
                "affected_area_fraction": self._area(region["bbox_normalized"]),
                "area_basis": "visible_leaf_area",
                "model_score": region["model_score"],
            })
        limitations = [
            "本次结果来自 DGX Spark 上的真实视觉模型推理，不是合成 fixture。",
            "模型只描述可见区域，不诊断病原；model_score 是模型对自己描述的信心，不是病害概率。",
            "可见叶片面积比例不是药效下降比例；表型观察不能替代植保人员鉴定。",
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
                "data_origin": "dgx_live_inference",
                "skill": "plant_vision",
                "version": self.version,
                "model_called": True,
                "model": call["model"],
                "dgx_hardware_used": True,
                "gpu": call["gpu"],
                "latency_ms": call["latency_ms"],
                "eval_count": call["eval_count"],
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
