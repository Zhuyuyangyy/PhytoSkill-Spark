"""Run herb mode over the real image set and record every result.

Each image is one real inference on the DGX Spark node. The record keeps the
structured observations, the model's raw reply, latency and GPU state, so a claim
about the dataset can be traced back to the call that produced it.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from dgx.cache import ObservationCache
from runtime.executor import SkillExecutor
from runtime.shield import ShieldRuntime
from sdk.manifest import seal_manifest
from sdk.schema import read_json


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-").lower() or "batch"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    directory = Path(sys.argv[1])
    species = sys.argv[2] if len(sys.argv) > 2 else "黄芪"
    mode = sys.argv[3] if len(sys.argv) > 3 else "herb"
    exclude_prefix = sys.argv[4] if len(sys.argv) > 4 else ""

    root = Path(tempfile.mkdtemp(prefix="phyto-herb-batch-"))
    private, public = root / "p.pem", root / "pub.pem"
    generate_keypair(private, public)
    skills = root / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)

    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    runtime = ShieldRuntime(registry, trace_id="trace-herb-batch-001")
    executor = SkillExecutor(registry, cache=ObservationCache())

    records = []
    images = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if exclude_prefix:
        # The annotate folder holds both subjects, and the leaf files share the
        # "huangqi_" stem — so filtering by include-prefix does not separate them.
        # A herb calibration must not spend node time re-reading leaf photographs
        # with the herb prompt.
        images = [path for path in images if not path.name.startswith(exclude_prefix)]
    for index, image in enumerate(images, start=1):
        case_id = f"herb-{index:03d}"
        response = executor.call("plant_vision", {
            "case_id": case_id, "species": species, "image_path": str(image.resolve()),
        }, mode=mode, tool_call_id=f"{mode}-call-{index}")
        if response["status"] != "success":
            print(f"[{index:2}] {image.name:18} FAILED {response['error']['message'][:90]}")
            records.append({"image": image.name, "case_id": case_id,
                            "status": "failed", "error": response["error"]})
            continue
        data = response["data"]
        provenance = data["provenance"]
        print(f"[{index:2}] {image.name:18} usable={str(data['image_usable']):5} "
              f"obs={len(data['observations'])} "
              f"{provenance['latency_ms']:>8.1f}ms eval={provenance['eval_count']}")
        for obs in data["observations"]:
            print(f"       {obs['observation_id']} {obs['phenotype']:22} "
                  f"{obs['region']['label'][:26]:26} "
                  f"bbox={[round(v, 3) for v in obs['region']['bbox_normalized']]} "
                  f"score={obs['model_score']}")
        records.append({"image": image.name, "case_id": case_id, "status": "success",
                        "image_usable": data["image_usable"],
                        "observations": data["observations"],
                        "latency_ms": provenance["latency_ms"],
                        "eval_count": provenance["eval_count"],
                        "gpu": provenance["gpu"]})

    out = Path(f"artifacts/dgx/{mode}-batch-{_slug(directory.name)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scope": f"dgx_{mode}_batch", "species": species, "mode": mode,
                               "excluded_prefix": exclude_prefix or None,
                               "images": len(images), "records": records,
                               "trace": runtime.report()},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = sum(1 for r in records if r["status"] == "success")
    with_obs = sum(1 for r in records if r.get("observations"))
    print(f"\n{ok}/{len(records)} succeeded; {with_obs} produced observations")
    print(f"wrote {out}")
    shutil.rmtree(root, ignore_errors=True)
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
