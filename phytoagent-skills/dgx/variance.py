"""Repeat live vision inference on the node and record the variance.

Model output is not deterministic, so a single run proves nothing about
stability. This runs the same image several times and records latency, token
count, GPU state and how many regions each run produced.
"""

import json
import sys
from pathlib import Path

from demo.fixture_workspace import fixture_registry
from runtime.executor import SkillExecutor

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    image = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        Path("artifacts/dgx/synthetic-leaf-03.png").resolve()
    species = sys.argv[2] if len(sys.argv) > 2 else "黄芪"
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    runs = []
    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        for index in range(repeats):
            response = executor.call("plant_vision", {
                "case_id": f"dgx-live-{index:03d}", "species": species,
                "image_path": str(image),
            }, mode="live", tool_call_id=f"dgx-run-{index}")
            if response["status"] != "success":
                print(f"run {index}: FAILED {response['error']['message']}")
                runs.append({"run": index, "status": "failed", "error": response["error"]})
                continue
            data = response["data"]
            provenance = data["provenance"]
            print(f"run {index}: usable={str(data['image_usable']):5} "
                  f"obs={len(data['observations'])} "
                  f"latency={provenance['latency_ms']:>8.1f}ms "
                  f"eval={provenance['eval_count']} "
                  f"gpu_procs={len(provenance['gpu']['processes'])}")
            runs.append({"run": index, "status": "success",
                         "image_usable": data["image_usable"],
                         "observations": data["observations"],
                         "latency_ms": provenance["latency_ms"],
                         "eval_count": provenance["eval_count"],
                         "gpu": provenance["gpu"]})

    record = {"model": MODEL, "image": str(image), "species": species,
              "repeats": repeats, "runs": runs}
    out = Path("artifacts/dgx/vision-variance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
