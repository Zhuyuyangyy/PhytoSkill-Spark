"""Debug one image through the Skill directly, surfacing the real exception.

The executor's generic catch hides the traceback behind SkillExecutionError, so a
failing image has to be run through the Skill itself to see what actually broke.
"""

import sys
import traceback
from pathlib import Path

from demo.fixture_workspace import fixture_registry
from runtime.executor import SkillExecutor
from sdk.exceptions import SkillError


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    image = Path(sys.argv[1]).resolve()
    mode = sys.argv[2] if len(sys.argv) > 2 else "herb"

    with fixture_registry() as registry:
        executor = SkillExecutor(registry)
        record = registry.verify_entry("plant_vision")
        package = Path(record["package_dir"])
        filename, class_name = record["manifest"]["entrypoint"].split(":")
        source = (package / filename).read_text(encoding="utf-8")
        namespace = {"__name__": "debug", "__file__": str(package / filename)}
        exec(compile(source, str(package / filename), "exec"), namespace)
        skill = namespace[class_name](package)
        try:
            result = skill.run({"case_id": "debug", "species": "黄芪",
                                "image_path": str(image)}, mode=mode)
        except SkillError as exc:
            print(f"SkillError: {type(exc).__name__}: {exc}")
            return 1
        except Exception:  # noqa: BLE001 - the point is to see the real traceback
            traceback.print_exc()
            return 1
        print("status:", result["status"])
        print("image_usable:", result["image_usable"],
              "observations:", len(result["observations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
