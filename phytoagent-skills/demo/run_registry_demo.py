"""Run the signed SDK/Registry loop without model credentials or network access."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from sdk import BaseSkill
from sdk.schema import package_file


def run_demo() -> dict:
    example = Path(__file__).resolve().parents[1] / "examples" / "contract_probe"
    with TemporaryDirectory(prefix="phyto-sdk-") as temporary:
        root = Path(temporary)
        package = root / "skills" / "contract_probe"
        shutil.copytree(example, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        private, public = root / "trust" / "publisher.private.pem", root / "trust" / "publisher.public.pem"
        generate_keypair(private, public)
        sign_package(package, private)
        registry = SkillRegistry(package.parent, trusted_public_key=public)
        records = registry.discover()
        tools = registry.available_tools
        instructions = registry.instructions("contract_probe")
        record = registry.verify_entry("contract_probe")
        filename, class_name = record["manifest"]["entrypoint"].split(":")
        source = package_file(package, filename)
        # Execute only this shipped SDK example, after verification. No general
        # runtime or Python sandbox is implied; discovery itself never imports it.
        namespace = {"__name__": "phyto_sdk_demo", "__file__": str(source)}
        exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
        skill_type = namespace[class_name]
        if not isinstance(skill_type, type) or not issubclass(skill_type, BaseSkill):
            raise TypeError("The demo entrypoint must implement BaseSkill")
        output = skill_type(package).execute({"message": "SDK + Registry contract verified"})
        registry.validate_output("contract_probe", output)
        return {
            "scope": "sdk_registry_only", "mode": "fixture", "model_called": False,
            "trace": ["signature_created", "signature_verified", "catalog_discovered",
                      "tool_schema_exported", "instructions_loaded", "contract_validated_execution"],
            "verification": records[0]["verification"], "available_tools": tools,
            "instructions_loaded": bool(instructions), "result": output,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the JSON execution report")
    args = parser.parse_args()
    report = json.dumps(run_demo(), ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
