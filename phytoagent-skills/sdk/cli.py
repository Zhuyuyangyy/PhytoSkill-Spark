"""Shared entrypoint for portable Skill scripts; requires the installed SDK."""

import argparse
import json
from pathlib import Path
import sys

from registry import SkillRegistry
from runtime.executor import SkillExecutor
from sdk.exceptions import SkillError
from sdk.schema import read_json


def run_package_cli(package: Path) -> int:
    parser = argparse.ArgumentParser(description="Execute one Phyto Skill with an explicit input and mode")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=["fixture", "replay", "live"], required=True)
    trust = parser.add_mutually_exclusive_group(required=True)
    trust.add_argument("--public-key", type=Path)
    trust.add_argument("--allow-unsigned", action="store_true")
    args = parser.parse_args()
    try:
        # Only register the requested package; siblings need not be installed.
        registry = SkillRegistry(package.parent, trusted_public_key=args.public_key,
                                 require_signature=not args.allow_unsigned)
        record = registry.register(package)
        result = SkillExecutor(registry).execute(record["manifest"]["name"], read_json(args.input), mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (SkillError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
