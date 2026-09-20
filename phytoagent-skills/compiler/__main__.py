"""Phyto Skill Compiler command line.

    python -m compiler compile --request "..." --output <dir>
    python -m compiler compile --spec skill_spec.json --output <dir>
    python -m compiler verify --spec skill_spec.json
    python -m compiler list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compiler.catalog import CAPABILITIES
from compiler.compiler import COMPILER_ID, COMPILER_VERSION, compile_request, compile_spec, validate_spec
from sdk.exceptions import SkillError
from sdk.schema import read_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compiler", description=__doc__)
    parser.add_argument("--version", action="version", version=f"{COMPILER_ID} {COMPILER_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile into a Skill package directory")
    compile_parser.add_argument("--request", help="Natural-language domain request")
    compile_parser.add_argument("--spec", type=Path, help="Existing SkillSpec to compile instead")
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--seal", action="store_true",
                                help="Write manifest.json (signing is a separate publisher step)")

    verify_parser = subparsers.add_parser("verify", help="Validate a SkillSpec; writes nothing")
    verify_parser.add_argument("--spec", type=Path, required=True)

    subparsers.add_parser("list", help="List the audited capability catalogue")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            print(json.dumps({"capabilities": sorted(CAPABILITIES)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "verify":
            spec = validate_spec(read_json(args.spec))
            print(json.dumps({"spec": spec["name"], "valid": True}, ensure_ascii=False, indent=2))
            return 0
        if args.spec is not None:
            if args.request:
                parser.error("use either --request or --spec, not both")
            result = compile_spec(read_json(args.spec), args.output, seal=args.seal)
        elif args.request:
            result = compile_request(args.request, args.output, seal=args.seal)
        else:
            parser.error("compile requires --request or --spec")
    except (SkillError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2
    print(json.dumps({"skill": result.spec["name"], "package": str(result.package_dir),
                      "files": result.files, "manifest_sha256": result.skill_manifest_sha256},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
