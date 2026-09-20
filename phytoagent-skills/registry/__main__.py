"""Small publisher and consumer CLI: python -m registry --help."""

import argparse
import json
from pathlib import Path
import sys

from registry.loader import SkillRegistry
from registry.signer import generate_keypair, sign_package
from registry.validator import validate_package
from sdk.exceptions import SkillError
from sdk.manifest import seal_manifest
from sdk.schema import read_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phyto Skill SDK / Registry tools")
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="Create a development Ed25519 keypair")
    keygen.add_argument("--private-key", type=Path, required=True)
    keygen.add_argument("--public-key", type=Path, required=True)
    seal = commands.add_parser("seal", help="Explicitly generate a manifest and file hashes")
    seal.add_argument("package", type=Path)
    seal.add_argument("--metadata", type=Path, required=True)
    sign = commands.add_parser("sign", help="Sign an unchanged, sealed package")
    sign.add_argument("package", type=Path)
    sign.add_argument("--private-key", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify package contracts, hashes and signature")
    verify.add_argument("package", type=Path)
    verify_trust = verify.add_mutually_exclusive_group(required=True)
    verify_trust.add_argument("--public-key", type=Path)
    verify_trust.add_argument("--allow-unsigned", action="store_true")
    export = commands.add_parser("tools", help="Discover Skills and print StepFun function tools")
    source = export.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--skills-dir", type=Path)
    trust = export.add_mutually_exclusive_group()
    trust.add_argument("--public-key", type=Path)
    trust.add_argument("--allow-unsigned", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "keygen":
            result = {"key_id": generate_keypair(args.private_key, args.public_key)}
        elif args.command == "seal":
            result = seal_manifest(args.package, read_json(args.metadata))
        elif args.command == "sign":
            result = sign_package(args.package, args.private_key)
        elif args.command == "verify":
            record = validate_package(args.package, trusted_public_key=args.public_key,
                                      require_signature=not args.allow_unsigned)
            result = {"name": record["manifest"]["name"], "verification": record["verification"]}
        else:
            if args.skills_dir is not None:
                registry = SkillRegistry(args.skills_dir, trusted_public_key=args.public_key,
                                         require_signature=not args.allow_unsigned)
            else:
                if args.public_key is not None or args.allow_unsigned:
                    parser.error("Use --skills-dir with trust flags; config files contain their own policy")
                registry = SkillRegistry.from_config(args.config or Path(__file__).with_name("registry.json"))
            registry.discover()
            result = registry.available_tools
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (SkillError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
