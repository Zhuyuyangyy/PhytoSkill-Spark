"""Probe the DGX Spark node and print a factual snapshot.

    python -m dgx probe
    python -m dgx vision --image path/to/leaf.jpg --species 黄芪
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dgx.client import DgxClient, load_credentials
from dgx.vision import run_vision

DEFAULT_MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(prog="dgx", description=__doc__)
    parser.add_argument("--env-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".dgx.env",
                        help="KEY=value file with PHYTO_DGX_* credentials (git-ignored)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="Print host, GPU and toolchain snapshot")
    vision = subparsers.add_parser("vision", help="Run one real vision inference")
    vision.add_argument("--image", type=Path, required=True)
    vision.add_argument("--species", default="黄芪")
    vision.add_argument("--model", default=DEFAULT_MODEL)
    vision.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        credentials = load_credentials(args.env_file)
    except ValueError as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2

    with DgxClient(credentials) as client:
        if args.command == "probe":
            print(json.dumps(client.environment(), ensure_ascii=False, indent=2))
            return 0
        image_bytes = args.image.read_bytes()
        record = run_vision(client, image_bytes=image_bytes,
                            species=args.species, model=args.model)
        record.pop("raw_response", None)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
