"""Portable Skill entrypoint; install phytoagent-skills 0.2.x first."""

from pathlib import Path

from sdk.cli import run_package_cli


if __name__ == "__main__":
    raise SystemExit(run_package_cli(Path(__file__).resolve().parents[1]))
