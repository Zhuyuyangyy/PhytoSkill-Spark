"""SSH client for the DGX Spark node.

The node is a remote aarch64 machine, not this workstation. Everything that needs
the GPU — vision inference, embeddings, vector search — runs over this connection.

Design rules:

* **Credentials never live in the repository.** They come from the environment
  (``PHYTO_DGX_*``) or a git-ignored file. This module reads them and nothing
  else, and never logs them.
* **Nothing is installed or modified on the node** by this module. It runs
  read-only probes and uploads/executes small scripts under ``/tmp``.
* **Every call records what it cost.** Latency and GPU state are captured per
  call, because "it works" is not the same claim as "it worked on the GPU".
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised only on a machine with paramiko
    import paramiko
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "paramiko is required for DGX Spark access: pip install paramiko") from exc

ENV_HOST = "PHYTO_DGX_HOST"
ENV_PORT = "PHYTO_DGX_PORT"
ENV_USER = "PHYTO_DGX_USER"
ENV_PASSWORD = "PHYTO_DGX_PASSWORD"
ENV_KEY = "PHYTO_DGX_KEY"

# Read-only queries. No installation, no container creation, no file writes
# outside /tmp.
GPU_QUERY = ("nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used "
             "--format=csv,noheader")
GPU_PROCESS_QUERY = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"


def _clean(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def load_credentials(env_file: str | Path | None = None) -> dict:
    """Resolve DGX credentials from the environment, then an optional env file.

    Real environment variables always win, so a shared host can override without
    editing a file. A missing password is a hard error, never a silent fallback.
    """
    values = {key: _clean(os.environ.get(key)) for key in
              (ENV_HOST, ENV_PORT, ENV_USER, ENV_PASSWORD, ENV_KEY)}
    if env_file is not None:
        path = Path(env_file)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key in values and not values[key]:
                    values[key] = value
    missing = [key for key in (ENV_HOST, ENV_USER, ENV_PASSWORD) if not values.get(key)]
    if missing:
        raise ValueError(f"Missing DGX credentials: {', '.join(missing)}")
    return {
        "host": values[ENV_HOST],
        "port": int(values[ENV_PORT] or 22),
        "user": values[ENV_USER],
        "password": values[ENV_PASSWORD],
        "key": values[ENV_KEY],
    }


@dataclass
class GpuState:
    """What the GPU was doing at one moment."""

    index: str
    name: str
    driver: str
    memory_total: str
    memory_used: str
    processes: list[dict]

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "driver": self.driver,
                "memory_total": self.memory_total, "memory_used": self.memory_used,
                "processes": self.processes}


class DgxClient:
    """A thin, explicit SSH session to the DGX Spark node."""

    def __init__(self, credentials: dict):
        self.credentials = credentials
        self._client: paramiko.SSHClient | None = None

    # ── connection ────────────────────────────────────────────────────────

    def connect(self) -> "DgxClient":
        if self._client is not None:
            return self
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(hostname=self.credentials["host"], port=self.credentials["port"],
                      username=self.credentials["user"], timeout=30, banner_timeout=30,
                      auth_timeout=30, look_for_keys=False, allow_agent=False)
        if self.credentials.get("password"):
            kwargs["password"] = self.credentials["password"]
        if self.credentials.get("key"):
            kwargs["key_filename"] = self.credentials["key"]
        client.connect(**kwargs)
        self._client = client
        return self

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "DgxClient":
        return self.connect()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── execution ─────────────────────────────────────────────────────────

    def run(self, command: str, *, timeout: int = 300) -> dict:
        """Run one command and return stdout, stderr and the exit status."""
        if self._client is None:
            raise RuntimeError("connect() before run()")
        _stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        return {
            "command": command,
            "stdout": stdout.read().decode("utf-8", "replace"),
            "stderr": stderr.read().decode("utf-8", "replace"),
            "status": stdout.channel.recv_exit_status(),
        }

    def run_script(self, source: str, *, timeout: int = 600) -> dict:
        """Upload a script to /tmp and run it. The node is never modified."""
        if self._client is None:
            raise RuntimeError("connect() before run_script()")
        remote = f"/tmp/_phyto_{abs(hash(source)) % 10**10}.py"
        sftp = self._client.open_sftp()
        try:
            with sftp.open(remote, "w") as handle:
                handle.write(source)
        finally:
            sftp.close()
        return self.run(f"python3 {remote}", timeout=timeout)

    # ── evidence ──────────────────────────────────────────────────────────

    def gpu_state(self) -> GpuState:
        """Read the GPU inventory and what is currently resident on it."""
        summary = self.run(GPU_QUERY)
        if summary["status"] != 0:
            raise RuntimeError(f"nvidia-smi failed: {summary['stderr'].strip()}")
        line = summary["stdout"].strip().splitlines()[0]
        parts = [item.strip() for item in line.split(",")]
        while len(parts) < 5:
            parts.append("N/A")
        processes = self._processes()
        return GpuState(index=parts[0], name=parts[1], driver=parts[2],
                        memory_total=parts[3], memory_used=parts[4],
                        processes=processes)

    def _processes(self) -> list[dict]:
        result = self.run(GPU_PROCESS_QUERY)
        if result["status"] != 0:
            return []
        rows = []
        for line in result["stdout"].strip().splitlines()[1:]:
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 2:
                rows.append({"pid": parts[0], "used_memory": parts[1]})
        return rows

    def environment(self) -> dict:
        """A factual snapshot of the node, for the report's platform section."""
        uname = self.run("uname -m; uname -r").get("stdout", "").split()
        hostname = self.run("hostname").get("stdout", "").strip()
        python_version = self.run("python3 --version").get("stdout", "").strip()
        cuda = self.run("/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1").get("stdout", "").strip()
        docker = self.run("docker --version").get("stdout", "").strip()
        state = self.gpu_state()
        return {
            "hostname": hostname,
            "machine": uname[0] if uname else "unknown",
            "kernel": uname[1] if len(uname) > 1 else "unknown",
            "python_version": python_version,
            "cuda": cuda,
            "docker": docker,
            "gpu": state.to_dict(),
            "credential_source": "environment",
        }


def redact(text: str, credentials: dict) -> str:
    """Keep a credential out of any log or report."""
    redacted = text
    for key in ("password", "key"):
        value = credentials.get(key)
        if value and value in redacted:
            redacted = redacted.replace(value, "***REDACTED***")
    return redacted


def _selftest() -> int:  # pragma: no cover - manual entry point
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    credentials = load_credentials()
    with DgxClient(credentials) as client:
        snapshot = client.environment()
        print(json_dumps(snapshot))
    return 0


def json_dumps(payload) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
