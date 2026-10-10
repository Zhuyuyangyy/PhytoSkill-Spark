"""OS-level execution boundary for Skill calls.

What this module is: a **resource boundary**. A Skill running in ``live``,
``herb`` or ``corpus`` mode executes in a separate process whose memory, child
process count and wall-clock time are capped by the operating system, so a
runaway or hostile Skill exhausts its own process instead of the host, and a
hung adapter cannot hang the run.

What this module is NOT: a security sandbox, and it does not claim to be one.
On Windows the caps come from a Job Object; on POSIX from ``setrlimit``. They
bound memory, processes and time. They do **not** confine filesystem or
network access — that needs an AppContainer, a container, or a kernel driver,
none of which this project has. Filesystem and network governance remains the
declarative permission model (manifest ∩ host policy ∩ task grants, enforced
in :mod:`runtime.shield`) plus the compile-time gate. Anyone who needs
OS-level filesystem or network confinement must run the whole runtime inside a
container; this boundary raises the cost of the attacks it does cover without
pretending to cover the rest.

What the process cap means in practice: on Windows the Job Object's
``ActiveProcessLimit`` is a hard per-tree quota — the default of two allows
the interpreter's own startup (a venv redirector plus the real interpreter)
and at most one helper, and grandchildren are refused by the operating
system. On POSIX ``RLIMIT_NPROC`` counts processes per real user id, not per
task tree, and may not be enforced at all in privileged environments; the
wall-clock timeout still terminates the whole process group there, but a
strict per-task process quota on Linux needs cgroup v2 ``pids.max``, which
this project does not implement.

One window is documented rather than closed: on Windows the child is assigned
to the job immediately after spawn, so a few milliseconds of execution happen
before the caps apply. Closing it needs a suspended spawn plus manual thread
resume; the project's threat model (buggy or hostile Skills, not a local
attacker with microsecond timing) does not justify that complexity yet.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from sdk.exceptions import SkillError

# Defaults chosen for this project's Skills: the heaviest is a vision adapter
# holding image bytes and a model reply. Two gigabytes is generous for that
# and still far below "allocate until the host dies". Five minutes covers the
# slowest real inference observed (12-167 s per image) with room to spare.
#
# The process default is two, not one, for a reason worth knowing: on Windows
# a virtualenv's ``python.exe`` is a redirector that spawns the real
# interpreter, so the interpreter's own startup already consumes one process
# slot. With a cap of one the child could not start at all. Two still bounds
# the tree hard: a Skill may spawn at most one helper, and that helper may not
# spawn anything. To forbid children entirely, run the runtime with a
# non-venv interpreter and set the cap to one.
DEFAULT_MEMORY_BYTES = 2 * 1024 ** 3
DEFAULT_MAX_PROCESSES = 2
DEFAULT_TIMEOUT_SECONDS = 300.0


class ExecutionBoundaryError(SkillError):
    """The execution boundary stopped the call: timeout, spawn failure, or a
    child that died without producing a result."""


@dataclass(frozen=True)
class ResourceLimits:
    memory_bytes: int = DEFAULT_MEMORY_BYTES
    max_processes: int = DEFAULT_MAX_PROCESSES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class BoundedResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


# ── Windows: Job Objects ────────────────────────────────────────────────────

_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JobObjectExtendedLimitInformation = 9


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _create_limited_job(limits: ResourceLimits):
    """Create a Windows job object carrying the memory and process caps."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ExecutionBoundaryError(
            f"Cannot create the execution boundary (CreateJobObjectW: "
            f"{ctypes.get_last_error()})")
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
    info.ProcessMemoryLimit = limits.memory_bytes
    info.BasicLimitInformation.ActiveProcessLimit = max(1, limits.max_processes)
    if not kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
            ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ExecutionBoundaryError(
            f"Cannot arm the execution boundary (SetInformationJobObject: {error})")
    return kernel32, job


def _run_windows(argv: list[str], limits: ResourceLimits, cwd, env,
                 input_bytes: bytes | None) -> BoundedResult:
    kernel32, job = _create_limited_job(limits)
    process = None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=env,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Assign immediately: the caps apply from here on. The milliseconds
        # between spawn and assignment are documented in the module docstring.
        if not kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(process._handle)):
            raise ExecutionBoundaryError(
                f"Cannot join the child to the execution boundary "
                f"(AssignProcessToJobObject: {ctypes.get_last_error()})")
        try:
            out, err = process.communicate(input=input_bytes,
                                           timeout=limits.timeout_seconds)
            return BoundedResult(process.returncode, out.decode("utf-8", "replace"),
                                 err.decode("utf-8", "replace"), False)
        except subprocess.TimeoutExpired:
            # Terminate the job, not just the child: the whole tree goes, so a
            # worker that spawned helpers cannot outlive the timeout.
            kernel32.TerminateJobObject(job, 1)
            process.wait(timeout=30)
            return BoundedResult(None, "", "", True)
    finally:
        kernel32.CloseHandle(job)


# ── POSIX: setrlimit ────────────────────────────────────────────────────────

def _posix_preexec(limits: ResourceLimits):
    def apply():
        import resource
        memory = limits.memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes,) * 2)
        except (ValueError, OSError):
            # RLIMIT_NPROC is not enforced for root on every platform; the
            # memory and time caps still apply.
            pass
    return apply


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Kill the whole process group, not just the direct child.

    ``start_new_session=True`` put the child in its own process group, so the
    group id is the child's pid and everything it spawned shares that group.
    Killing only the child would leave helpers running past the timeout — a
    timed-out Skill's children would outlive the boundary.
    """
    import signal
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # The group is already gone, or this platform refuses the call; the
        # direct child at least must not survive.
        process.kill()


def _run_posix(argv: list[str], limits: ResourceLimits, cwd, env,
               input_bytes: bytes | None) -> BoundedResult:
    process = subprocess.Popen(argv, cwd=cwd, env=env,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               preexec_fn=_posix_preexec(limits),
                               start_new_session=True)
    try:
        out, err = process.communicate(input=input_bytes,
                                       timeout=limits.timeout_seconds)
        return BoundedResult(process.returncode, out.decode("utf-8", "replace"),
                             err.decode("utf-8", "replace"), False)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        process.wait(timeout=30)
        return BoundedResult(None, "", "", True)


# Environment variables that belong to the *caller's* tooling, not to the
# bounded child. The boundary is not the agent: another process's sandbox
# instrumentation must not travel into it, both because the child should not
# depend on it and because a helper that fails under load must not be able to
# kill a legitimate Skill call. Anything a Skill legitimately needs — API
# endpoints, model names, PATH, TEMP — is inherited untouched.
_INSTRUMENTATION_PREFIXES = ("CODEBUDDY_", "WORKBUDDY_")


def child_environment(repo_root: str | Path, base: dict | None = None) -> dict:
    """The environment a bounded child runs in.

    Functional variables (API endpoints, model names, PATH, TEMP) are
    inherited; instrumentation variables are dropped. ``PYTHONPATH`` is set to
    the repository root and nothing else: the child's import surface is this
    project, not whatever the caller happened to have on its own path — a
    boundary that inherits the caller's module path is not a boundary. The
    interpreter's own site-packages still load, because the child *is* the
    same interpreter.
    """
    env = {key: value for key, value in (base if base is not None else os.environ).items()
           if not key.upper().startswith(_INSTRUMENTATION_PREFIXES)}
    env["PYTHONPATH"] = str(repo_root)
    return env


def run_bounded(argv: list[str], *, limits: ResourceLimits | None = None,
                cwd: str | Path | None = None, env: dict | None = None,
                input_bytes: bytes | None = None) -> BoundedResult:
    """Run ``argv`` under OS-level resource limits and return what it produced.

    ``input_bytes`` is written to the child's stdin — the request travels this
    way so no temporary file is created and none has to be deleted. The
    wall-clock timeout kills the whole process tree, not just the direct
    child. Raises :class:`ExecutionBoundaryError` when the boundary itself
    cannot be established — a call that cannot be bounded is not run.
    """
    limits = limits or ResourceLimits()
    argv = [str(part) for part in argv]
    if sys.platform == "win32":
        return _run_windows(argv, limits, str(cwd) if cwd else None, env, input_bytes)
    return _run_posix(argv, limits, str(cwd) if cwd else None, env, input_bytes)
