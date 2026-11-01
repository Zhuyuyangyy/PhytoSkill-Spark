# Security triage (Bandit + pip-audit, 2026-11-01)

## Bandit — CI runs it strictly, no `|| true`
The findings below were reviewed and marked `# nosec` at the exact line,
with the reason in code. A new Bandit finding anywhere else fails the
security job.

- `runtime/executor.py`, `runtime/worker.py` — B102 exec: executes only a
  Skill entrypoint whose sha256 was checked against the registry manifest
  (executor) or re-checked in the bounded child (worker). This is the
  product's core function, not execution of untrusted input.
- `backends/ollama.py` — B310 urlopen: the URL is the operator-configured
  ollama endpoint (environment variable or localhost default), never
  skill payload data. Calls go only to ollama `/api/generate` / `/api/ps`.
- `runtime/sandbox.py` — B404/B603 subprocess: `Popen` is used with a list
  argv and shell disabled (the default). The only production caller is
  `executor` with `[sys.executable, "-m", "runtime.worker"]`, under
  OS resource limits (POSIX setrlimit / Windows job object) and timeouts.

## pip-audit — what it audits, and what it does not
- Scope is `requirements-audit.txt` (project dependencies only).
  Auditing the whole CI environment would fail on runner tooling such as
  pip itself, which is not a project dependency.
- `python -m pip-audit` is the wrong module name; the correct invocation
  is `python -m pip_audit` (the binary is `pip-audit`). The old workflow
  used the wrong name and hid the error behind `|| true`, so pip-audit
  never actually ran.
- Dependency fixes in this change: cryptography raised from `>=42,<47`
  to `>=48,<51` and Pillow raised from `>=10,<12` to `>=13,<14`, because
  pip-audit reported known vulnerabilities in cryptography 46.x and
  Pillow 11.x with fixes in 50.x and 13.x respectively.

- Dependency fixes in this change (continued): paramiko raised from
  `>=3,<5` to `>=5,<6`, because pip-audit reported a SHA-1 RSA-key issue
  in paramiko 4.0.0 with no fixed release inside the old `<5` range;
  5.1.0 audits clean.
