# Contributing to PhytoSkill-Spark

Thank you for taking a look. This repository combines a research prototype with reusable skill and runtime components. **Small, well-scoped improvements with verifiable evidence are preferred** over large feature claims.

## Where to start

- **Documentation and examples:** clarify setup steps, error messages, or differences between fixture replay and real inference.
- **Tests and reliability:** add minimal reproductions for input validation, permissions, signatures, or evaluator behavior.
- **Skills and runtime:** open an issue describing the contract or boundary that needs to change before proposing a large refactor.

Start with [the project overview](README.md), [the detailed package guide](phytoagent-skills/README.md), and [known limitations](phytoagent-skills/docs/evaluation.md).

## Local development (offline)

Requires Python 3.11+.

~~~bash
cd phytoagent-skills
python -m venv .venv
python -m pip install -e ".[test]"
python -m demo.phytoforge_demo
python -m evals.run_contracts
python -m pytest -q
~~~

If your shell does not use the venv automatically, activate it before the installation command or invoke its Python directly. **These commands are project-provided validation paths; contributors should report actual results**, including the environment and failing tests.

## Before opening a pull request

1. Describe the problem and link an issue when relevant.
2. Keep the change focused, with a reproducible before/after example.
3. Add or update tests when behavior, permissions, or contracts change.
4. Explain whether the result uses **fixtures, replay, a real model call, or human labels**.
5. Include the commands you ran and their output summary. Do not claim a pass for checks that you could not execute.

## Boundaries

- Never commit API keys, credentials, private datasets, patient data, remote-machine host details, or unlicensed third-party assets.
- Do not describe the project as NVIDIA-verified, an isolated sandbox, or a validated plant-disease diagnostic system.
- Preserve the distinctions between evidence *consistency* and external ground-truth *correctness*.
- When changing manifests, signing, or schema contracts, update dependent tests and document migration/compatibility impacts.

## Licensing

This repository's root [LICENSE](LICENSE) is Apache-2.0. Some package-level skill cards still declare `NOASSERTION` and require separate license reconciliation; please do not assume those cards are independently relicensed.

For security-sensitive findings, avoid posting exploit secrets, credentials, or private artifacts in public Issues. Contact the maintainer through GitHub first to coordinate an appropriate disclosure route.
