# Changelog

## v0.3.0 — 2026-11-01

Composable Agent Skills for medicinal-plant assessment: SDK, Registry,
Compiler and Runtime with Ed25519 package signature verification.
Five skills: plant_vision, growth_risk, herbal_knowledge,
evidence_fusion, agentshield_audit. Offline demo and a real-model
harness (bring your own endpoint).

Local verification on 2026-11-01, fresh checkout, Python 3.12:
558 offline tests passed, 2 skipped (missing upstream TCM-Mind-RAG
data directory, documented in the tests).

Fix in this release: tests/test_data_independence.py imports Pillow,
which is now declared in the `[test]` extra in
phytoagent-skills/pyproject.toml.

Limits, unchanged: research / engineering prototype, NOT a validated
plant-disease diagnostic system. Single-annotator experiment recall
0.1864; no blind-test generalisation evidence. Root LICENSE is
Apache-2.0; some package-level skill cards still declare NOASSERTION
and must not be assumed independently relicensed.

Install:
```
cd phytoagent-skills
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m demo.phytoforge_demo
.venv/bin/python -m pytest -q
```
