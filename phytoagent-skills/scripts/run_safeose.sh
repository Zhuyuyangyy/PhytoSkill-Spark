#!/usr/bin/env bash
# One command to reproduce the SafeOSE '27 hackathon results (AgentShield NLIP SE).
#   bash scripts/run_safeose.sh            # venv + install + tests + evaluation + demo
# Needs Python >= 3.11 and network access only for `pip install`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ ! -d .venv-safeose ]; then
  "$PY" -m venv .venv-safeose
fi
# shellcheck disable=SC1091
source .venv-safeose/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[test,nlip]"
echo "== 1/3 NLIP SE test suite (unit + HTTP end-to-end + attack suite)"
python -m pytest -q -p no:cacheprovider tests/test_nlip_se.py
echo "== 2/3 Evaluation (writes artifacts/safeose/nlip-eval.json and .md)"
python -m scripts.nlip_eval > /dev/null
sed -n '/## Attack success by configuration/,/## Scenario benign/p' artifacts/safeose/nlip-eval.md
echo "== 3/3 Demo walkthrough"
python -m demo.nlip_demo
