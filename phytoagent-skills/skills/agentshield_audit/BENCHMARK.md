# agentshield-audit · Evaluation status

- Fixture contract checks: run `python -m evals.run_contracts` from the project checkout.
- Agent A/B with and without this Skill: **NOT RUN**.
- Real adversarial audit corpus and detection rates: **NOT MEASURED**.
- SkillSpector / SkillEvaluator / OMS verification: **NOT RUN**.

The package includes positive, missing-parameter and negative-trigger cases in
evals/evals.json. These are specifications, not passed Agent tests. Keep the same
model, endpoint, tools, dataset and harness for both A/B arms; vary only Skill
instructions. Do not use fixture-contract success as a Tier-3 verdict.
