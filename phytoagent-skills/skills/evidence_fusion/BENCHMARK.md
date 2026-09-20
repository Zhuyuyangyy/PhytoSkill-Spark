# evidence-fusion · Evaluation status

- Fixture contract checks: run with `python -m evals.run_contracts` from the project checkout; the external report records case verdicts and manifest hashes.
- Real StepFun baseline versus with-skill comparison: **NOT RUN**.
- Plant diagnosis accuracy, GPU latency, and DGX Spark throughput: **NOT MEASURED**.
- SkillSpector / SkillEvaluator / OMS verification: **NOT RUN**.

The package includes positive, missing-parameter and negative-trigger Agent cases in evals/evals.json. These are specifications, not passed Agent tests. Keep the same model, endpoint, tools, dataset and harness for both A/B arms; vary only Skill instructions. Record every prompt, tool call, final result, latency, token usage and judge rubric. Do not use fixture-contract success as a Tier-3 verdict.
