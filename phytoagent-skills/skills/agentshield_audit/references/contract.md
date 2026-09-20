# 审计契约说明

## 输入

- `audit_id`：本次审计标识。
- `stage`：`pre_registration`（注册前审包）或 `pre_output`（输出前审报告）。
- `subject.kind`：`skill_package` 或 `report`。

`skill_package` 需要 `subject.package_dir`（相对于本Skill包的路径）与可选的
`subject.declared_permissions`、`subject.trace`。

`report` 需要 `subject.report`（必须含 `claims` 数组）与可选的 `subject.trace`。

## 输出

- `verdict`：`PASS` / `QUARANTINE` / `INSUFFICIENT`。
- `checks`：每项含 `name`、`status`（passed / failed / not_run）与 `detail`。
- `trust_level`：`SUPPORTED` / `LIMITED` / `INSUFFICIENT`。
- `violations`：命中的具体违规描述，可据此修复。

## 明确不做

- 不生成 0–100 分数。
- 不调用语言模型判定语义。
- 不替代 Runtime Middleware 的实时权限拦截。
