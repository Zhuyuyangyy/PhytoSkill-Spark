---
name: agentshield-audit
description: "注册前与输出前的可信审计：校验Skill包完整性、权限最小化、Claim-Evidence关联与越权调用。本Skill只是可调用的审计入口，真正的权限拦截位于Runtime Middleware，Agent无法通过不调用本Skill绕过治理。"
license: NOASSERTION
---

# agentshield-audit

对Skill包或Agent报告做确定性、离线的可信审计。

1. 确认审计阶段：`pre_registration` 审包，`pre_output` 审报告。
2. 需要字段细节时读取 [契约说明](references/contract.md)。
3. 执行审计并保留每条检查的 name / status / detail。
4. 把审计结论与 trace_id 一并返回，不在此Skill内修改被审计对象。

## 执行

依赖已安装的 phytoagent-skills SDK 0.2.x。从本Skill目录执行：

```bash
python scripts/run.py --input examples/request.json --mode fixture --public-key /trusted/publisher.public.pem
```

开发调试可显式改用 `--allow-unsigned`；不能把该状态称为签名验证通过。

## 边界

- 本Skill**不是**权限执行点。Runtime Middleware 在每次工具调用前后拦截，
  Agent 选择不调用本Skill不能绕过治理。
- 审计结论来自本地确定性规则，不调用语言模型。
- 本Skill不签发、不修改、不重签任何 manifest；签名由发布者用包外私钥完成。
- 不使用 0–100 分；可信等级只有 SUPPORTED / LIMITED / INSUFFICIENT。
- 不声称 NVIDIA 官方 Verified，也不对接 skill.oms.sig。

权限、依赖与许可见 [Skill Card](skill-card.md)。行为用例见 [evals](evals/evals.json)。
