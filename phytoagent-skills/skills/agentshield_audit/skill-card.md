---
name: agentshield-audit
version: 0.3.0
owner: PhytoForge Spark contributors
license: NOASSERTION
license_status: owner_decision_required_before_public_release
deployment_geography: local-development-only
runtime: local_python
execution_modes: [fixture]
network_access: deny
filesystem_permissions: [package:read]
tool_dependencies: [none]
model_weights: none
nvidia_verified: false
security_scan: not_run
agent_ab_evaluation: not_run
---

# Skill Card

对Skill包与Agent报告做离线可信审计。

## 数据与权限

只读取包内资源、被审计包目录与调用方提供的JSON；不联网、不读取模型密钥。
权限最小化为 `package:read`；无工具依赖，无模型权重。

## 风险与限制

- 本Skill是可调用审计入口，不是权限执行点；拦截由 Runtime Middleware 完成。
- 确定性规则无法覆盖全部对抗输入；审计结论需与 Runtime trace 一起阅读。
- 不签发签名、不修改 manifest、不构成第三方认证或 NVIDIA 官方 Verified。

## 验证与来源

项目内签名文件为 manifest.sig，公钥固定在包外。NVIDIA 的 skill.oms.sig / OMS
证书链是另一套验证，当前未接入，不能改名冒充。
评测范围见 BENCHMARK.md；行为用例见 evals/evals.json。
