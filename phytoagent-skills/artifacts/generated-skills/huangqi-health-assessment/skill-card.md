---
name: huangqi-health-assessment
version: 0.3.0
owner: PhytoForge Spark compiler
license: NOASSERTION
license_status: owner_decision_required_before_public_release
deployment_geography: local-development-only
runtime: local_python
execution_modes: [fixture]
network_access: deny
filesystem_permissions: [case_workspace:read, corpus:read, package:read]
tool_dependencies: [plant_vision, growth_risk, herbal_knowledge, evidence_fusion]
model_weights: none
nvidia_verified: false
security_scan: not_run
agent_ab_evaluation: not_run
---

# Skill Card

工作流包，不含模型权重，不联网。

## 权限

- 文件系统：case_workspace:read, corpus:read, package:read（只读，按最小权限授予）。
- 网络：deny。
- 工具：仅可调用声明的专业 Skill。

## 风险与限制

- 本包不生成任意代码；执行器为固定模板。
- 融合结果为 co_occurrence_only，不是因果结论，也不是疾病概率。
- 表型观察不能确诊病原；本草药性知识不能替代栽培病理证据。
