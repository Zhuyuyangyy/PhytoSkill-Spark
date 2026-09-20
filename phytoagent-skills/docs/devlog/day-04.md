# 十日谈 · 第四天：从四个Skill到一个产品

日期：2026-09-21。内容为本地工程记录草稿，尚未发布到技术社区。

前三轮把四个能力做成可验证的Skill包，并落地了真实 StepFun 调度层。但把 Studio、AgentShield、植物识别并排摆在一起不构成产品：范围过大，也无法在三分钟内展示行业价值。本次把三者收敛成一个方向——**PhytoForge Spark**，并落地它最核心的三层。

## 做了什么

**Skill Compiler（`compiler/`）。** 自然语言需求进，受约束 Skill 包出。关键决定是**不做代码生成**：Compiler 从已审核能力目录组合工作流，渲染模板并写声明式 JSON，唯一生成的 Python 文件是对所有产物逐字节相同的固定薄执行器。理由是代码生成器无法审计，而交付物需要版本、契约、触发边界和可移植性。

失败全部显式：未知物种、无能力匹配、超长请求、要求网络权限、缺融合步骤、步骤乱序、无负向 eval、名称暗示官方认证——都是 `CompileError`，不留模糊输出。物种到包名走显式白名单，不做音译猜测。

**AgentShield 编译门禁（`shield/gate.py`）。** 六项检查：清单完整性、schema 一致性、最小权限、负向用例覆盖、隐藏指令与危险模式扫描、项目签名。门禁与 Compiler 独立——从磁盘重读并重新校验，Compiler 无法自证清白。无签名密钥时报 `not_run` 而不是 `passed`：没做过的事不能记成通过。未通过则隔离并给可操作修复项。

**AgentShield Runtime 中间件（`runtime/shield.py`）。** 每次调用带 `trace_id` / `tool_call_id`，权限、网络、工具白名单、调用预算实时拦截，拦截发生在执行之前。治理点刻意与 `agentshield-audit` 分离：Agent 选择不调用审计 Skill 时，中间件照样拦。这是"不可绕过"的实质含义。

**Claim-Evidence 审计。** 报告拆成 Claim，每个事实性 Claim 必须关联证据 ID。三类硬拒绝：无证据 ID、过度确定措辞、不可验证来源。可信等级只有 SUPPORTED / LIMITED / INSUFFICIENT，由证据覆盖与硬违规推导，不用 0–100 分。拒绝理由记为 auditor 必须拒绝的 claim，可归因到具体政策。

**本地语料（`corpus/`）。** vendored TCM-Mind-RAG 知识数据副本，带版本与 SHA-256，不 import 原项目。检索器是确定性字符二元组重合度，没有 Embedding 模型也没有向量库。检索不到返回空，不编造引用。

## 一个必须写下来的边界

该语料是 TCM 证型知识，**没有植物生理学证据**。按 `leaf_yellowing` 表型过滤检索结果为空。这是诚实的空结果，不是缺陷——本草药性知识不能替代栽培病理证据。把空结果包装成"检索完成"才是缺陷。

同样必须写下：编译门禁与 Runtime 中间件是**策略层，不是隔离层**。Python 仍运行在本进程，尚无安全沙箱、并发文件修改隔离或强制超时。签名证明来源和完整性，不证明行为安全。

## 可复现证据

| 项目 | 值 |
| --- | --- |
| 测试 | 230 passed，全离线 |
| 契约评测 | 28 / 28 |
| 四 Skill 组合 | completeness: complete |
| 失败降级 | partial, missing_inputs: ["knowledge"] |
| 三分钟演示 | outcome: pass，8 次工具调用，闭环一致 |
| 语料索引 | 540 chunk，sha256 已记录 |

产物在 `artifacts/pytest-v0.3-results.xml`、`artifacts/skill-contract-evals.json`、`artifacts/phytoforge-demo.json`、`artifacts/generated-skills/huangqi-health-assessment/`。

## 仍未做

StepFun、YOLO、Embedding、DGX GPU **一次都没调用**。所有产物里 `agent_model_called` 与 `dgx_hardware_used` 都是 `false`。Agent A/B 未运行，`artifacts/agent-ab.json` 不存在。SkillSpector / OMS 未接入，Registry 的 `scanned` / `evaluated` 仍为 `not_checked`，没有 NVIDIA 官方 Verified 声明。农学准确率与 GPU 性能指标未实测，不提供推算值。

## 下一步

按权重排序，成本最低收益最高的是**接入真实 StepFun 端点跑 A/B**——同一个模型、端点、工具和任务集，只改变是否加载生成的 Skill，比较正确性、触发率、误调用率、证据覆盖率和调用成本。当前三路仍是精确匹配的合成案例，真实端点调用是唯一能改变这一状态的动作。其次是 DGX Spark 上的真实视觉与 Embedding 记录。
