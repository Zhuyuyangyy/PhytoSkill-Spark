# PhytoForge Spark · 架构与实现说明

本文记录 0.3.0 的三个新增层：Skill Compiler、AgentShield 编译门禁与 Runtime 中间件、本地语料检索。所有结论都来自本仓库可复现的运行，未实测的一律标注。

## 一、为什么是"编译"而不是"生成代码"

一次性的 Prompt 无法承担版本、契约、触发边界、评测任务和可移植性。但把自然语言直接交给代码生成器会得到一个无法审计的产物。PhytoForge Spark 取中间路线：

**Compiler 只做组合，不做发明。**

```
自然语言需求
   │  compiler/intent.py        确定性关键词解析（无模型）
   ▼
Intent {species, task, capabilities, inputs}
   │  compiler/compiler.py      结构不变量检查 + JSON Schema 校验
   ▼
SkillSpec（受约束契约）
   │  渲染文本模板 + 写声明式 JSON
   ▼
Skill 包目录
   │  shield/gate.py            六项编译门禁，从磁盘重读
   ▼
Registry（仅通过门禁的版本）
```

`compiler/catalog.py` 是唯一的能力来源。Compiler 只能从这里选条目；任何不在目录里的能力会被门禁拒绝，而不是被生成。

生成的 `skill.py` 是 `compiler/templates/workflow_executor.py` 的逐字节副本。它不含任务特定逻辑：读 `workflow.json`、校验输入映射、组装请求。因此两个不同物种编译出的 `skill.py` 完全相同——这一点由 `tests/test_compiler.py::test_the_emitted_executor_is_a_fixed_template_not_generated_code` 守住。

它**不调用提供方 Skill**。Skill 包是可移植产物，必须能在只安装了本包的机器上运行；从包内调用提供方会要求那些包同时安装并签名，这既让"可移植 Skill"不成立，也让工作流包能触达门禁从未针对该组合审查过的代码。委派因此发生在 `runtime/workflow.py`，那里有完整签名集合且每次调用都经过 AgentShield。单独运行工作流包会把每步报为 `not_delegated`、整体 `INSUFFICIENT`，从不为没发生的工作报 `success`。

## 二、拒绝优于猜测

Compiler 的失败是显式的，不留模糊输出：

| 触发条件 | 结果 |
| --- | --- |
| 物种不在已批准目录 | `CompileError` |
| 无任何能力关键词匹配 | `CompileError` |
| 请求超过 4000 字符 | `CompileError` |
| SkillSpec 要求网络权限 | `CompileError` |
| 工作流缺少 `evidence_fusion` | `CompileError` |
| 工作流步骤乱序 | `CompileError` |
| 无负向 eval | `CompileError` |
| 负向 eval id 重复 | `CompileError` |
| 输入映射逃出请求文档 | `CompileError` |
| 声明的输入与工作流实际读取的字段不一致 | `CompileError` |
| 名称含 `nvidia` / `verified` / `official` / `oms` | `CompileError` |
| 只有环境或知识触发器，没有图片触发器 | `CompileError`（没有可评估的表型） |

### 一条必须写下来的限制

提供方 Skill 只发布了**一个**合成 fixture，且是黄芪案例。任何其他物种编译出的包，其正向用例**不可能通过**——提供方会明确拒绝那些输入。

Compiler 不把这种用例伪装成通过的用例。`evals.json` 里它们是 `"status": "declared_only"`，带明确理由，文件级记 `"benchmark_status": "NOT RUN"`。`tests/test_compiler.py::test_a_package_for_a_species_without_a_fixture_does_not_claim_a_pass` 钉住了这一点。

物种到包名的映射是显式白名单（`SPECIES_SLUGS`），不做音译猜测：包名必须是 ASCII 小写，而"雪莲"没有批准的拉丁 slug，于是整体编译失败。

## 三、AgentShield 编译门禁

`shield/gate.py` 与 Compiler **独立**：它从磁盘重读已写入的包，重新校验清单、重新解析 schema、重新扫描文本。Compiler 无法给自己的输出打勾。

六项检查：

| 检查 | 内容 |
| --- | --- |
| `manifest_integrity` | 文件清单 SHA-256 + 元数据哈希 |
| `schema_conformance` | 声明的 input/output schema 可解析；有 `skill_spec.json` 时名称须与包一致 |
| `permission_least_privilege` | `network` 必须 `deny`；只允许白名单只读权限；拒绝通配符与写权限；能力须在已审核集合内 |
| `negative_eval_coverage` | 必须有正向、缺参数、负向三类用例 |
| `hidden_instruction_scan` | 拒绝 `subprocess` / `os.system` / `eval(` / `exec(` / `socket` / `requests` / `__import__` / shell 载荷 / 隐藏指令 / 外传意图 / 凭据读取 |
| `project_signature` | 用包外固定公钥验 Ed25519 签名；**无密钥时报 `not_run`，绝不算通过** |

未通过则 `quarantined=True`，并给出可操作修复项（`repair_items()`），例如"Remove undeclared, wildcard or write permissions; keep network: deny."

门禁支持两种包作者风格：Compiler 产物用 `cases` + SkillSpec `negative_evals`；手工包用 `contract_cases` + `agent_cases`。两者都必须展示正向、缺参数、负向覆盖。

## 四、AgentShield Runtime 中间件

`runtime/shield.py` 的 `ShieldRuntime` 包装 `SkillExecutor`，在**每次**工具调用前后生效：

- `trace_id` 由会话生成；`tool_call_id` 必须由调用方提供，缺失即 `TraceError`。
- 调用预算超限即 `BudgetExceeded`。
- 权限按 manifest 声明做最小权限校验：未声明的文件系统权限、任何网络请求都会被 `PermissionViolation` 拦在**执行之前**，并记入 `blocked`。
- 记录 Skill 版本、Manifest 哈希、耗时、使用的权限与证据 ID。

**治理点不是可调用 Skill。** `skills/agentshield_audit` 是审计入口，但 Agent 选择不调用它时，中间件照样拦截。`tests/test_shield.py::test_skipping_the_audit_skill_does_not_remove_interception` 守住了这一点。

`runtime/workflow.py` 的 `WorkflowRunner` 是工作流 Skill 的实际执行者：解析 `$.field` 引用、经 Shield 调用提供方、把结果按 `result_key` 汇回请求、再交给 Claim-Evidence 审计。提供方缺失时步骤记为 `skipped`，不伪造。

## 五、Claim-Evidence 审计

执行后把报告拆成 Claim。四类硬拒绝：

1. **指名病因** —— `病原是` / `病因是` / `感染了` / `由.*引起` 等。指名病原就是诊断，任何词面证据都承载不了。
2. **过度确定表述** —— `确诊` / `已确定` / `一定是` / `100%` 等确定性措辞。
3. **无证据 ID** —— `every_claim_requires_evidence`。
4. **不可验证来源** —— 证据 ID 不在本次运行的证据索引中。**空索引同样拒绝**：一个无法交代的 ID 不可验证，静默接受它会让编造引用拿到 `SUPPORTED`。

`agentshield-audit` Skill 的顶层裁决多一档：任一项检查是 `not_run` 时判 `INCOMPLETE`、`trust_level` 降到 `LIMITED`，并在 limitations 里列出未执行的检查。没做过的事不能记成通过——只降裁决而留着 `SUPPORTED` 会读成背书。

可信等级由证据覆盖与硬违规推导，**不是** 0–100 分：

| 条件 | 等级 |
| --- | --- |
| 有硬违规或无 supported claim | `INSUFFICIENT` |
| 图像可用、证据齐备、无缺失、无未连接项 | `SUPPORTED` |
| 有 supported claim 但缺来源 / 缺图像 / 证据 ID 少于 2 | `LIMITED` |

拒绝理由被记录为 auditor 必须拒绝的 claim，因此可以归因到具体政策，而不是静默消失。

## 六、本地语料

`corpus/` 是 TCM-Mind-RAG 知识数据的 **vendored 副本**（540 个 chunk，来自 `syndromes.yaml`、`herb_interactions.yaml`、`synthetic_cases.json`），带 `corpus_version` 与索引 SHA-256。不 import、不 shell out、不读取原项目。

检索器 `corpus/retriever.py` 是确定性字符二元组重合度，**没有 Embedding 模型，没有向量库**。`min_score` 默认 0.05：重合度为零的 chunk 不是匹配，返回它就是编造引用。

`herbal_knowledge` 因此有两个显式模式：

- `fixture` —— 已发布合成案例，换任何输入明确失败。
- `corpus` —— 返回证据 ID、`source_file:line_start-line_end`、语料版本。检索不到返回空 evidence 并记 limitations。

**必须说明的边界**：该语料是 TCM 证型知识，没有植物生理学证据。按 `leaf_yellowing` 过滤检索结果为空——这是诚实的空结果，不是缺陷。本草药性知识不能替代栽培病理证据。

## 七、实测记录

| 项目 | 值 | 命令 |
| --- | --- | --- |
| 测试 | **233 passed**，全离线 | `python -m pytest -q` |
| 契约评测 | **28 / 28** | `python -m evals.run_contracts` |
| 四 Skill 组合 | `completeness: complete` | `python -m demo.run_demo` |
| 失败降级 | `completeness: partial`, `missing_inputs: ["knowledge"]` | `python -m demo.run_demo --simulate-failure herbal_knowledge` |
| 三分钟演示 | `outcome: pass`（该值由运行结果推导，非预设） | `python -m demo.phytoforge_demo` |
| 真实 preflight | `passed: true`，`failures: []`，22 次采样身份一致 | `python -m harness preflight` |
| 真实 A/B | 两臂各 17 任务，0 失败；`forbidden_violations` 2→0 | `python -m harness ab --repeat 1` |
| 语料索引 | 540 chunk，sha256 已记录 | `python -m corpus.vendor --source ...` |

对应产物：`artifacts/pytest-v0.3-results.xml`、`artifacts/skill-contract-evals.json`、`artifacts/phytoforge-demo.json`、`artifacts/generated-skills/huangqi-health-assessment/`。

## 八、明确未做

- **StepFun 已真实调用，其余均未**：`step-5-preview` 上 preflight 三项硬检查全过，
  A/B 两臂各 17 个任务，产物 `artifacts/agent-ab.json`。但 YOLO、Embedding、
  DGX GPU 仍未调用，`dgx_hardware_used` 在所有产物中都是 `false`。
- **A/B 样本量不足**：`repeat=1`、17 个任务，`significance_test` 是 `none`。
  差值只能当方向性观察，不能当结论。免费额度 10 RPM 限制了跑更多轮次。
- **未接入 SkillSpector / OMS**：Registry 的 `scanned` / `evaluated` 仍为 `not_checked`，
  无 NVIDIA 官方 Verified 声明。
- **无农学准确率、无 GPU 性能指标**：工具返回内容仍是 fixture，无法支撑真实诊断结论。不提供推算值。
- **策略层不是隔离层**：Python 仍运行在本进程，尚无安全沙箱、并发文件修改隔离或强制超时。
