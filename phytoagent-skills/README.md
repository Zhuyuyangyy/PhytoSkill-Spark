# PhytoSkill-Spark · 可组合的中药植物 Agent Skills

PhytoSkill-Spark 把药用植物异常研判拆成四个可独立发现、调用和验证的专家能力包，并在其上收敛出一个产品：**PhytoForge Spark** —— 基于 NVIDIA DGX Spark 的可信中药植物 Skill 编译与执行系统。

一句话：

> 将一次性的中药植物研判需求，编译成可执行、可审计、可评测、可复用的专业 Agent Skill，并由 StepFun 自动调度。

**0.3.0 新增：Skill Compiler（`compiler/`）、AgentShield 编译门禁（`shield/gate.py`）、AgentShield Runtime 中间件与 Claim-Evidence 审计（`runtime/shield.py`）、工作流 Skill 执行器（`runtime/workflow.py`）、可调用审计 Skill（`skills/agentshield_audit/`）、本地语料检索（`corpus/`）与三分钟演示（`demo/phytoforge_demo.py`）。**

测试 **233 项，全部离线通过**。契约评测 **28/28**。

**真实模型已跑通**：`step-5-preview` 上 preflight 三项硬检查全过，A/B 两臂各 17 个任务，产物 `artifacts/agent-ab.json`。真实差异见 [真实 Harness](docs/harness.md)。

**DGX Spark 已连通**：`gx10-9ec6`（aarch64 / NVIDIA GB10 / CUDA 13.0 / Docker），通过 ollama 跑通真实视觉推理，`plant_vision` 新增 `live` 模式，产物 `artifacts/dgx/`。见 [DGX 实测](docs/dgx-spark.md)。

## 立即运行

需要 Python 3.11+，PowerShell 示例：

```powershell
Set-Location 'D:\DRX SPARK\phytoagent-skills'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m demo.phytoforge_demo      # 三分钟完整闭环
.\.venv\Scripts\python.exe -m demo.run_demo             # 四 Skill fixture 组合
.\.venv\Scripts\python.exe -m evals.run_contracts
.\.venv\Scripts\python.exe -m pytest -q
```

`demo.phytoforge_demo` 依次展示：自然语言需求 → Compiler 生成受约束 Skill 包 → AgentShield 六项编译门禁 → 通过门禁才进 Registry → 真实 Tool Calling Trace → Claim-Evidence 可信结果 → 模糊图负向拒绝 → 再次调用 Registry 中的同一 Skill 证明闭环。

## 三层职责

| 层 | 模块 | 职责 |
| --- | --- | --- |
| Skill 编译 | `compiler/` | 自然语言需求 → 受约束 `SkillSpec` → 声明式 Skill 包 |
| 可信运行时 | `shield/gate.py` + `runtime/shield.py` + `runtime/workflow.py` | 编译门禁、权限拦截、trace_id、Claim-Evidence 审计 |
| 专业能力 | `skills/` | plant_vision / growth_risk / herbal_knowledge / evidence_fusion / agentshield_audit |

StepFun 仍是唯一的 Agent 大脑：意图解析、Compiler 调用、Skill 选择、Tool Calling 与最终报告。DGX Spark 是本地执行底座（远程 SSH / 容器形态）。

## Skill Compiler

Compiler **不生成任意 Python**。它从经过审核的能力目录（`compiler/catalog.py`）组合工作流，渲染文本模板并写出声明式配置；唯一生成的 Python 文件是固定薄执行器，对所有编译产物逐字节相同。

```bash
python -m compiler compile \
  --request "创建一个黄芪健康研判Skill，根据叶片图片、环境参数和知识库分析异常原因，并要求每个结论给出证据。" \
  --output skills

python -m compiler verify --spec skill_spec.json   # 只校验，不落盘
python -m compiler list                            # 已审核能力目录
```

生成的 `huangqi-health-assessment` 是轻量工作流包：只声明依赖、顺序、输入映射和拒绝策略，不复制四个专业 Skill 的实现。

```
huangqi-health-assessment/
├── SKILL.md  skill-card.md  schema.json
├── workflow.json            # 声明式执行计划
├── skill_spec.json          # 受约束的编译契约
├── skill.py                 # 固定薄执行器（模板，非生成代码）
├── evals/evals.json         # 正向 / 缺参数 / 负向触发
├── examples/request.json
├── references/contract.md  BENCHMARK.md  metadata.json
└── manifest.json
```

**拒绝优于猜测**：未知物种、空能力匹配、超长请求、要求网络权限、缺少融合步骤、步骤乱序、缺少负向 eval、名称暗示官方认证 —— 全部是硬失败，不留模糊输出。

## AgentShield：不可绕过的可信运行时

两个关注点刻意分离：

* **编译门禁** `shield/gate.py` —— 包在进入 Registry 前必须通过六项检查：`manifest_integrity`、`schema_conformance`、`permission_least_privilege`、`negative_eval_coverage`、`hidden_instruction_scan`、`project_signature`。门禁与 Compiler 独立：它从磁盘重读包并重新校验，Compiler 无法自证清白。未通过则隔离并给出可操作修复项。
* **Runtime 中间件** `runtime/shield.py` —— 为每次调用生成 `trace_id` / `tool_call_id`，实时拦截路径、网络、工具白名单与调用预算，记录 Skill 版本、Manifest 哈希、参数摘要与耗时。

`agentshield-audit` 是**可调用的审计 Skill**，但真正的权限拦截位于 Runtime 中间件。Agent 选择"不调用审计 Skill"不能绕过治理。

### Claim-Evidence 审计与可信等级

执行后把报告拆成 Claim，要求每个事实性 Claim 关联 Evidence ID，并检查物种、案例、时间和单位一致性。不使用虚假的 0–100 分：

| 等级 | 含义 |
| --- | --- |
| `SUPPORTED` | 证据齐备、一致性通过、无硬性违规 |
| `LIMITED` | 非关键来源缺失，只允许描述有限趋势 |
| `INSUFFICIENT` | 图像不可用、证据冲突、来源不可验证或权限违规，拒绝原因判断 |

最终输出示例：

```json
{
  "assessment": "观察到叶缘黄化区域，环境应激是待复核因素之一",
  "trust_level": "SUPPORTED",
  "claims": [
    {"text": "观察到叶缘黄化区域", "evidence_ids": ["obs-yellow-01"], "status": "supported"}
  ],
  "refused_claims": [],
  "trace_id": "trace-demo-001"
}
```

## 本地语料检索

`corpus/` 是 TCM-Mind-RAG 知识数据的 **vendored 副本**，带 SHA-256 与 `corpus_version`。不 import、不 shell out、不读取原项目。检索器是确定性字符二元组重合度，**没有 Embedding 模型，没有向量库**，`retrieval_score` 只反映字面重合度。

`herbal_knowledge` 因此有两个显式模式：

* `fixture` —— 已发布的合成案例，换任何输入明确失败。
* `corpus` —— 查询本地语料，返回证据 ID、文件位置（`source_file:line_start-line_end`）与语料版本。**检索不到就返回空**，不编造引用。

必须说明的边界：该语料是 TCM 证型知识，**没有植物生理学证据**，因此按表型过滤的检索结果为空。本草药性知识不能替代栽培病理证据。

## 工程结构

```text
phytoagent-skills/
├── sdk/                   BaseSkill、契约、Manifest、fixture约束、CLI、SkillSpec schema
├── registry/              发现、按需加载、签名校验、Tool Schema导出
├── runtime/
│   ├── executor.py        校验后执行源码、保留调用ID、结构化错误
│   ├── shield.py          AgentShield Runtime：权限拦截、trace、Claim-Evidence审计
│   └── workflow.py        工作流 Skill 执行：解析 $.field、经 Shield 调用提供方
├── compiler/              SkillSpec、能力目录、意图解析、模板化 Compiler、CLI
│   └── templates/         固定薄执行器模板
├── shield/gate.py         AgentShield 编译门禁（六项检查 + 隔离 + 修复项）
├── corpus/                vendored 语料索引与确定性检索器
├── harness/               真实StepFun调度：配置、传输、工具循环、任务集、规则化评分、A/B
├── skills/
│   ├── plant_vision/  growth_risk/  herbal_knowledge/  evidence_fusion/
│   └── agentshield_audit/
├── demo/                  phytoforge_demo（三分钟闭环）、四Skill组合、SDK示例
├── evals/                 包内fixture契约评测器
├── scripts/seal_skills.py 发布者显式封包
├── dgx/                   DGX Spark SSH 客户端、真实视觉推理、端到端脚本
└── tests/                 271 项全离线测试
```

## 治理与评测的实际范围

| 项目 | 当前状态 |
| --- | --- |
| Compiler | 模板化组合，不生成任意代码；拒绝优于猜测 |
| 编译门禁 | 六项检查；未通过隔离并给修复项；与 Compiler 独立 |
| Runtime 拦截 | 权限 / 网络 / 工具白名单 / 调用预算；每次调用带 trace_id |
| Claim-Evidence 审计 | 无证据ID、过度确定、不可验证来源一律拒绝 |
| Catalog / Documented | 五包可发现；含触发边界、说明、契约和 Skill Card |
| Integrity / Signed | SHA-256 清单 + 项目 Ed25519 签名；公钥在包外固定 |
| Fixture contracts | `evals.run_contracts` 28/28 |
| DGX Spark 实测 | 已连通 `gx10-9ec6`（GB10 / aarch64 / CUDA 13.0），真实视觉推理已跑通，`artifacts/dgx/` |
| plant_vision live 模式 | 真实 GPU 推理；无 fixture 回退，缺节点或缺密钥即报错 |
| Agent 触发与 A/B | 每包有正向、负向、缺参数用例；**已真实运行**：`step-5-preview`，17 任务 × 2 臂，见 `artifacts/agent-ab.json` |
| SkillSpector / OMS | 未接入；**没有** NVIDIA 官方 Verified 声明 |
| DGX / 模型准确率 | **未实测**，不提供推算指标 |
| 本地语料 | 已 vendored 并带哈希；确定性检索，无 Embedding / 向量库 |

`manifest.sig` 是项目格式，与 OpenSSF Model Signing 的 `skill.oms.sig` 不能互换。临时 Demo 密钥只验证本次流程，不认证第三方发布者。

## 执行边界

SDK 使用 Draft 2020-12，只允许本地 JSON Pointer 引用；拒绝非有限数字、重复 JSON 键、越界路径和链接资源。执行器载入源码前检查模式和输入，并校验源码哈希，不执行未覆盖的缓存字节码。

必须明确：Python 仍运行在本进程，**尚无安全沙箱**、并发文件修改隔离或强制超时。签名证明来源和完整性，不证明行为安全。编译门禁与 Runtime 中间件是策略层，不是隔离层。

未实现的 `live`/`replay` 明确失败，不回退 fixture。`harness/` 是唯一会发起网络请求的模块，必须显式配置端点与密钥（`.env` 已被 gitignore），缺密钥直接报错而不是退回 fixture。密钥不入库。

A/B 已真实运行，但 `repeat=1`、17 个任务，样本量不足以支撑统计显著性；报告里 `significance_test` 是 `none`，差值只能当方向性观察。免费额度 10 RPM，`harness/transport.py` 因此有请求间隔节流；跑更多轮次受限于配额。

本仓库代码以 Apache-2.0 授权，见仓库根 `LICENSE`。Skill 包内 Skill Card 仍标 `NOASSERTION`：包内文件受 Manifest 哈希与 Ed25519 签名保护，改动后需重新封包并重签，因此许可尚未随源码同步，属已知遗留项。
