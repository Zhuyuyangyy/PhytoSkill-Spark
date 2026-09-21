# PhytoSkill-Spark

面向中药药用植物异常研判的**可组合 Agent Skills** 工程，并在此之上收敛出一个产品：**PhytoForge Spark** —— 基于 NVIDIA DGX Spark 的可信中药植物 Skill 编译与执行系统。

一句话：

> 将一次性的中药植物研判需求，编译成可执行、可审计、可评测、可复用的专业 Agent Skill，并由 StepFun 自动调度。

Skill 是交付物，StepFun 负责规划与报告生成，DGX Spark 是本地执行底座（远程 SSH / 容器形态）。

## 仓库结构

```text
.
├── LICENSE                     Apache-2.0
├── docs/
│   ├── contracts/              早期设计契约（input/output schema、registry、StepFun tool schema）
│   └── skill-package/          phyto-diagnosis 单包设计稿（design-only，entrypoint 未实现）
└── phytoagent-skills/          0.3.0 主体工程
    ├── sdk/                    BaseSkill、契约、Manifest、fixture 约束、SkillSpec schema、CLI
    ├── registry/               发现、按需加载、Ed25519 签名校验、Tool Schema 导出
    ├── runtime/
    │   ├── executor.py         校验后执行源码、保留调用 ID、结构化错误
    │   ├── shield.py           AgentShield Runtime：权限拦截、trace、Claim-Evidence 审计
    │   └── workflow.py         工作流 Skill 执行：解析 $.field，经 Shield 调用提供方
    ├── compiler/               SkillSpec、已审核能力目录、意图解析、模板化 Compiler、CLI
    ├── shield/gate.py          AgentShield 编译门禁：六项检查、隔离、修复项
    ├── corpus/                 vendored 本地语料索引与确定性检索器
    ├── harness/                真实 StepFun 调度：配置、传输、工具循环、任务集、规则化评分、A/B
    ├── skills/                 plant_vision / growth_risk / herbal_knowledge /
    │                           evidence_fusion / agentshield_audit
    ├── demo/                   phytoforge_demo（三分钟闭环）、四 Skill 组合、SDK 示例
    ├── evals/                  包内 fixture 契约评测器
    ├── tests/                  233 项，全离线
    ├── artifacts/              pytest 结果、契约评测、release 校验记录、演示记录、发布包
    └── docs/                   比赛资料对照、开发日志、分轮范围、Harness 运行手册
```

主体工程的详细说明见 **[phytoagent-skills/README.md](phytoagent-skills/README.md)**。

## 快速开始

需要 Python 3.11+。

```powershell
Set-Location phytoagent-skills
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m demo.phytoforge_demo
.\.venv\Scripts\python.exe -m demo.run_demo
.\.venv\Scripts\python.exe -m evals.run_contracts
.\.venv\Scripts\python.exe -m pytest -q
```

三分钟演示依次展示：自然语言需求 → Compiler 生成受约束 Skill 包 → AgentShield 六项编译门禁 → 通过门禁才进 Registry → 真实 Tool Calling Trace → Claim-Evidence 可信结果 → 模糊图负向拒绝 → 再次调用 Registry 中的同一 Skill 证明闭环。

接真实模型（需要自备 StepFun 端点与密钥，密钥不入库）：

```bash
cp .env.example .env && chmod 600 .env && $EDITOR .env
python -m harness dry-run      # 不联网，打印脱敏配置与任务集
python -m harness preflight    # 联网自检：可达性、鉴权、工具调用、模型一致性
python -m harness ab --repeat 1 --output artifacts/agent-ab.json
```

详见 [phytoagent-skills/docs/harness.md](phytoagent-skills/docs/harness.md)。

## 当前状态与边界

0.3.0 已完成 Skill Compiler、AgentShield 编译门禁与 Runtime 中间件、Claim-Evidence 审计、可调用审计 Skill、本地语料检索、工作流 Skill 执行与三分钟演示。

需要明确的是：

- 视觉、环境、知识三路**均使用精确匹配的合成案例**，未读取真实图片、未调用 YOLO / StepFun / DGX GPU。本地语料检索是真实代码执行，但语料是 TCM 证型知识，**没有植物生理学证据**。
- 更换图片、物种、案例 ID 或测量值会**明确失败**，不会用同一份合成输出回答真实输入。
- Compiler **不生成任意代码**：它渲染模板并写出声明式配置，唯一生成的 Python 文件是对所有产物逐字节相同的固定薄执行器。
- 融合为确定性 ID 连接逻辑（真实执行），不把视觉分数与检索分数组合成疾病概率。
- `artifacts/` 中的现有报告**全部来自 fixture 或本地确定性代码**，`agent_model_called` 为 `false`。
- Harness **已真实运行**：`step-5-preview` 上 preflight 三项硬检查全过，A/B 两臂各 17 个任务，产物 `artifacts/agent-ab.json`。即便运行，工具返回的内容仍是 fixture，`dgx_hardware_used` 也如实记录是否触及 GPU。`repeat=1`、17 个任务不足以支撑统计显著性，差值只作方向性观察。
- Registry 的 `scanned` / `evaluated` 状态仍为 `not_checked`；未接入 SkillSpector 或 OMS，**没有** NVIDIA 官方 Verified 声明。项目内部可称 `Validated`，不能冒充官方认证。
- 项目自有的 `manifest.sig` 是 Ed25519 格式，与 OpenSSF Model Signing 的 `skill.oms.sig` 不能互换。
- 编译门禁与 Runtime 中间件是**策略层，不是隔离层**：Python 仍运行在本进程，尚无安全沙箱、并发文件修改隔离或强制超时。
- 未实现的 live / replay 模式明确失败，不回退 fixture。离线部分不联网；`harness/` 是唯一会发起网络请求的模块，缺密钥直接报错而非降级。
- 私钥不随源码发布，开发私钥为未加密 PEM；Demo 每次生成临时密钥，仅证明本次签名校验流程。
- Agent A/B、农学准确率与 GPU 性能指标**尚未实测**，仓库内不提供推算值。

## 许可

本仓库代码以 **Apache-2.0** 授权，全文见 [LICENSE](LICENSE)。

四个 Skill 包内的 Skill Card 仍标 `NOASSERTION`：包内文件受 Manifest 哈希与 Ed25519 签名保护，改动后需重新封包并重签，因此许可尚未同步到包内，属已知遗留项。
