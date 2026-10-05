# PhytoSkill-Spark

面向中药药用植物异常研判的**可组合 Agent Skills** 工程，并在此之上收敛出一个产品形态：**PhytoForge** —— 把专业研判需求编译成可执行、可审计、可评测、可复用的 Agent Skill。

一句话：

> 将一次性的中药植物研判需求，编译成可信的 Agent Skill，并由模型自动调度。

Skill 是交付物，模型负责规划与报告生成。**硬件是配置，不是身份**：推理可跑在本机 CPU、CUDA 工作站、租用 GPU 实例、经 SSH 的远程节点，或完全离线地回放已记录观察。DGX Spark 是其中一种后端，不是前提。

## 仓库结构

```text
.
├── LICENSE                     Apache-2.0
├── docs/
│   ├── contracts/              早期设计契约（input/output schema、registry、tool schema）
│   └── skill-package/          phyto-diagnosis 单包设计稿（design-only，entrypoint 未实现）
└── phytoagent-skills/          0.3.0 主体工程
    ├── backends/               执行后端：协议、ollama HTTP、SSH、离线回放、解析门面
    ├── sdk/                    BaseSkill、契约、Manifest、fixture 约束、SkillSpec schema、CLI
    ├── registry/               发现、按需加载、Ed25519 签名校验、Tool Schema 导出
    ├── runtime/
    │   ├── executor.py         校验后执行源码、保留调用 ID、结构化错误
    │   ├── shield.py           AgentShield Runtime：权限拦截、trace、Claim-Evidence 审计
    │   └── workflow.py         工作流 Skill 执行：解析 $.field，经 Shield 调用提供方
    ├── compiler/               SkillSpec、已审核能力目录、意图解析、模板化 Compiler、CLI
    ├── shield/gate.py          AgentShield 编译门禁：六项检查、隔离、修复项
    ├── corpus/                 vendored 本地语料索引与确定性检索器
    ├── harness/                真实模型调度：配置、传输、工具循环、任务集、规则化评分、A/B
    ├── dgx/                    SSH 传输实现 + 视觉提示词与解析器（历史位置，见下）
    ├── skills/                 plant_vision / growth_risk / herbal_knowledge /
    │                           evidence_fusion / agentshield_audit
    ├── demo/                   phytoforge_demo（三分钟闭环）、四 Skill 组合、SDK 示例
    ├── evals/                  包内 fixture 契约评测器
    ├── tests/                  387 项，全离线
    ├── artifacts/              pytest 结果、契约评测、真实链路与标注产物、发布包
    └── docs/                   开发日志、分轮范围、Harness 与 DGX 运行手册
```

主体工程的详细说明见 **[phytoagent-skills/README.md](phytoagent-skills/README.md)**。

## 快速开始

需要 Python 3.11+。

```powershell
Set-Location phytoagent-skills
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m demo.phytoforge_demo
.\.venv\Scripts\python.exe -m pytest -q
```

以上全部离线，不需要任何硬件或密钥。

### 选择推理在哪儿跑

```bash
export PHYTO_VISION_BACKEND=local_cpu    # 本机 ollama（默认 127.0.0.1:11434）
export PHYTO_VISION_BACKEND=local_cuda   # 本机有 GPU 的 ollama
export PHYTO_VISION_BACKEND=ollama@http://10.0.0.5:11434   # 任何可达的 ollama
export PHYTO_VISION_BACKEND=dgx_spark    # 原 DGX 节点，经 SSH（需 pip install '.[ssh]'）
export PHYTO_VISION_BACKEND=replay       # 离线回放已记录观察
```

不设置则**直接报错，不猜测**。两台机器若各自悄悄选了不同硬件，同一份提交会产出不可比的结果。

接真实模型（需要自备端点与密钥，密钥不入库）：

```bash
cp .env.example .env && chmod 600 .env && $EDITOR .env
python -m harness dry-run      # 不联网，打印脱敏配置与任务集
python -m harness preflight    # 联网自检：可达性、鉴权、工具调用、模型一致性
python -m harness ab --repeat 1 --output artifacts/agent-ab.json
```

## What is verified

下列结论都有仓库内可复现的产物支撑，不是推算值。

| 项目 | 证据 |
|---|---|
| 387 项测试全离线通过 | `pytest -q` |
| StepFun 真实 tool calling | `artifacts/preflight.json`：可达性、鉴权、工具调用三项通过；22 次采样模型身份一致（`step-5-preview`） |
| 真实视觉推理 | `artifacts/dgx/`：30 张图全链路跑通，79 个观察、139 条 supported claims、0 refused；机型 `gx10-9ec6`（aarch64 / NVIDIA GB10 / CUDA 13.0） |
| 30 张图人工标注与评分 | `artifacts/dgx/annotation-score.json`、`baseline-v1.json` |
| Claim–Evidence 审计 | `artifacts/dgx/end-to-end.json`、`error-taxonomy.json` |
| 签名 Skill Registry | Ed25519 校验，`artifacts/registry-demo.json` |
| 观察缓存 | 15 张图 411 s → 12 s（约 34 倍），见 `docs/harness.md` |
| 真实 Agent A/B | `artifacts/agent-ab.json`（fixture 工具）与 `agent-ab-real.json`（真实图像） |

**30 张人工标注的结果（Dev-30，单人标注，非金标）：**

| 口径 | Precision | Recall | F1 |
|---|---|---|---|
| 30 张整体（15 药材切片 + 15 植株叶片） | 0.550 | 0.1864 | 0.2785 |
| 药材切片 15 张 · 校准前 | 0.4706 | 0.1905 | 0.2712 |
| 药材切片 15 张 · 校准后 | 1.000 | 0.3810 | 0.5517 |

校准后精确率到 1.0，是**用召回率换来的**：FP 从 9 降到 0，FN 仍有 26。模型变得更保守，不是更准。而且这是 **Dev calibration，不是 generalisation evidence** —— 这 30 张图在写 prompt 之前就被看过，任何在其上继续调高 F1 的举动都会让数字越来越不可信。

已知系统性错误（`error-taxonomy.json`）：`cut_surface_dense` 系统性误报（5 次），`cut_surface_powder`（15）、`slice_irregular`（8）、`colour_pale_yellow`（7）、`leaf_spot`（5）、`leaf_yellowing`（5）、`wilting`（4）系统性漏报。**叶片异常的召回尤其低。**

## What is not yet verified

| 缺口 | 说明 |
|---|---|
| 校准的泛化能力 | 无 blind holdout。下一步是冻结 prompt 后在从未看过的图上只跑一次 |
| 未见图的准确率 | 同上 |
| 生产级沙箱隔离 | 编译门禁与 Runtime 中间件是**策略层，不是隔离层**；Python 仍在本进程运行 |
| 统计显著性 | A/B 为 `repeat=1`，差值只作方向性观察。已观测到的差值：触发通过率 **0.0**，覆盖率 +0.1，越权调用 −2 |
| 标注一致性 | 单人标注，无第二标注者，未测 inter-annotator agreement |

## 边界与已知遗留

- 视觉 `fixture` 模式是**精确匹配的合成案例**：换图、换物种、换测量值会明确失败，不会用同一份输出回答不同输入。`live` / `herb` 才是真实推理。
- 本地语料是 TCM 证型知识，**没有植物生理学证据**。这是知识侧最大的短板。
- Compiler **不生成任意代码**：渲染模板并写出声明式配置，唯一生成的 Python 文件是对所有产物逐字节相同的固定薄执行器。
- 融合是确定性 ID 连接逻辑，不把视觉分数与检索分数组合成疾病概率。
- Registry 的 `scanned` / `evaluated` 仍为 `not_checked`；未接入 SkillSpector 或 OMS，**没有** NVIDIA 官方 Verified 声明。项目内部可称 `Validated`，不能冒充官方认证。
- `manifest.sig` 是项目自有 Ed25519 格式，与 OpenSSF Model Signing 的 `skill.oms.sig` 不能互换。
- 私钥不随源码发布，开发私钥为未加密 PEM；Demo 每次生成临时密钥，仅证明本次签名校验流程。
- 未实现的模式明确失败，不回退 fixture。离线部分不联网；`harness/` 与后端层是仅有的会发起网络请求的部分，缺密钥直接报错而非降级。

**遗留项（有意未在本轮处理）：**

1. **解析实现仍在 `dgx/`。** `backends/parsing.py` 现在是门面，真正的提示词与解析器还在 `dgx/vision.py`、`dgx/herb_vision.py` —— 那是 DGX 作为唯一后端时写下它们的地方。下移到 `backends/` 是独立一轮，因为四个测试文件直接引用了那些符号。
2. **`dgx_hardware_used` 字段名是历史遗留。** 它现在的语义是「本次是否观测到 GPU」，取值由后端真实报告决定，与品牌无关。改名会波及 8 个模块、两个 Skill 包的 schema 与全部既有产物，因此保留并加了 schema 说明。
3. **包内 `permissions.network` 仍声明 `deny`，** 而真实后端（ollama HTTP / SSH）会走网络。这是既有的一致性问题，属于 AgentShield 策略范围，本轮按要求冻结未动。
4. 项目名仍含 `Spark`。本轮只解耦运行时；改名涉及包名、pyproject、远端仓库名，留到后端层稳定后一次做完。

## 许可

本仓库代码以 **Apache-2.0** 授权，全文见 [LICENSE](LICENSE)。

四个 Skill 包内的 Skill Card 仍标 `NOASSERTION`：包内文件受 Manifest 哈希与 Ed25519 签名保护，改动后需重新封包并重签，因此许可尚未同步到包内。
