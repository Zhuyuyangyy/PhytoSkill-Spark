# PhytoSkill-Spark

面向中药药用植物异常研判的**可组合 Agent Skills** 工程。把「看见叶片异常 → 查环境因素 → 找文献证据 → 融合成结论」拆成四个可独立发现、调用、验证的能力包，配一套本地 SDK、Registry、签名校验与契约评测。

Skill 是交付物，StepFun 负责规划与报告生成，DGX Spark 是计划中的本地执行底座。

## 仓库结构

```text
.
├── docs/
│   ├── contracts/              早期设计契约（input/output schema、registry、StepFun tool schema）
│   └── skill-package/          phyto-diagnosis 单包设计稿（design-only）
└── phytoagent-skills/          0.2.0 主体工程
    ├── sdk/                    BaseSkill、契约、Manifest、fixture 约束、CLI
    ├── registry/               发现、按需加载、Ed25519 签名校验、Tool Schema 导出
    ├── runtime/executor.py     校验后执行源码、保留调用 ID、结构化错误
    ├── skills/                 plant_vision / growth_risk / herbal_knowledge / evidence_fusion
    ├── demo/                   正常组合与失败降级演示
    ├── evals/                  包内 fixture 契约评测器
    ├── tests/                  SDK、签名、调用、组合、降级测试
    ├── artifacts/              pytest 结果、契约评测、release 校验记录、发布包
    └── docs/                   比赛资料对照、开发日志、分轮范围
```

主体工程的详细说明见 **[phytoagent-skills/README.md](phytoagent-skills/README.md)**。

## 快速开始

需要 Python 3.11+。

```powershell
Set-Location phytoagent-skills
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m demo.run_demo
.\.venv\Scripts\python.exe -m evals.run_contracts
.\.venv\Scripts\python.exe -m pytest -q
```

## 当前状态与边界

0.2.0 已完成 SDK、Registry、四个 Skill 接口、确定性证据融合，以及离线 fixture 组合与失败降级。

需要明确的是：

- 视觉、环境、知识三路**均使用精确匹配的合成案例**，未读取真实图片、未调用 YOLO / TCM-Mind-RAG / StepFun / DGX GPU。
- 更换图片、物种、案例 ID 或测量值会**明确失败**，不会用同一份合成输出回答真实输入。
- 融合为确定性 ID 连接逻辑（真实执行），不把视觉分数与检索分数组合成疾病概率。
- Registry 的 `scanned` / `evaluated` 状态仍为 `not_checked`；未接入 SkillSpector 或 OMS，**没有** NVIDIA 官方 Verified 声明。
- 项目自有的 `manifest.sig` 是 Ed25519 格式，与 OpenSSF Model Signing 的 `skill.oms.sig` 不能互换。
- 未实现的 live / replay 模式明确失败，不回退 fixture；现阶段不联网。
- 私钥不随源码发布，开发私钥为未加密 PEM；Demo 每次生成临时密钥，仅证明本次签名校验流程。

Agent A/B、农学准确率与 GPU 性能指标**尚未实测**，仓库内不提供推算值。

## 许可

开源许可尚待所有者确认，当前 Skill Card 标记为 `NOASSERTION`。
