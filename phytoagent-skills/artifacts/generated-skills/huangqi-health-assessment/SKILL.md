---
name: huangqi-health-assessment
description: "黄芪健康研判工作流Skill：按声明的顺序调用专业Skill，每个结论必须关联证据ID，证据不足时拒绝判断。原始图片分析、无图片无测量的凭空评估、人体疾病诊断和开处方不使用。"
license: NOASSERTION
---

# huangqi-health-assessment

本Skill是**轻量工作流包**：只声明依赖、顺序、输入映射和拒绝策略，不复制 plant_vision / growth_risk / herbal_knowledge / evidence_fusion 的实现。

## 触发

输入包含图片、带单位测量或本地语料检索需求，且物种为黄芪时触发。

## 不触发

- 纯知识问答，没有图片、测量或语料检索需求。
- 要求编造文献、给出处方或把表型写成病理确诊。
- 与已声明物种不同的案例。

## 执行顺序

1. `plant_vision` — 药用植物图片的异常表型观察与区域定位。
2. `growth_risk` — 带单位环境测量的因素核对与缺失上下文提示。
3. `herbal_knowledge` — 本地语料检索，返回文献片段、位置与证据 ID。
4. `evidence_fusion` — 确定性连接多来源证据，保留缺失项与未连接证据。

## 拒绝策略

- 证据策略：`every_claim_requires_evidence`。
- 拒绝策略：`insufficient_evidence`。
- 任一来源缺失时记入 missing_inputs，不借用其他案例结果。
- 任一来源案例 ID 或物种不一致时拒绝融合。

## 执行

```bash
python scripts/run.py --input examples/request.json --mode fixture \
    --public-key /trusted/publisher.public.pem
```

行为用例见 evals/evals.json；权限与依赖见 skill-card.md。
