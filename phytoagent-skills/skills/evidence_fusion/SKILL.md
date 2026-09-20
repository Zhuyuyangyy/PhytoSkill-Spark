---
name: evidence-fusion
description: "按案例和物种合并已取得的视觉、环境与知识JSON，保留证据ID和缺失项。原始图片分析、凭空补证据、Agent规划或确定病因不使用。"
license: NOASSERTION
---

# evidence-fusion

确定性地连接已有结果；不是第二个Agent，也不调用语言模型。

1. 读取 [契约说明](references/contract.md)，确认每个来源属于同一 case_id 和 species。
2. 缺失的来源使用 null；不得借用其他案例结果补齐。
3. 执行融合并保留缺失项、未连接证据和 co_occurrence_only 标记。
4. 将JSON交回StepFun Harness，由StepFun生成最终报告；不在此Skill调度或重试其他Skill。

## 执行

依赖已安装的 phytoagent-skills SDK 0.2.x。从本Skill目录执行：

```bash
python scripts/run.py --input examples/request.json --mode fixture --public-key /trusted/publisher.public.pem
```

开发调试可显式改用 `--allow-unsigned`；不能把该状态称为签名验证通过。
本版 `live` / `replay` 会明确失败，不能回退fixture回答真实输入。

## 边界

不加权平均视觉分数和检索分数；不把共同出现写成因果关系；来源缺失降低完整性；输出不包含医学确定诊断。

权限、依赖与许可见 [Skill Card](skill-card.md)。行为用例见 [evals](evals/evals.json)，实测状态见 [BENCHMARK](BENCHMARK.md)。
