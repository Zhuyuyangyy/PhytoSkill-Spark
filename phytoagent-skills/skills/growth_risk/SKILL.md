---
name: growth-risk
description: "分析已给出的药用植物环境测量与待复核因素。需要带单位的温度、空气相对湿度、土壤pH及可选N/P/K；只有图片或天气闲聊不使用。"
license: NOASSERTION
---

# growth-risk

整理可追溯的环境因素和缺失背景，避免未经核实的农学风险等级。

1. 确认物种、测量单位与时间；本接口温度为°C、空气湿度为%、N/P/K为mg/kg。
2. 未提供的读数填写 null，不能从叶片颜色猜测N/P/K或土壤水分。
3. 阅读 [契约说明](references/contract.md)，在支持的模式下执行。
4. 保留 observed_value、factor_id 和 requires_species_reference。没有核实的阈值时 risk_level 保持 undetermined。

## 执行

依赖已安装的 phytoagent-skills SDK 0.2.x。从本Skill目录执行：

```bash
python scripts/run.py --input examples/request.json --mode fixture --public-key /trusted/publisher.public.pem
```

开发调试可显式改用 `--allow-unsigned`；不能把该状态称为签名验证通过。
本版 `live` / `replay` 会明确失败，不能回退fixture回答真实输入。

## 边界

空气相对湿度不等于土壤水分；单点数据不能建立趋势。没有物种、生育期和可靠阈值时，不输出高风险等级。

权限、依赖与许可见 [Skill Card](skill-card.md)。行为用例见 [evals](evals/evals.json)，实测状态见 [BENCHMARK](BENCHMARK.md)。
