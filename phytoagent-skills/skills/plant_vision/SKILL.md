---
name: plant-vision
description: "药用植物图片的异常表型观察与区域定位。提供图片和物种后使用；纯知识问答、环境分析和人体疾病诊断不使用。"
license: NOASSERTION
---

# plant-vision

观察叶片黄化、斑点或萎蔫表型，输出可追溯的区域和观察ID。

1. 确认图片路径和物种；缺少任一项先澄清，不猜测植物名称。
2. 需要字段细节时读取 [契约说明](references/contract.md)。
3. 按输入对象选择 manifest 声明的模式：`fixture` 用于精确匹配的合成案例，`live` 用于田间植株/叶片，`herb` 用于干燥药材切片。真实模式需要可用的视觉后端，不会回退到 fixture。
4. 调用脚本，保留 observation_id、归一化坐标、面积分母、模型分数和来源标记。

## 执行

依赖已安装的 phytoagent-skills SDK 0.2.x。从本Skill目录执行：

```bash
python scripts/run.py --input examples/request.json --mode fixture --public-key /trusted/publisher.public.pem
```

开发调试可显式改用 `--allow-unsigned`；不能把该状态称为签名验证通过。
`--mode` 只接受本 Skill manifest 声明的执行模式；`replay` 是后端类型，不是此 Skill 的执行模式。

## 边界

模型分数不是病害概率；表型不能确诊病原；可见叶片面积比例不是药效变化比例。坐标顺序为 [left, top, right, bottom]，范围0–1。

权限、依赖与许可见 [Skill Card](skill-card.md)。行为用例见 [evals](evals/evals.json)，实测状态见 [BENCHMARK](BENCHMARK.md)。
