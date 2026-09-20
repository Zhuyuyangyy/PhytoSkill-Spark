---
name: herbal-knowledge
description: "检索与药用植物问题有关的可核实资料并返回证据ID。适用于本草及栽培证据查询；图像定位、无来源断言或面向患者的处方不使用。"
license: NOASSERTION
---

# herbal-knowledge

封装TCM-Mind-RAG的证据接口，保留来源、内容、适用范围及检索分数。

1. 确认检索问题、物种与证据用途；临床用药需求不进入植物研判流程。
2. 参照 [契约说明](references/contract.md) 返回证据ID及来源，检索不到时返回空证据。
3. 本版fixture只返回明确标注的合成条目，不能将其引用为真实文献。
4. 将证据及限制交回调用方，不自行规划其他Skill或补写虚构出处。

## 执行

依赖已安装的 phytoagent-skills SDK 0.2.x。从本Skill目录执行：

```bash
python scripts/run.py --input examples/request.json --mode fixture --public-key /trusted/publisher.public.pem
```

开发调试可显式改用 `--allow-unsigned`；不能把该状态称为签名验证通过。
本版 `live` / `replay` 会明确失败，不能回退fixture回答真实输入。

## 边界

本草药性记载不能直接证明叶片病因或药效下降；相似度分数不是事实确定性；必须区分真实文献与合成样例。

权限、依赖与许可见 [Skill Card](skill-card.md)。行为用例见 [evals](evals/evals.json)，实测状态见 [BENCHMARK](BENCHMARK.md)。
