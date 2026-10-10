# 评测协议与选择性指标

本文规定 PhytoForge 怎么回答一个问题：**这个模型到底好不好？**

在此之前，项目只能报 precision / recall / F1。这不够——不是因为算错了，而是因为
**这三个数看不见弃权**。Dev-30 的 `live` 子集报出 `precision = 1.000`，而模型在
15 张图里只回答了 3 张。那个数不是错的，是不可解释的。

本层做三件事：把弃权变成被报告的量；把「冻结什么」变成机器可校验的指纹；
把「一次性 holdout」变成由账本强制执行、而不是靠自觉的规则。

---

## 一、指标集

以图像为单位、标签集合为取值。对第 i 张图，模型给出预测集 `P_i`，标注给出参考集 `A_i`。

术语先钉死，否则报告会漂：

| 术语 | 定义 |
| --- | --- |
| **answered**（已答） | 模型至少给出一个标签 |
| **abstained**（弃权） | 模型一个都没给，**且**参考集非空 |
| **coverage** | answered / annotated |
| **abstention_rate** | abstained / annotated |

注意 answered 与 abstained 不是互补的：两边都为空的图（模型安静、标注也认为没东西）
既不算已答、也不算弃权——它是对的沉默。

### 报什么

| 指标 | 含义 |
| --- | --- |
| precision / recall / f1 | 微平均（micro），全样本 |
| **coverage** | 模型回答了多大比例 |
| **abstention_rate** | 该答而没答的比例 |
| **selective_precision / selective_recall** | 只在「已答」子集上算 |
| **risk_at_coverage** | `1 − selective_precision`，用户在已答样本上实际遇到的错误率 |
| **per-label P/R/F1 + support** | 每个标签各自的混淆计数 |
| **macro P/R/F1** | 只在 support > 0 的标签上平均 |

### 两条必须写进文档的性质

**1. 弃权只会损失 recall，永远不会损失 precision。**
空预测不可能产生 false positive。所以一个随便弃权的模型会报出「高 precision、
低 coverage」——而只看 precision 完全看不出来。这正是 `risk_at_coverage` 存在的理由：
把 precision 和 coverage 绑在一起报。

**2. selective_recall 与 recall 的差，就是弃权的代价。**
`selective_recall` 把因弃权而丢掉的标签排除在分母之外，因此是乐观的。
它必须与 `recall` 并列读，不能替代。Dev-30 上：`selective_recall = 0.4074`，
`recall = 0.1864` —— 弃权吃掉的比留下的还多。

### 两条宏平均的约定

- 参考集从未使用的标签（support = 0）**排除**在 macro 之外，并列入
  `labels_without_support` 以便可见。把它当 0 会惩罚模型没被要求回答的东西。
- 有 support 但模型从未预测的标签**记为 0，不跳过**。跳过会让模型通过「悄悄忽略最难的
  标签」抬高 macro 分数——那恰好是本层要暴露的失效模式。
- F1 由原始计数算出，**不由四舍五入后的 P、R 推出**。Dev-30 的计数
  (tp=11, fp=9, fn=48)：精确值 22/79 = 0.2785；用已舍入的 P、R 会得到 0.2784，
  与冻结基线在第 4 位小数上不一致。

### Risk–Coverage 曲线

按每条样本的置信度降序「逐步作答」，在覆盖率 k/n 处算错误率。置信度有效时，
曲线会显示用覆盖率换准确率；置信度是噪声时不会。

**当前不可计算**：Dev-30 标注文件没有 `model_confidence` 字段。代码已就绪
（`risk_coverage_curve`），缺的是数据——这是数据缺口，报告如实标注为缺口，
不编造排序。补法是给 worksheet 增加每图置信度（模型报告的最高 `model_score`）。

---

## 二、数据集角色

三种角色，区别是全部意义所在：

| 角色 | 用途 | 引用规则 |
| --- | --- | --- |
| `calibration` | 调 prompt | 只能写作「在 X 上的校准结果」，**不得**称为验证 |
| `development` | 比较设计选择 | 同上 |
| `holdout` | **只跑一次** | 看过结果之后再改任何东西，该次即作废 |

角色必须互斥：同一张图不能既在 calibration 又在 holdout。`DatasetSplit` 在构造时
就拒绝重叠——一张被调 prompt 时看过的图，不可能同时是 holdout。

分配必须**显式**：`split_entries(entries, holdout=(...), development=(...))`。
用一个没人记录下来的随机种子分配 holdout，就是 holdout 不再是 holdout 的方式。

---

## 三、冻结指纹

一次 holdout 要有意义，前提是「除了数据之外什么都没变」。冻结把这些东西哈希成一个
`freeze_id`：

| 字段 | 来源 | 为什么这样取 |
| --- | --- | --- |
| `prompt_hash` | 两个模式**渲染后的** prompt 文本 | 模型看到的就是它；纯空白重构不改文本，不应使冻结失效 |
| `ontology_hash` | 词表与守卫标记，**含顺序** | 药材 prompt 按 tuple 顺序列出表型，重排就是真实变更 |
| `parser_hash` | `vision/parser.py` **源码** | 行为即代码，没有更便宜且忠实的代理 |
| `scorer_hash` | `evaluation/metrics.py` **源码** | 同上 |
| `model` / `quantization` | 调用方传入 | 运行时配置；同代码换量化就是另一个实验 |

源码哈希的代价：改一行注释也会改指纹。**这是安全的方向**——过度失效只是浪费一次运行，
失效不足则会静默产出一个没人能辩护的数字。

```bash
python -m evaluation freeze --model modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest --quantization Q4_K_M
```

---

## 四、一次性 holdout 由账本强制

规则：**同一个 `freeze_id` 的 holdout 只允许评估一次。**

靠自觉不行，因为「这个指纹跑过了」是关于*历史*的事实，单个进程无从知道。
所以落盘：

```text
artifacts/evaluations/holdout-ledger.json
```

`HoldoutLedger.guard(freeze)` 在已存在同 `freeze_id` 记录时抛 `HoldoutAlreadyRun`。
逃生口是 `supersede=True`——刻意做成必须显式传参的独立开关，因为「看过结果再重跑」
正是本协议唯一禁止的事，所以必须点名要求，并在记录里留下它取代了哪一次。

```bash
# 第一次：正常记录
python -m evaluation holdout holdout-30.json --model M --quantization Q4_K_M \
    --holdout img_01 img_02 ... --note "blind holdout, first and only run"

# 第二次：会被拒绝（退出码 2）
python -m evaluation holdout holdout-30.json --model M --quantization Q4_K_M
```

---

## 五、标注协议

单人标注只能叫 **human reference annotations**，不能叫 gold standard。在两个人独立标注
同一批图之前，任何准确率都无法与「一个人的判断」分开。

### 双人标注

1. 两人**独立**标注同一批图，互不可见对方的标签。
2. 跑一致性：

```bash
python -m evaluation agree annotator_a.json annotator_b.json
```

3. 输出每个标签的 **Cohen's kappa** 与原始一致率，**并列**报告。

为什么 kappa 不能省：原始一致率会抬高稀有标签。一个只在 10% 图上出现的标签，
两个永远说「没有」的标注者一致率 90%，却什么都没一致。kappa 扣掉随机一致成分；
没有方差时（双方都恒定为同一值）kappa 无定义，返回 `None`——**报「无法测量」，
不报 0**，那是两个不同的结论。

### 分歧裁决

`resolution_queue` 列出两人不一致的图。每一条必须记录一个裁决：

| 裁决 | 含义 |
| --- | --- |
| `accept_reference` | 采信第一位 |
| `accept_second` | 采信第二位 |
| `accept_both` | 取并集（两人各看到一部分） |
| `drop` | 取交集（只有两人都同意才保留） |

`adjudicate` 拒绝为未裁决的分歧猜一个默认值——悄悄选一个人，就是让分歧在没被解决的
情况下消失。

---

## 六、当前状态（诚实版）

| 项 | 状态 |
| --- | --- |
| 选择性指标（coverage / abstention / selective / macro） | **已有**，见下 |
| 冻结指纹 | **已有** |
| 一次性 holdout 账本 | **已有** |
| 标注者间一致性工具 | **已有**，尚无第二位标注者 ⇒ 未测 |
| **blind holdout 数据集** | **没有**。这是最大的缺口，且只能由人建立 |
| 每图置信度 ⇒ risk–coverage 曲线 | **缺数据**，代码就绪 |
| 双人标注 | 没有 |

Dev-30（calibration 集）当前数字：

```text
annotated 30   coverage 0.3667   abstention_rate 0.4000
precision 0.55   recall 0.1864   f1 0.2785      ← 与冻结基线一致
selective_recall 0.4074          risk_at_coverage 0.45
macro_f1 0.1552                  ← 远低于 micro f1：有标签被整类漏掉

herb  coverage 0.5333  precision 0.4706  recall 0.1905  macro_f1 0.1391
live  coverage 0.2000  precision 1.0000  recall 0.1765  macro_f1 0.1818
      ↑ 只回答了 3/15。旧报告里那个「precision = 1.0」的全部真相在此
```

**这些是校准数据上的数字，不是验证结果。** 在 blind holdout 建立之前，本项目不得
声称任何泛化性能。

---

## 七、跑法

```bash
python -m evaluation report [annotations.json]        # 指标，整体 + 分模式
python -m evaluation split  [annotations.json] --holdout ... --development ...
python -m evaluation freeze --model M --quantization Q
python -m evaluation holdout annotations.json --model M --quantization Q --holdout ...
python -m evaluation agree  annotator_a.json annotator_b.json
```

`report` 写 `artifacts/evaluations/dev30-report.json`；`holdout` 写
`artifacts/evaluations/holdout-run.json` 并落账本。全部离线，不调用模型。
