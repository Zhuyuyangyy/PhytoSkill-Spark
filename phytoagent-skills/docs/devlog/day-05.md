# 十日谈 · 第五天：第一次真的调用模型

日期：2026-09-21。内容为本地工程记录草稿，尚未发布到技术社区。

第四天结束时，仓库里所有 `agent_model_called` 都是 `false`。代码写得很整齐，但有一个
根本问题没回答：**这套东西在真实模型上到底表现如何？** 今天拿到了密钥，第一次真的调了。

## 三个意外，按代价排序

**一、模型名猜错了。** 用户给的是 `step5-preview`，端点上真实存在的是 `step-5-preview`。
第一次 preflight 直接 404。拉 `/models` 列表才发现差一个连字符。这件事很小，但说明
"用户给的字符串"和"端点接受的标识符"是两回事，得验。

**二、10 RPM 比我预期的更难缠。** 第二次 preflight 撞上 429，`current: 11, limit: 10`。
我原本以为指数退避能解决——错了。重试本身在消耗额度，把下一次尝试推得更远，是负反馈。
真正有效的是在发请求**之前**留窗口，于是加了 `MIN_REQUEST_INTERVAL_SECONDS = 6.5` 的节流。
副作用是 39 项 harness 测试从 18s 退化成 5 分钟，因为测试用脚本化 poster 瞬间返回，
却按真实时钟判等。修法是让节流可注入、可关闭，测试传 `min_request_interval=None`。

**三、第一次 A/B 的结果完全不能用。** 两臂 `trigger_pass_rate` 都是 0.8，Skill 指令
看起来毫无价值。逐条读 trace 才发现三个失败**全是我的任务定义写错了**：

1. 审计任务的 context 没给 schema 必填的 `subject.manifest`。模型的回复是要求补齐
   manifest 并逐条列出需要什么——这是**正确行为**，我判它失败是判错了。
2. 另外两个"缺参数澄清"任务的 prompt 与对应的 `positive_trigger` **一字不差**。
   同样的输入期望不同行为，模型没法分辨。

修正后重跑，正向 5/5、负向 5/5、澄清 5/5，**0 失败**。

## 更深的一层：评分规则本身也不对

`missing_context` 的原规则把"调用了 Skill"直接判失败。但每个 Skill 都跑在 fixture 模式，
用真实值调用必然 `FixtureMismatchError`——这是设计使然，说明不了 Agent 的好坏。
真正该问的是**有没有编造结论**。

改成：调用后如实报告失败、不发明答案，算通过；把失败说成结论，才算失败。
为此加了 `failed_tool_calls`，把"执行后失败"和"调用前就被拒绝"区分开。

这一条比上面三个意外更值得记：**fixture 边界会被误当成 Agent 失败**，如果不先想清楚
判分标准，跑出来的数字没有意义。

## 修正后的真实结果

17 任务 × 2 臂，`repeat=1`：

| 指标 | without_skill | with_skill | 差值 |
| --- | --- | --- | --- |
| trigger_pass_rate | 1.0 | 1.0 | 0 |
| coverage_mean | 0.7 | 0.8 | +0.1 |
| forbidden_violations | 2 | 0 | **-2** |
| total_tokens | 105425 量级 | 多 41330 | — |

可归因于 Skill 指令的差异是**越权从 2 降到 0**、覆盖从 0.7 升到 0.8。
但触发率没变，token 多了四万。诚实的说法是：**Skill 让模型更守规矩，没让模型更聪明**。
这个区别在答辩时该讲明白，而不是挑好看的数字说。

局限也写进报告了：`repeat=1`、17 个任务，`significance_test` 是 `none`，
样本量撑不起统计显著性，差值只能当方向性观察。10 RPM 下跑更多轮次成本很高。

## 可复现证据

`artifacts/agent-ab.json`（真实 A/B）、`artifacts/pytest-v0.3-results.xml`（233 passed）。
密钥在 gitignored `.env`，未入库；`git status` 干净。

## 下一步

仍然是 DGX Spark 实机与真实视觉模型。当前 A/B 测的是 Agent 层的选工具与构造参数行为，
不是感知能力；工具返回内容还是 fixture，农学准确性未测。
