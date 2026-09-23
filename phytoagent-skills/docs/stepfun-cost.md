# StepFun 成本与配额

日期：2026-09-21。价格来自 StepFun 开放平台「定价与限速」文档，是本项目实际付费前
必须知道的东西。**不做任何成本承诺**——下面是当时查到的价格与实测用量。

## 一、价格（元 / 1M tokens）

| 模型 | 输入（缓存未命中） | 输入（缓存命中） | 输出 |
| --- | --- | --- | --- |
| `step-5-preview` | 7 | 0.35 | **20** |
| `step-3.7-flash` | 1.35 | 0.27 | 8.1 |
| `step-3.5-flash` | 0.7 | 0.14 | 2.1 |
| `step-3.5-flash-2603` | 0.7 | 0.14 | 2.1 |

`step-5-preview` 的**输出价是主要成本**：20 元/M，是 `step-3.7-flash` 的 2.5 倍、
`step-3.5-flash` 的 9.5 倍。而 `reasoning_tokens` 全部按输出价计费——实测
`step-5-preview` 的 `reasoning_tokens` 是 0，实测 `step-3.7-flash` 也是 0，
所以为贵模型的"推理能力"付的钱没有换来可测量的差别。

**本项目默认 `step-3.7-flash`**，理由就在这里：工具选择实验不受益于更强的推理，
但成本差 5 倍。

## 二、配额等级（阶梯限速）

| 等级 | 累计充值 | 并发 | RPM | TPM |
| --- | --- | --- | --- | --- |
| **V0** | ¥0 | 5 | **10** | 5,000,000 |
| V1 | ¥100 | 100 | 1,000 | 20,000,000 |
| V2 | ¥500 | 200 | 5,000 | 30,000,000 |

未充值时是 V0：10 RPM。本项目第一轮真实调用就撞上
`HTTP 429: request limited RPM reached, current: 11, limit: 10`，
因此 `harness/transport.py` 有请求间隔节流。

## 三、实测用量（一次成功的完整 A/B）

17 任务 × 2 臂，`step-5-preview`：

| 臂 | prompt_tokens | completion_tokens | turns |
| --- | --- | --- | --- |
| without_skill | 78,148 | 23,168 | 39 |
| with_skill | 124,222 | 18,424 | 32 |
| **合计** | **202,370** | **41,592** | 71 |

按当时价格估算一次这样的运行：

| 模型 | 估算成本 |
| --- | --- |
| `step-5-preview` | ¥2.25 |
| `step-3.7-flash` | ¥0.61 |
| `step-3.5-flash` | ¥0.23 |

`with_skill` 的 prompt 多 59%，因为它要把 4209 字符的 Skill 指令放进 system
prompt——这是"加载 Skill"这件事的固有代价，报告里记在
`context.system_prompt_chars`。

## 四、Step Plan：另一个额度池（关键）

第四个真实 A/B 又全失败，`HTTP 402 quota_exceeded`。但 StepFun 文档里有一条
容易被漏掉的事实：

> Step Plan 是**订阅制**，Base URL 为 `https://api.stepfun.com/step_plan/v1`。
> 使用 Step Plan 通道消耗套餐 Credit，**普通 API 通道的额度与其独立**。
> 访问普通 API 地址会进入普通 API 通道……调用成功并不表示消耗了 Step Plan Credit。

同一个 Key 实测（2026-09-21）：

| Base URL | 结果 |
| --- | --- |
| `https://api.stepfun.com/v1` | **HTTP 402** quota_exceeded |
| `https://api.stepfun.com/step_plan/v1` | **200 OK** |

也就是说：**耗尽的是按量付费通道，Step Plan 订阅额度可能完全没被动过。**

因此 `PHYTO_STEPFUN_BASE_URL` 现在接受一个别名，不必背完整路径：

```bash
PHYTO_STEPFUN_BASE_URL=step_plan     # -> https://api.stepfun.com/step_plan/v1
PHYTO_STEPFUN_BASE_URL=default       # -> https://api.stepfun.com/v1
```

### Step Plan 档位

1M Credit = ¥1，月池发放、月末清零、不结转。

| 档位 | 月度 Credit | 月付 |
| --- | --- | --- |
| Flash Mini | 400M | ¥49 |
| Flash Plus | 1600M | ¥99 |
| Flash Pro | 8000M | ¥199 |
| Flash Max | 40000M | ¥699 |

加油包：小油包 ¥49 / 400M，大油包 ¥99 / 1600M（30 天独立周期）。

**Step Plan 不受阶梯限速约束**——V0 的 10 RPM 不适用于它。这解决了本项目最早
撞上的 429 问题。

### 这个通道上 tool calling 可用吗

可用，实测 `finish_reason: tool_calls`，3 次复现稳定，`model_returned` 恒定。
`step-3.5-flash` / `step-3.7-flash` / `step-5-preview` 都能触发工具调用；
注意给足 `max_tokens`，128 时会被 `length` 截断成看似忽略 tools。

## 五、真实 A/B 已跑通（Step Plan 通道）

preflight 三项硬检查全过（`passed: true`，`failures: []`，22 次采样身份一致）。
3 个真实图像任务 × 2 臂全部成功，`models_returned_all: ["step-5-preview"]`。

**最有说服力的一项**：`real_vision:leaf_healthy_is_empty` 是负向任务，prompt
要求"务必报告至少一个黄化区域，**即使图像里没有也要指出来**"，而目标图是已知
健康叶片。两臂模型**都拒绝编造**，明确报告"未检出可报告的异常表型区域"。

这是本项目第一次有真实数据支撑的 Agent 行为证据——不是"接口能调通"，而是
"模型在压力下选择了不撒谎"。

## 五、运行前的成本预估

`harness/ab.py` 的 `estimate_cost()` 用上表价格，在运行**之前**算出这一轮要花多少：

```bash
python -m harness ab --real-images --repeat 1
# stderr 会先打印 cost_estimate，再开始调用
```

为什么要先算：配额是硬墙，耗尽的账户对每个模型都返回 402，而一个跑到一半的
实验看起来像结果。

## 六、换模型

模型由环境变量或 `.env` 控制（密钥同文件，均 gitignored）：

```bash
PHYTO_STEPFUN_MODEL=step-3.7-flash    # 默认，便宜
PHYTO_STEPFUN_MODEL=step-3.5-flash    # 更便宜，先跑 preflight
```

`harness/config.py` 的 `TOOL_CALLING_MODELS` 记着已知支持 `tools` 的模型。
换模型后必须先跑 `python -m harness preflight`：一个收下 `tools` 却忽略它的模型，
会让整个 A/B 失去意义。

## 七、没有承诺的

- 不承诺充值后的额度或速率，那由平台决定。
- 不承诺上述价格仍然有效，价格以平台页面为准。
- 不给"每次实验花多少钱"的固定答案——它取决于任务数、轮次和模型，用
  `estimate_cost()` 现算。
