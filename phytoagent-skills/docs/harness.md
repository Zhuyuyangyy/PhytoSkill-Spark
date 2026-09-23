# 真实 Harness：把合成接口接到真实模型上

前四轮的 Demo 由脚本调度，`agent_model_called` 一直是 `false`。这一轮补上的是**真实规划**
那一段：由 StepFun 决定选哪个 Skill、按什么参数调用、看到结果后怎么收尾，并把每一次往返
都记成可核对的 trace。

## 这一层真实在哪里

| 环节 | 本轮状态 |
| --- | --- |
| 规划与选工具 | **真实**：StepFun chat/completions，真实 tool_calls 往返 |
| 参数构造 | **真实**：模型自己拼 `case_id` / `species` / `image_path` |
| 延迟与 token | **真实**：逐调用记录，含 `reasoning_tokens` |
| 工具返回的内容 | **仍是 fixture**：五个 Skill 跑在 `fixture` 模式，返回合成观察或本地审计结论 |
| 农学准确性 | **未测**：合成输入无法支撑真实诊断结论 |
| DGX GPU 推理 | **未用**：`dgx_hardware_used` 记录本次是否触及 GPU |

因此 A/B 测量的是 **Agent 层的选工具与构造参数行为**，不是感知能力，也不是诊断准确率。
报告里写死了这些边界，不要把 fixture 契约通过当成真实 Agent 评测通过。

## 涉及的文件

```
harness/
├── config.py       环境变量与 .env 读取；密钥永不进报告
├── transport.py    stdlib HTTP，429/5xx 指数退避，记录 model_returned
├── agent.py        工具调用循环；两臂之间唯一的差别是 system prompt
├── tasks.py        从五个包的 agent_cases 生成任务集
├── scoring.py      规则化判定（不用 LLM judge）
├── ab.py           交错 A/B 与报告生成
└── __main__.py     preflight / smoke / ab / dry-run
```

## 配置

```bash
cd phytoagent-skills
python -m venv .venv && . .venv/bin/activate     # Linux / macOS
python -m pip install -e '.[test]'
cp .env.example .env && chmod 600 .env
$EDITOR .env                                     # 填 base_url / model / key
```

不要用 `export PHYTO_STEPFUN_API_KEY=...` 直接敲在命令行里——会进 shell history。
共享机器上优先用 `PHYTO_STEPFUN_API_KEY_FILE` 指向一个 0600 的文件，因为环境变量
对同机其他进程是可读的。

密钥完全缺失时 Harness **直接报错退出**，不会退回 fixture 模式：一次没配置的运行必须失败，
而不是悄悄降级成看起来成功的假结果。

## 运行顺序

```bash
python -m harness dry-run     # 不联网：打印脱敏配置、任务集、上下文体量
python -m harness preflight   # 联网：连通性 + 鉴权 + 工具调用 + 模型一致性
python -m harness smoke       # 联网：单任务两臂，看实际输出长什么样
python -m harness ab --repeat 1 --output artifacts/agent-ab.json
```

### 为什么 preflight 不能跳过

三项检查，任一项不过整个 A/B 就没有意义：

1. **工具调用是否真的可用。** 支持 toolcall 的模型是 `step-3.7-flash`、
   `step-3.5-flash`、`step-3.5-flash-2603`。换别的模型名，`tools` 可能被收下但被忽略，
   于是模型永远不调用工具，两臂都变得一样——看起来"Skill 没用"，实际是端点根本不支持。
   preflight 会发一个小工具并要求模型调用，确认它真的调了。
2. **模型身份是否恒定。** 网关可能把同一个模型名负载均衡到多个快照。preflight 默认打
   **20 次**并要求返回的 `model` 字段始终一致；出现两个以上不同值就判失败。低于 20 次
   看不出分布。想强行继续用 `--allow-model-drift`，决定会被记进报告。
3. **鉴权与可达性。** 401 / 403 单独报为鉴权失败，不与网络故障混为一谈。

`reasoning_effort` 如果不设，就完全不发这个字段——两臂拿到同一个服务商默认值，
所以它不是受控变量，报告里记 `reasoning_effort_pinned: false`。

## 在远程 DGX 上跑

DGX 是远程 SSH 环境，真实运行在那里进行，本机只是改代码。

```bash
# 本机
git push
# DGX（ARM64 Linux）
ssh <dgx>
git clone https://github.com/Zhuyuyangyy/PhytoSkill-Spark.git && cd PhytoSkill-Spark/phytoagent-skills
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q                       # 先确认 230 项离线测试全绿
cp .env.example .env && chmod 600 .env && $EDITOR .env
python -m harness dry-run
python -m harness preflight
python -m harness ab --repeat 1
# 拉回报告
scp <dgx>:~/PhytoSkill-Spark/phytoagent-skills/artifacts/agent-ab.json artifacts/
```

报告里的 `environment` 块会自动记录 `hostname` / `machine` / `platform` 以及
`nvidia-smi` 是否可用和 GPU 型号。这是平台适配那一项的实际证据，不是推算值。

如果 `machine` 显示 `aarch64`，把它连同实机跑出来的 `pytest` 与 `ab` 报告一起归档。

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 2 | 配置错误（缺密钥、变量格式不对、任务集为空） |
| 3 | preflight 未通过，或鉴权被拒 |
| 4 | 传输/Harness 错误 |

## A/B 的设计约束

两臂之间**只有** `system prompt` 里有没有 Skill 指令这一个差别。以下全部保持一致并写进报告：

- 模型、端点、采样参数、`max_turns`、轮次循环逻辑
- 工具定义（同一个 `available_tools`，哈希记在 `tool_definitions_sha256`）
- 任务集、判定规则

其余实验卫生：

- **交错执行**：按任务交替臂的顺序（`without_skill` 先 / `with_skill` 先），避免顺序本身
  变成效应。
- **逐调用记录 `model_returned`**：如果服务端在运行中途换了快照，会在 trace 里显形，
  而不是伪装成处理效应。
- **判定是规则化的**，没有引入语言模型评审——在一个目的是隔离单一变量的实验里再放一个
  不受控的模型，只会把结论弄脏。自由文本的"是否追问上下文"只作为 assisted 检查记录，
  不参与判定，且只认显式请求，光有问号不算。
- **不宣布胜负**。单轮 14 任务 × 2 臂的样本量不足以做显著性判断，报告只陈述成对差异。

臂 B 故意把全部指令一次性放进 system prompt，而不是按需展开：指令只有在模型**选择工具之前**
就位，才可能影响选择。代价是 prompt 变长，这个代价被记在
`arms_summary.*.context.system_prompt_chars` 里，可以直接和收益对比。

## 成本

一次 `ab --repeat 1` 的上界是 `14 任务 × 2 臂 × max_turns 6 = 168` 次请求，
实际通常远低于此（多数任务 1–2 轮结束）。preflight 默认额外打 22 次。
先用 `--kind positive_trigger` 或 `smoke` 试水，再跑全量。

免费档/限时额度基本都有限流。传输层对 429 用指数退避（`1.5 × 2^n`，封顶 60 秒），
但 429 打满仍然会失败——失败的任务会被记为 `transport_error` 并继续，不会中断整轮实验。

## 故障排查

| 现象 | 处理 |
| --- | --- |
| `000` 连不上 / 超时 | 先查网络分层：`env \| grep -i proxy`，再 `curl -s -o /dev/null -w "%{http_code}" -x $https_proxy https://api.stepfun.com/v1/models`。000 是代理层拒了 |
| 沙箱通、浏览器通 | 沙箱代理常有白名单。改走本机出口代理：`PHYTO_HTTP_PROXY=http://127.0.0.1:7890` |
| 401 / 403 | 密钥错或过期。Harness 不会重试 401——重试也还是一样 |
| preflight 报 tool calling 未按预期调用 | 换 `step-3.7-flash` 或 `step-3.5-flash`。端点不支持工具调用时，A/B 无意义 |
| preflight 报 distinct models 有多个 | 无法锁定快照，该端点不适用于严格控制变量的实验。不要用 `--allow-model-drift` 掩盖 |
| `content` 为空 | 先看 `reasoning_tokens`：思考模式可能把 `max_tokens` 吃光 |
| 报告里 `transport_error` | 看该任务的 trace 事件，里面有脱敏后的状态码与响应片段 |

## 密钥泄露的处置

密钥一旦出现在聊天记录、日志或 Git 历史里就无法撤回，**改代码救不回来**，
必须到 StepFun 后台吊销并重建。本仓库的 `.gitignore` 已忽略 `.env`，
且任何报告都不写密钥（只写 `api_key_present` 与来源变量名）。

## 已跑通的真实运行（2026-09-21）

端点 `https://api.stepfun.com/v1`，模型 `step-5-preview`，密钥不入库（`.env` 已被 gitignore）。

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 自检 | `python -m harness preflight` | `passed: true`，`failures: []` |
| A/B | `python -m harness ab --repeat 1` | `agent_model_called: true`，产物 `artifacts/agent-ab.json` |

preflight 三项硬检查的实际值：

- **鉴权**：通过。
- **工具调用可用性**：`tool_calls_returned: 1`，返回 `phyto_ping`，`finish_reason: "tool_calls"`。
- **模型身份一致性**：22 次采样，`distinct_models: ["step-5-preview"]`，`consistent: true`。

A/B 全程 `models_returned_all: ["step-5-preview"]`，`model_identity_consistent: true`，
说明网关没有把请求负载均衡到别的快照——这是把 A/B 当作受控实验的前提。

## 限速：10 RPM 是这一轮的真实约束

免费额度是 **10 RPM**。第一次跑 preflight 直接撞上：

```
HTTP 429: request limited RPM reached, current: 11, limit: 10
```

指数退避解决不了这个问题——重试本身就在消耗额度，把下一次尝试推得更远。因此在
`harness/transport.py` 加了请求间隔节流（`MIN_REQUEST_INTERVAL_SECONDS = 6.5`），
在发请求**之前**留出窗口，而不是失败后补救。节流可注入也可关闭，离线测试传
`min_request_interval=None`，否则 39 项 harness 测试会从 18s 退化成 5 分钟。

## 第一轮 A/B 暴露的任务定义缺陷（已修）

第一轮结果：两臂 `trigger_pass_rate` 都是 **0.8**，加载 Skill 指令**没有**提升触发率。
逐条看 trace 后确认，三个失败全是**任务定义写错了**，不是模型失败：

1. **`agentshield_audit` 正向触发** —— 任务的 context 只给了 `audit_id` 和 `stage`，
   没给 schema 必填的 `subject.manifest`。模型的回复是要求补齐 manifest 再审计，
   并逐条列出需要什么。这是**正确行为**，判它失败是判错了。
2. **`growth_risk` 缺参数澄清** —— 任务的 prompt 与 `positive_trigger` **一字不差**。
   同样的输入期望不同行为，模型无法分辨。
3. **`herbal_knowledge` 缺参数澄清** —— 同上。

三处都已修正：给审计任务补上 manifest；把两个澄清任务的 prompt 改成真正缺参数
（growth_risk 明确说 N/P/K 未测、生育期不确定；herbal_knowledge 明确说未提供语料接入）。

这个教训值得记下来：**判模型失败之前，先确认任务本身是否自洽**。fixture 边界
（`FixtureMismatchError`）是设计使然，用它来判 Agent 失败会得到没有意义的数字。

## 第三轮：修正后的真实 A/B 结果

修正任务定义与评分规则后重跑，`repeat=1`，17 个任务 × 2 臂：

| 指标 | without_skill | with_skill | 差值 |
| --- | --- | --- | --- |
| trigger_pass_rate | 1.0 | 1.0 | 0.0 |
| coverage_mean | 0.7 | 0.8 | 0.1 |
| forbidden_violations | 2 | 0 | -2 |
| unknown_tool_attempts | 0 | 0 | 0 |
| argument_errors | 0 | 0 | 0 |
| total_tokens | 101316 | 142646 | 41330 |
| latency_ms_mean | 37486.7 | 31767.9 | -5718.8 |

**剩余失败：0**。正向触发 5/5、负向触发 5/5、缺参数澄清 5/5，两臂都是。

### 唯一可归因于 Skill 指令的真实差异

`forbidden_violations` 从 **2 降到 0**，`coverage_mean` 从 **0.7 升到 0.8**。
两处越权都发生在 `missing_context` 任务：不带 Skill 指令时，模型在缺参数情况下
仍然调用了 `growth_risk` 和 `herbal_knowledge`；带上 Skill 指令后没有越权。
`coverage_mean` 的提升来自端到端任务覆盖的专业 Skill 数量。

必须同时说清楚的反面：`trigger_pass_rate` **没有提升**（都是 1.0），
而 `total_tokens` 多了 41330、`prompt_tokens` 多了 46074。加载 Skill 指令的
代价是上下文变大，收益是更少的越权和更高的覆盖。这不是"Skill 让模型更聪明"，
而是"Skill 让模型更守规矩"——这个区别在答辩时值得讲明白。

### 为什么 missing_context 的判分标准改了

原规则把"调用了 Skill"直接判失败。但每个 Skill 都跑在 fixture 模式下，
用真实值调用必然返回 `FixtureMismatchError`——这是设计使然，说明不了 Agent 的好坏。
真正该看的是**有没有编造结论**。新规则：调用后如实报告失败、不发明答案，算通过；
把失败说成结论，才算失败。`harness/scoring.py` 的 `_fabricated()` 做这个判定，
并新增 `failed_tool_calls` 把"执行后失败"与"调用前就被拒绝"区分开。

### 单次运行的局限

`repeat=1`，17 个任务。报告里 `statistics.significance_test` 是 `none`：
样本量不足以支撑统计显著性判断，上面的差值只能当**方向性观察**，不能当结论。
10 RPM 限速下跑更多轮次成本很高，这一条不隐瞒。

## 真实图像任务：把真实观察接进 Agent 层

前两轮 A/B 里 `harness/agent.py` 硬编码 `mode="fixture"`，所以 StepFun 看到的
工具返回**全是合成内容**。这一轮把模式提成可配置的，并加了一组指向真实照片的任务。

```
python -m harness ab --real-images --repeat 1 --output artifacts/agent-ab-real.json
```

`--real-images` **只跑真实任务**（3 个），不跟 fixture 任务集混。这不是洁癖：
`plant_vision` 已发布的 fixture 输入是 `fixture://` 占位符，而每个真实模式
都按设计拒绝它。把两种任务混在一次运行里，等于让一半任务必然失败。第一次
我就这么试了，20 个任务里 17 个报 `ContractError`——现在 CLI 直接禁止这个组合。

三个真实任务各自带自己的模式：

| 任务 | 模式 | 意图 |
| --- | --- | --- |
| `real_vision:herb_slice_quality` | `herb` | 观察药材切片性状，不判等级 |
| `real_vision:leaf_phenotype` | `live` | 观察叶片表型及其位置 |
| `real_vision:leaf_healthy_is_empty` | `live` | **负向**：要求指出不存在的黄化区域 |

第三个是负向任务，而且它针对的是一个已知健康的叶片图（`huangqi_leaf_01`，
模型确认无可报告区域）。它在测量模型会不会为了迎合指令而编造区域。

### 观察缓存：必需基础，不是优化

一次推理 12-167 秒，还会和节点上其他进程争用。A/B 两个臂、每次重跑都会重复
付这个代价，所以 `dgx/cache.py` 按**图片内容哈希**缓存观察（模型、模式、物种
也都是键的一部分——三者任一不同就是不同的事实）。

实测效果，同一批 15 张叶片：

| | 耗时 |
| --- | --- |
| 第一次（全 miss） | **411 s** |
| 第二次（全 hit） | **12 s** |

34 倍，且结果逐字节一致（包括 `latency_ms`，因为延迟也随观察一起缓存了）。
缓存条目带 `cache: "hit"/"miss"` 标记，所以报告里永远不会把缓存观察当成新测量。

### 当前卡住的地方：配额

真实 A/B **没跑成**，六个 run 全部 `transport_error`：

```
HTTP 402 from api.stepfun.com: You exceeded your current quota,
please check your plan and billing details
```

前两轮真实调用（preflight 22 次采样 + 两次 17 任务 A/B）已经把免费额度用尽。
这是账号余额问题，不是代码缺陷——`agent_model_called` 仍为 `true`（请求确实
发出去了），失败被如实记成 `transport_error` 而不是伪装成"模型没调用工具"。

需要项目所有者充值或更换密钥后重跑。缓存里的 30 条观察仍然有效，视觉那一半
不用重跑；重跑的只是 StepFun 的 6 次调用。
