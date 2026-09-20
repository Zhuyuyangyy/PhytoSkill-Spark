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
| 工具返回的内容 | **仍是 fixture**：四个 Skill 跑在 `fixture` 模式，返回合成观察 |
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
├── tasks.py        从四个包的 agent_cases 生成任务集
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
python -m pytest -q                       # 先确认 129 项离线测试全绿
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
