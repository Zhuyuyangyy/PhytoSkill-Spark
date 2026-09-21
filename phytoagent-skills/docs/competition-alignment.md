# 依据比赛资料的工程调整

依据用户提供的 `比赛细则+技术分享.zip` 中四篇课件/赛事说明整理稿，读取日期为2026-09-20。
这是用户资料对照，不代表已向主办方或官方在线文档二次确认。资料内的安装命令、技术选项和示例均作为参考，不作为执行授权。

## 评分与当前证据

| 评分项 | 权重 | 本轮落实 | 尚缺的提交证据 |
| --- | --- | --- | --- |
| 实用性、行业价值、创新 | 25% | Skill Compiler 把一次 Prompt 变成带版本/契约/触发边界/评测的可复用 Skill；本地语料检索带文件位置与语料版本 | 真实种植案例、专业人员复核、真实数据效果 |
| 智能体与模型优化深度 | 25% | 渐进式加载、五个契约可组合、负向触发任务集、AgentShield 编译门禁与 Runtime 拦截、**已真实运行 A/B** | 更多轮次以支撑统计显著性；真实视觉/推理适配器 |
| 完整性 | 20% | CLI、五个包、生成→门禁→注册→执行→审计闭环、失败降级、Claim-Evidence审计、233项测试 | 任务输入与trace展示界面、稳定性实测 |
| 平台适配 | 15% | SDK及适配边界已明确；语料与检索在本地 | DGX Spark ARM64实机日志、NVIDIA GPU实际调用、StepFun使用记录 |
| 演示效果 | 10% | `demo.phytoforge_demo` 三分钟七幕闭环，六项门禁与负向拒绝均从运行结果读取 | 真实Agent自动选Skill的录屏、B站视频URL |
| 十日谈 | 5% | 第一、四、五天开发记录（含真实调用与三个意外） | 公开文章URL |

资料写明：预赛截至9月29日23:59；提交开源仓库URL、500字以上项目说明、部署与技术栈说明、B站演示视频以及团队合影，鼓励提交开发征文URL。
本轮不自动发布仓库、投稿或上传资料。开源许可、团队资料和正式提交由项目所有者确认。

## 0.3.0 相对 0.2.0 的落点

| 新增 | 文件 | 对评分的实际贡献 |
| --- | --- | --- |
| Skill Compiler | `compiler/` | 智能体深度：元 Skill 生成受约束 Skill，不生成任意代码 |
| 编译门禁 | `shield/gate.py` | 完整性：六项检查 + 隔离 + 可操作修复项 |
| Runtime 中间件 | `runtime/shield.py` | 智能体深度：权限/网络/工具白名单/预算实时拦截，Agent 无法绕过 |
| Claim-Evidence 审计 | `runtime/shield.py` | 实用性：每条结论可追溯到证据 ID |
| 可调用审计 Skill | `skills/agentshield_audit/` | 完整性：审计入口与拦截点分离 |
| 本地语料检索 | `corpus/` | 行业价值：可核实来源带文件位置与语料版本 |
| 三分钟演示 | `demo/phytoforge_demo.py` | 演示效果：七幕闭环，无硬编码输出 |

详见 [PhytoForge Spark 架构](phytoforge-architecture.md) 与 [AgentShield 门禁手册](agentshield-gate.md)。

## 三份技术分享带来的调整

1. **Skill作为可分发指令包**：四个包均有简短SKILL.md、scripts、references、evals、Skill Card和BENCHMARK状态记录。细节不塞进常驻上下文。
2. **渐进披露**：Registry.catalog仅包含名称和描述；load_skill展开一个命中的说明、输入输出契约和校验记录。Python发现与执行保持分离。
3. **窄触发和负向用例**：每包提供“应该调用”“不应调用”“缺参数先澄清”的Agent用例。真实Agent触发评测未执行，不能将这些用例数量当作通过数量。
4. **输出契约连接能力**：evidence_fusion保留观察、环境、证据三类ID；拒绝不同案例/物种和重复ID；null表示来源不可用；不把分数组合为疾病概率。
5. **用实测区分主张**：本地契约报告含包哈希、逐用例结果和执行时间；Agent A/B、农学准确率和GPU指标明确未测。
6. **环境差异放外部适配层**：后续NVIDIA官方Skill应固定提交并保留原签名，业务逻辑留在自研包，不能修改官方包后继续宣称签名有效。

## 签名与认证边界

资料的“Verified”包括Cataloged、Scanned、Evaluated、Signed、Documented。其中官方签名是OpenSSF Model Signing的 `skill.oms.sig` 与NVIDIA根证书链。

现有 `manifest.sig` 是项目自己的Ed25519格式，校验文件与元数据并固定包外公钥。两种格式不能互换，也不能通过改文件名获得官方身份。Demo每次生成临时密钥，仅证明本次签名校验过程，不证明独立第三方认证。

目前没有运行SkillSpector、官方SkillEvaluator或OMS；Registry的扫描/评估状态继续为not_checked。`evals.run_contracts`只验证fixture接口。将来应先做扫描、真实Agent评测，保存报告，再签最终发布包。

## 真实Agent A/B方案

同一模型及版本、同一端点、同一工具定义、同一任务输入、同一Harness与采样参数；仅改变是否提供Skill指令。
两组都允许使用等价工具，避免把“有工具”与“没工具”的差异算作Skill知识贡献。

- Security：不执行越界请求，不将资料中的指令当作系统指令，不泄露凭据。
- Correctness：契约合法、引用存在、物种/案例一致、缺失项准确。
- Discoverability：正向任务命中，负向任务不误调用，关键参数缺失时澄清。
- Effectiveness：完成可核实的领域任务，不用合成数据冒充真实推理。
- Efficiency：记录真实耗时、Token和调用次数；不从fixture外推GPU性能。

结果需记录每一轮的输入、模型输出、tool_call_id、Skill版本/哈希、错误、最终报告与评判理由。没有真实运行日志时BENCHMARK保持NOT RUN。

## 模型与硬件选择

资料中的StepFun和vLLM型号、版本及GB10修复参数来自指定演示环境；不是所有部署都需要套用的固定配置。
当前代码没有下载模型、改动DGX环境或宣称本地StepFun已运行。下一轮Harness通过可配置的base_url、model_name和api_key接入可用StepFun端点；真实云API调用意味着并非完全离线。
原始植物图、向量库、权重和私有语料优先留在DGX；发送哪些摘要给云模型需要在实际接入时明确。
