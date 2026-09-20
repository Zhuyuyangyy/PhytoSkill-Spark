# AgentShield · 编译门禁运行手册

## 入口

```python
from shield.gate import gate_package, gate_many, repair_items

result = gate_package("skills/huangqi-health-assessment",
                      project_signature_key=".trust/publisher.public.pem")
if result.passed:
    ...  # 可以进 Registry
else:
    for item in repair_items(result):
        print(item)
```

CLI 侧没有单独入口：`python -m demo.phytoforge_demo` 会在第三幕完整跑一遍门禁并打印六项结果。

## 六项检查

| 检查 | 通过条件 | 失败含义 |
| --- | --- | --- |
| `manifest_integrity` | 文件清单 SHA-256 与元数据哈希都匹配 | 包被改动过，需重新封包 |
| `schema_conformance` | 声明的 input/output schema 可解析；有 `skill_spec.json` 时名称与包一致 | 契约不可用 |
| `permission_least_privilege` | `network: deny`；只有白名单只读权限；无通配符；能力在已审核集合内 | 权限过大 |
| `negative_eval_coverage` | 有正向、缺参数、负向三类用例 | 无法证明失败路径 |
| `hidden_instruction_scan` | 无危险模式、隐藏指令、外传意图、凭据读取 | 包可能有害 |
| `project_signature` | 包外固定公钥验通过 | 来源不可信 |

任一项非 `passed` → `quarantined = True`。

## 一个刻意设计：`not_run` 不算通过

未提供签名密钥时，`project_signature` 报 `not_run`，包仍被隔离。理由很简单：**没有做过的事不能记成通过**。这也意味着 CI 里不带密钥跑门禁，结果必然是隔离——这是期望行为，不是误报。

`agentshield-audit` Skill 的顶层裁决遵循同一原则：任一项检查是 `not_run` 时判 `INCOMPLETE`、`trust_level` 降到 `LIMITED`，并在 limitations 里列出未执行的检查，不判 `PASS` / `SUPPORTED`。只降裁决而留着 `SUPPORTED` 会读成背书。

## 门禁与 Compiler 独立

`gate_package` 从磁盘重读已写入的包：重新校验清单、重新解析 schema、重新扫描文本。Compiler 无法给自己的输出打勾，一个写错了的 Compiler 也不会让坏包通过。

## 两种包作者风格

| 风格 | 用例位置 | 负向要求 |
| --- | --- | --- |
| Compiler 产物 | `evals/evals.json` 的 `cases` | SkillSpec `negative_evals` 须含 `missing_input` / `unsupported_claim` / `permission_violation` |
| 手工包 | `evals/evals.json` 的 `contract_cases` + `agent_cases` | 用例 id 或 agent kind 体现正向 / 缺参数 / 负向 |

`repair_items()` 对每种失败给出可操作指令，例如：

```
Remove undeclared, wildcard or write permissions; keep network: deny.
Add positive, missing-parameter and negative-trigger cases to evals/evals.json.
Remove subprocess/eval/exec/network code and any hidden instructions.
Re-seal the package: python -m scripts.seal_skills.
Sign the sealed package with the project key held outside the package.
```

## 拦截模式

`hidden_instruction_scan` 拒绝这些模式（在包内任何文本文件中）：

`subprocess`、`os.system` / `os.popen` / `os.exec*`、`eval(` / `exec(`、`socket` / `urllib` / `requests` / `httpx` / `http.client`、`__import__` / `importlib.import_module`、`bash -c` / `powershell` / `cmd.exe` / `/bin/sh`、`ignore previous instructions` 类隐藏指令、上传到外部 URL 的外传意图、`api_key` / `secret` / `token` / `password` 赋值。

`.pyc` / `.pyo` / 图片 / zip / whl 不扫描（二进制，且 Manifest 已覆盖其哈希）。

## 这不是隔离层

门禁是**策略层**。它检查声明与文本模式，不提供执行隔离。Python 仍运行在本进程，无沙箱、无并发文件修改隔离、无强制超时。签名证明来源和完整性，不证明行为安全。
