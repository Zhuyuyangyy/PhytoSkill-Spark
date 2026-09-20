# 第一轮：SDK + Registry

范围由用户提供的分轮实施要求确定。当前只实现 SDK、包校验、发现、Tool Schema 导出及其测试。

1. BaseSkill：metadata、execute、validate_input、validate_output，fixture/live/replay 显式能力声明。
2. Manifest：代码、Schema、SKILL.md 和其他资源的 SHA-256 清单；元数据哈希；拒绝新增、缺失、篡改资源。
3. Registry：扫描、校验、原子注册、输出 StepFun 兼容 tools；发现时不导入代码。
4. 信任：默认要求 Ed25519 签名，公钥由调用方在包外固定；开发可显式关闭，状态为 not_checked；不宣称 NVIDIA 官方 Verified。
5. 验收：契约、异常输入、输出拒绝、注册失败、篡改、签名、CLI 烟雾演示的实际测试。

后续轮次：第二轮四个 Skill；第三轮 StepFun Harness；第四轮完整 Demo、失败降级与端到端测试。

本轮不把临时测试 Skill 当作已完成的植物模型，不把本机测试当作 DGX Spark 实测。
