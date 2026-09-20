# evidence-fusion 契约

完整机器契约：[schema.json](../schema.json)；请求样例：[request.json](../examples/request.json)。输入输出都是JSON对象，禁止多余字段，环境单位写在字段名中。

所有结果携带 case_id、species 和 provenance；当前 data_origin 固定为 synthetic_fixture，model_called 固定为 false。
输入内嵌前三个Skill的输出契约快照，独立分发无需跨包读取Schema；测试验证快照一致。null 表示该来源不可用。

不加权平均视觉分数和检索分数；不把共同出现写成因果关系；来源缺失降低完整性；输出不包含医学确定诊断。

## 后续真实接入

先保持确定性融合；真实来源接入时升级契约和测试，不增加新的规划模型。
新增live适配器前需更新 supported_modes、契约、来源说明及评测，然后由发布者重新封包和签名。消费端不得自动修复签名。
