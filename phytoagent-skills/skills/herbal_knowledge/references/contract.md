# herbal-knowledge 契约

完整机器契约：[schema.json](../schema.json)；请求样例：[request.json](../examples/request.json)。输入输出都是JSON对象，禁止多余字段，环境单位写在字段名中。

所有结果携带 case_id、species 和 provenance；当前 data_origin 固定为 synthetic_fixture，model_called 固定为 false。
fixture只服务 fixture.json 中的精确请求；不同图片、物种、案例或测量不会得到同一份假结果。

本草药性记载不能直接证明叶片病因或药效下降；相似度分数不是事实确定性；必须区分真实文献与合成样例。

## 后续真实接入

接入用户现有TCM-Mind-RAG，并增加文献位置、语料版本、检索配置及证据适用领域；使用真实来源后单独升级契约。
新增live适配器前需更新 supported_modes、契约、来源说明及评测，然后由发布者重新封包和签名。消费端不得自动修复签名。
