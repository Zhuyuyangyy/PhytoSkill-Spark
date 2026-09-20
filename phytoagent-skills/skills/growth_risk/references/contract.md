# growth-risk 契约

完整机器契约：[schema.json](../schema.json)；请求样例：[request.json](../examples/request.json)。输入输出都是JSON对象，禁止多余字段，环境单位写在字段名中。

所有结果携带 case_id、species 和 provenance；当前 data_origin 固定为 synthetic_fixture，model_called 固定为 false。
fixture只服务 fixture.json 中的精确请求；不同图片、物种、案例或测量不会得到同一份假结果。

空气相对湿度不等于土壤水分；单点数据不能建立趋势。没有物种、生育期和可靠阈值时，不输出高风险等级。

## 后续真实接入

接入经确认且带来源的物种/生育期规则集；阈值、单位转换、缺失值策略与规则版本必须一起记录。
新增live适配器前需更新 supported_modes、契约、来源说明及评测，然后由发布者重新封包和签名。消费端不得自动修复签名。
