# plant-vision 契约

完整机器契约：[schema.json](../schema.json)；请求样例：[request.json](../examples/request.json)。输入输出都是JSON对象，禁止多余字段，环境单位写在字段名中。

所有结果携带 case_id、species 和 provenance；当前 data_origin 固定为 synthetic_fixture，model_called 固定为 false。
fixture只服务 fixture.json 中的精确请求；不同图片、物种、案例或测量不会得到同一份假结果。

模型分数不是病害概率；表型不能确诊病原；可见叶片面积比例不是药效变化比例。坐标顺序为 [left, top, right, bottom]，范围0–1。

## 后续真实接入

在 run() 接入已验证的本地 YOLO/分割适配器，记录权重版本、物种范围、实际图像尺寸与原始输出。
新增live适配器前需更新 supported_modes、契约、来源说明及评测，然后由发布者重新封包和签名。消费端不得自动修复签名。
