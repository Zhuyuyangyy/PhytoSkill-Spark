---
name: plant-vision
version: 0.2.0
owner: PhytoSkill-Spark contributors
license: NOASSERTION
license_status: owner_decision_required_before_public_release
deployment_geography: local-development-only
runtime: local_python
execution_modes: [fixture]
network_access: none
credentials: [external_pinned_public_key_for_signed_loading]
model_weights: none
nvidia_verified: false
security_scan: not_run
agent_ab_evaluation: not_run
---

# Skill Card

观察叶片黄化、斑点或萎蔫表型，输出可追溯的区域和观察ID。

## 数据与权限

脚本读取包内资源及调用方指定的JSON请求，向标准输出写JSON；不联网、不读取模型密钥。当前没有真实植物图片、私有种植记录或TCM语料。依赖 phytoagent-skills SDK 0.2.x，其依赖在项目 pyproject.toml 中声明。

## 风险与限制

模型分数不是病害概率；表型不能确诊病原；可见叶片面积比例不是药效变化比例。坐标顺序为 [left, top, right, bottom]，范围0–1。
当前结果仅用于接口验证。代码签名不代表农学准确性、执行隔离或NVIDIA官方认证。本项目未设定开源授权，发布前需由项目所有者确认代码和数据许可。

## 验证与来源

项目内签名文件为 manifest.sig，公钥固定在包外。NVIDIA资料所述 skill.oms.sig / OMS证书链是另一套验证，当前未接入，不能改名冒充。
评测范围见 BENCHMARK.md；契约用例和Agent正反例见 evals/evals.json。没有复制或修改官方NVIDIA Skill，也不声明使用其签名。
