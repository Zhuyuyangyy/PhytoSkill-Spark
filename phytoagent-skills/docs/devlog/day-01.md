# 十日谈 · 第一天：把四个能力变成可验证的Skill包

日期：2026-09-20。内容为本地工程记录草稿，尚未发布到技术社区。

完成SDK/Registry后，对照比赛规则与三份技术分享，发现“接口可调用”与“Skill可复用”之间还需要补齐指令、边界、按需加载、评测和来源说明。

本次将PlantVision、GrowthRisk、HerbalKnowledge及EvidenceFusion拆成四个独立包，加入SKILL.md、脚本、参考契约、正反触发任务、Skill Card和Benchmark状态。
视觉、环境和知识目前使用精确匹配的合成案例；任意真实图片或未匹配读数不会复用假结果。融合采用确定性ID连接，知识调用失败会保留缺失项而不是生成新文献。

代码引入catalog与load_skill两级上下文接口，并在执行前重验包、模式和输入。发布源包中的文件由Manifest覆盖；临时演示签名证明完整性流程，不代表NVIDIA官方认证。

可复现证据由 `python -m pytest`、`python -m evals.run_contracts` 和 `python -m demo.run_demo` 生成，保存在artifacts。确切数量与耗时以本次输出为准。

下一步重点是StepFun真实调度与触发评测，再接现有YOLO和TCM-Mind-RAG，最后在DGX Spark上保留可核实的实际推理记录。当前没有产生模型效果、云API或GPU性能数据。
