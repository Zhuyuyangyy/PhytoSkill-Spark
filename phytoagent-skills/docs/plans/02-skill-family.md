# 第二轮：四个Skill与可验证组合

范围：四个可独立分发的Skill包、渐进加载、最小本地执行器、fixture组合、失败传播、包内契约用例。

验收命令：

```powershell
python -m pytest -q
python -m evals.run_contracts
python -m demo.run_demo
python -m demo.run_demo --simulate-failure herbal_knowledge --output artifacts/four-skills-degraded.json
```

生成源码ZIP前需校验所有Manifest、用例结果及解压运行；不自动重新封包被篡改的消费端资源。

第三轮仍为StepFun Agent Harness。当前脚本按固定顺序组合，只证明输出契约能连接，不能在演示中称为Agent自动规划。
